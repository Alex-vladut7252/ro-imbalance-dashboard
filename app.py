"""
Energy Prediction Platform — Romania (General)
Replicates daciaenergy.ro/prediction with live ENTSO-E + Transelectrica data.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from threading import Thread
import time

from flask import Flask, render_template, jsonify, request

from entsoe_client import EntsoeClient
from transelectrica_client import TranselectricaTracker
import transelectrica_newmarkets_api as damas
import mavir_client as mavir
import mavir_rtdw_client as mavir_rtdw
import forex_client
import tso_forecast_client
import database as db
from negative_price_predictor import NegativePricePredictor
import market_analysis
import countries

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize database
db.init_db()

# ENTSO-E token: from env (set ENTSOE_API_KEY as a deploy secret), otherwise from a
# gitignored local_config.py for dev. Never hardcode the token in a committed file.
ENTSOE_KEY = os.environ.get('ENTSOE_API_KEY', '')
if not ENTSOE_KEY:
    try:
        from local_config import ENTSOE_API_KEY as ENTSOE_KEY
    except Exception:
        ENTSOE_KEY = ''
entsoe = EntsoeClient(ENTSOE_KEY)
te_tracker = TranselectricaTracker(max_history=500)
predictor = NegativePricePredictor()

# Cache for ENTSO-E data (expensive calls) — keyed by date
# NOTE: the main caches below are Romania-only (used by background refresh threads).
# Non-RO countries route through `_foreign_entsoe_cache` for on-demand fetching.
_entsoe_cache = {}   # {date_str: {'data': ..., 'time': datetime}}
_actual_cache = {}   # {date_str: {'data': ..., 'time': datetime}} — XML API actual data
_dam_cache = {}      # {(country, date_str): (prices, time)}
_damas_cache = {}    # {(endpoint, date): {'data': ..., 'time': datetime}}
_foreign_entsoe_cache = {}  # {(country, date_str): {'data', 'time'}}  — HU and beyond


def _country_from_request():
    """Read the `?country=` query param, default to the system default."""
    c = request.args.get('country', countries.DEFAULT_COUNTRY)
    return c.upper() if countries.is_valid(c) else countries.DEFAULT_COUNTRY


def _get_foreign_entsoe_cached(country, date_str):
    """On-demand ENTSO-E forecast fetch for non-RO countries. Simple time-based cache."""
    now = datetime.now()
    key = (country, date_str)
    is_today = (date_str == _today())
    ttl = ENTSOE_CACHE_SECS if is_today else ENTSOE_HIST_CACHE_SECS
    if key in _foreign_entsoe_cache:
        entry = _foreign_entsoe_cache[key]
        if (now - entry['time']).total_seconds() < ttl:
            return entry['data']
    try:
        data = entsoe.get_forecasts(date_str, country=country)
        _foreign_entsoe_cache[key] = {'data': data, 'time': now}
        return data
    except Exception as e:
        log.error(f"ENTSO-E {country} forecast error for {date_str}: {e}")
        return _foreign_entsoe_cache.get(key, {}).get('data', {})

ENTSOE_CACHE_SECS = 900  # 15 min (background thread refreshes more often)
ENTSOE_HIST_CACHE_SECS = 3600  # 1 hour for historical dates
ACTUAL_CACHE_SECS = 300  # 5 min for actual generation — XML API rate limits aggressively
ACTUAL_HIST_CACHE_SECS = 3600  # 1 hour for historical
DAMAS_CACHE_SECS = 15  # 15s for live DAMAS data — fetch aggressively
DAMAS_HIST_CACHE_SECS = 3600  # 1 hour for historical


def _damas_time_range(date_str, country='RO'):
    """Convert YYYY-MM-DD to UTC time range for the given country's delivery day.
    Delivery day = local midnight to local midnight in the country's TZ."""
    from zoneinfo import ZoneInfo
    cfg = countries.get(country)
    local_tz = ZoneInfo(cfg['tz'])
    local_start = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=local_tz)
    local_end = local_start + timedelta(days=1)
    utc_start = local_start.astimezone(ZoneInfo('UTC'))
    utc_end = local_end.astimezone(ZoneInfo('UTC'))
    return utc_start.strftime('%Y-%m-%dT%H:%M:%S.000Z'), utc_end.strftime('%Y-%m-%dT%H:%M:%S.000Z')


def _entsoe_hu_imbalance_damas_shape(date_str):
    """Convert ENTSO-E A85 HU imbalance prices to DAMAS-shape records.

    Strict 1:1 mapping to what ENTSO-E's Transparency UI shows:
      - category A04 → estimatedPricePositiveImbalance (no fallback)
      - category A05 → estimatedPriceNegativeImbalance (no fallback)
    No collapse, no picking non-zero. If ENTSO-E shows 0 for A04 that MTU,
    we show 0 for the positive imbalance column — same as the UI.

    Values are returned in HUF/MWh (the native ENTSO-E currency for HU).
    EUR conversion will be re-added later once the user supplies an
    authoritative HUF/EUR rate source. Timestamps are Budapest-local with
    DST-aware offset so frontend keys align with other adapters.
    """
    raw = entsoe.get_imbalance_prices(date_str, country='HU')
    a04_map = raw.get('pos') or {}   # raw key is historical — holds A04 values
    a05_map = raw.get('neg') or {}   # holds A05 values
    if not (a04_map or a05_map):
        return []
    from zoneinfo import ZoneInfo
    hu_tz = ZoneInfo('Europe/Budapest')
    date_d = datetime.strptime(date_str, '%Y-%m-%d').date()
    # Live ECB HUF/EUR rate; fetched once per call, cached for an hour
    # inside forex_client so the 15 s dashboard refresh is cheap.
    fx = forex_client.get_huf_per_eur()
    huf_per_eur = fx['rate']
    all_hm = set(a04_map.keys()) | set(a05_map.keys())
    out = []
    for hm in sorted(all_hm):
        h, m = int(hm[:2]), int(hm[3:5])
        isp = h * 4 + m // 15 + 1
        a04 = a04_map.get(hm)
        a05 = a05_map.get(hm)
        # System-state tag (purely informational; not used for value picking):
        if (a04 is not None and a04 != 0) and (a05 is None or a05 == 0):
            state = 'LONG'
        elif (a05 is not None and a05 != 0) and (a04 is None or a04 == 0):
            state = 'SHORT'
        elif a04 is not None and a05 is not None and abs(a04 - a05) > 1e-6 \
                and a04 != 0 and a05 != 0:
            state = 'DUAL'
        else:
            state = 'SINGLE'
        pos_huf = round(a04, 4) if a04 is not None else None
        neg_huf = round(a05, 4) if a05 is not None else None
        pos_eur = round(pos_huf / huf_per_eur, 4) if pos_huf is not None else None
        neg_eur = round(neg_huf / huf_per_eur, 4) if neg_huf is not None else None
        start_local = datetime.combine(date_d, datetime.min.time(),
                                       tzinfo=hu_tz).replace(hour=h, minute=m)
        # Compute end via UTC so DST transitions (skipped / duplicated hours
        # on the last Sundays of March / October) bump the wall-clock
        # correctly instead of emitting a non-existent local time.
        end_local = (start_local.astimezone(ZoneInfo('UTC'))
                     + timedelta(minutes=15)).astimezone(hu_tz)
        out.append({
            'id': f"hu_entsoe_imb_{date_str}_{isp:03d}",
            'ISP': isp,
            'timeInterval': {
                'from': start_local.isoformat(timespec='milliseconds'),
                'to':   end_local.isoformat(timespec='milliseconds'),
            },
            'estimatedPriceNegativeImbalance': neg_huf,
            'estimatedPricePositiveImbalance': pos_huf,
            'estimatedPriceNegativeImbalanceEUR': neg_eur,
            'estimatedPricePositiveImbalanceEUR': pos_eur,
            'estimatedSystemImbalance': None,
            'realizedConsumption': None,
            'sumQup': None, 'sumQdn': None,
            'sumQupPup': None, 'sumQdownPdn': None,
            'imbalanceNettingImport': None, 'imbalanceNettingExport': None,
            'fcr': None, 'type': state,  # LONG | SHORT | SINGLE | DUAL
            'currency': 'HUF',
            'fxRate': huf_per_eur,
            'fxSource': fx.get('source'),
            'fxDate': fx.get('date'),
        })
    return out


def _damas_cached(endpoint_name, fetch_func, date_str, country='RO'):
    """Generic balancing-data caching wrapper.

    For RO, calls the DAMAS function passed in. For non-RO countries (HU),
    routes to the matching `mavir_client` function with the same JSON shape
    so callers don't need to know which country they're serving.
    """
    now = datetime.now()
    key = (country, endpoint_name, date_str)
    is_today = (date_str == _today())
    ttl = DAMAS_CACHE_SECS if is_today else DAMAS_HIST_CACHE_SECS

    if key in _damas_cache:
        entry = _damas_cache[key]
        if (now - entry['time']).total_seconds() < ttl:
            return entry['data']

    # Pick the right client for this country.
    actual_fetch = fetch_func
    if country == 'HU':
        # All HU balancing data goes TSO→us with no ENTSO-E hop:
        #   imbalance  → MAVIR settlement-unit-price XLSX (monthly batch,
        #                 ~2-week publication lag for the current month)
        #   marginal   → MAVIR XLSX
        #   activated  → MAVIR XLSX (with MAVIR RTDW live fallback further
        #                 down for current-month dates)
        mavir_fn_map = {
            'imbalance':  mavir.get_estimated_imbalance_prices,
            'marginal':   mavir.get_marginal_prices_overview,
            'activated':  mavir.get_activated_balancing_energy_overview,
        }
        actual_fetch = mavir_fn_map.get(endpoint_name)
        if actual_fetch is None:
            return []  # sysimbal/consumption/exchanges not yet wired for HU

    try:
        if country == 'HU':
            # MAVIR adapters work directly off the local YYYY-MM-DD; using the
            # UTC range would shift the request to the previous Budapest day.
            data = actual_fetch(date_str)
            # Live fallback: MAVIR's monthly XLSX lags ~2 weeks, so the XLSX
            # adapter returns [] for current-month dates. For 'activated', we
            # can synthesize the same shape from RTDW chart 11326 (live, 15-min
            # resolution, volumes only — MAVIR does not publish live prices).
            if not data and endpoint_name == 'activated':
                try:
                    data = mavir_rtdw.get_activated_energy_damas(date_str)
                    if data:
                        log.info(f"HU activated_energy: filled {len(data)} ISPs from RTDW for {date_str}")
                except Exception as e:
                    log.error(f"HU activated_energy RTDW fallback failed for {date_str}: {e}")
        else:
            dfrom, dto = _damas_time_range(date_str, country=country)
            data = actual_fetch(dfrom, dto)
        _damas_cache[key] = {'data': data, 'time': now}
        return data
    except Exception as e:
        log.error(f"Balancing fetch error ({country}, {endpoint_name}): {e}")
        if key in _damas_cache:
            return _damas_cache[key]['data']
        return []


def _today():
    return datetime.now().strftime('%Y-%m-%d')


def _yesterday():
    return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


# ── Background SEN polling ──────────────────────────────────────

def _sen_poll_loop():
    while True:
        try:
            data = te_tracker.fetch_and_store()
            if data:
                try:
                    db.save_sen_snapshot(data)
                except Exception as e:
                    log.debug(f"SEN DB save: {e}")
        except Exception as e:
            log.error(f"SEN poll error: {e}")
        time.sleep(20)


def _entsoe_poll_loop():
    """Background loop to keep retrying ENTSO-E until we get data."""
    time.sleep(5)  # Let app start first
    while True:
        try:
            today = _today()
            yesterday = _yesterday()
            # Always refresh today's data (actuals update throughout the day)
            # Only skip yesterday if we already have full data
            yesterday_good = (yesterday in _entsoe_cache and
                              len(_entsoe_cache[yesterday].get('data', {}).get('consumption', {})) >= 90)

            # Always refresh today; only fetch yesterday if not cached
            fetch_dates = [today]
            if not yesterday_good:
                fetch_dates.append(yesterday)
            for fetch_date in fetch_dates:
                log.info(f"ENTSO-E background fetch for {fetch_date}...")
                data = entsoe.get_forecasts_aggressive(fetch_date)
                has_data = any(len(v) > 0 for v in data.values() if isinstance(v, dict))
                if has_data:
                    old_data = _entsoe_cache[fetch_date]['data'] if fetch_date in _entsoe_cache else {}
                    merged = _merge_entsoe_data(old_data, data)
                    _entsoe_cache[fetch_date] = {'data': merged, 'time': datetime.now()}
                    act_pts = len(merged.get('post_actual_gen', {}))
                    log.info(f"ENTSO-E refresh: {fetch_date} — {act_pts} actual gen pts")
                else:
                    log.warning(f"ENTSO-E background fetch: no data for {fetch_date}")
        except Exception as e:
            log.error(f"ENTSO-E poll error: {e}")
        # Retry every 60s until we have good data
        time.sleep(60)


def _actual_gen_poll_loop():
    """Background loop to fetch actual generation data from ENTSO-E XML API."""
    # Wait for ENTSO-E POST API data to be available first (needs post_actual_gen/load)
    time.sleep(20)
    while True:
        try:
            today = _today()
            now = datetime.now()
            is_fresh = (today in _actual_cache and
                        (now - _actual_cache[today]['time']).total_seconds() < ACTUAL_CACHE_SECS)
            if not is_fresh:
                # Need POST API data for reliable totals - skip if not yet available
                post_data = _entsoe_cache.get(today, {}).get('data', {})
                if not post_data.get('post_actual_gen'):
                    log.info("Waiting for ENTSO-E POST API data before fetching actuals...")
                    time.sleep(10)
                    post_data = _entsoe_cache.get(today, {}).get('data', {})
                log.info(f"Fetching actual generation from ENTSO-E XML API for {today}...")
                data = entsoe.get_actual_generation(today, post_api_data=post_data)
                has_data = any(len(v) > 0 for v in data.values() if isinstance(v, dict))
                if has_data:
                    # Merge with existing cache to preserve wind/solar from earlier successful fetches
                    if today in _actual_cache:
                        old = _actual_cache[today]['data']
                        for k in ['actual_wind', 'actual_solar']:
                            if not data.get(k) and old.get(k):
                                data[k] = old[k]  # Keep previous wind/solar if XML API failed
                    _actual_cache[today] = {'data': data, 'time': now}
                    log.info(f"Actual generation: wind={len(data.get('actual_wind', {}))}, "
                             f"solar={len(data.get('actual_solar', {}))}, "
                             f"prod={len(data.get('actual_production', {}))}, "
                             f"cons={len(data.get('actual_consumption', {}))} pts")
                else:
                    log.warning("Actual generation: no data from XML API")

            # Also fetch yesterday if not cached
            yesterday = _yesterday()
            if yesterday not in _actual_cache:
                post_data = _entsoe_cache.get(yesterday, {}).get('data', {})
                data = entsoe.get_actual_generation(yesterday, post_api_data=post_data)
                has_data = any(len(v) > 0 for v in data.values() if isinstance(v, dict))
                if has_data:
                    _actual_cache[yesterday] = {'data': data, 'time': now}
        except Exception as e:
            log.error(f"Actual generation poll error: {e}")
        time.sleep(120)  # Check every 2 min (XML API rate limits aggressively)


_poll_thread = Thread(target=_sen_poll_loop, daemon=True)
_poll_thread.start()

_entsoe_thread = Thread(target=_entsoe_poll_loop, daemon=True)
_entsoe_thread.start()

_actual_thread = Thread(target=_actual_gen_poll_loop, daemon=True)
_actual_thread.start()

# Initial fetch
te_tracker.fetch_and_store()


# ── ML Prediction Background Threads ──────────────────────────

def _ml_historical_loop():
    """Collect historical DAMAS data and train model on startup, retrain every 6h."""
    time.sleep(15)  # Let other services start
    try:
        log.info("ML: Collecting historical DAMAS data...")
        predictor.collect_historical_data(days_back=60)
        log.info("ML: Training negative price model...")
        predictor.train_model()
        log.info("ML: Model ready for predictions")
    except Exception as e:
        log.error(f"ML initial training error: {e}")

    while True:
        time.sleep(3600 * 6)  # Retrain every 6 hours
        try:
            predictor.collect_historical_data(days_back=3)
            predictor.train_model()
        except Exception as e:
            log.error(f"ML retrain error: {e}")


def _ml_prediction_loop():
    """Run prediction every 15 seconds for real-time quarter data."""
    time.sleep(20)  # Wait for data + model
    while True:
        try:
            if predictor.model_ready:
                predictor.predict_current(damas_cache_func=_damas_cached,
                                          sen_live_func=te_tracker.get_live)
        except Exception as e:
            log.error(f"ML prediction error: {e}")
        time.sleep(15)


def _ml_actuals_loop():
    """Backfill actual prices into prediction log every 5 minutes."""
    time.sleep(300)
    while True:
        try:
            predictor.update_actuals(damas_cache_func=_damas_cached)
        except Exception as e:
            log.debug(f"ML actuals backfill error: {e}")
        time.sleep(300)


Thread(target=_ml_historical_loop, daemon=True).start()
Thread(target=_ml_prediction_loop, daemon=True).start()
Thread(target=_ml_actuals_loop, daemon=True).start()


# ── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    # Disable browser caching — without this, stale HTML from before the
    # country-selector was added keeps fetching /marginal_prices without
    # the ?country= param, so HU users see RO data.
    from flask import make_response
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


def _merge_entsoe_data(existing, new_data):
    """Merge new ENTSO-E data into existing, keeping the most complete version of each key."""
    if not existing:
        return new_data
    merged = {}
    all_keys = set(list(existing.keys()) + list(new_data.keys()))
    for k in all_keys:
        old_val = existing.get(k, {})
        new_val = new_data.get(k, {})
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            # Keep whichever has more data points, then fill gaps from the other
            if len(new_val) >= len(old_val):
                merged[k] = {**old_val, **new_val}
            else:
                merged[k] = {**new_val, **old_val}
        elif new_val:
            merged[k] = new_val
        else:
            merged[k] = old_val
    return merged


def _get_entsoe_cached(date_str):
    """Get ENTSO-E forecasts for a date, with caching.

    For today: always returns cache (background thread handles refresh).
    For historical: fetches on demand with caching.
    """
    now = datetime.now()
    is_today = (date_str == _today())

    # For today, just return whatever the background thread has cached
    if is_today:
        if date_str in _entsoe_cache:
            return _entsoe_cache[date_str]['data']
        return {}  # Background thread will fill this shortly

    # Historical dates: fetch on demand with cache
    ttl = ENTSOE_HIST_CACHE_SECS
    if date_str in _entsoe_cache:
        entry = _entsoe_cache[date_str]
        if (now - entry['time']).total_seconds() < ttl:
            return entry['data']

    try:
        data = entsoe.get_forecasts(date_str)
        has_data = any(v for v in data.values() if v)
        if has_data:
            old_data = _entsoe_cache[date_str]['data'] if date_str in _entsoe_cache else {}
            merged = _merge_entsoe_data(old_data, data)
            _entsoe_cache[date_str] = {'data': merged, 'time': now}
            return merged
        if date_str in _entsoe_cache:
            return _entsoe_cache[date_str]['data']
        return data
    except Exception as e:
        log.error(f"ENTSO-E forecast error for {date_str}: {e}")
        if date_str in _entsoe_cache:
            return _entsoe_cache[date_str]['data']
        return {}


def _get_actual_cached(date_str):
    """Get actual generation data, with caching.

    Falls back to POST API actual data if the XML API cache isn't available.
    """
    now = datetime.now()
    is_today = (date_str == _today())

    # First check the dedicated actual cache (from XML API + POST API)
    if is_today and date_str in _actual_cache:
        return _actual_cache[date_str]['data']

    if not is_today:
        ttl = ACTUAL_HIST_CACHE_SECS
        if date_str in _actual_cache:
            entry = _actual_cache[date_str]
            if (now - entry['time']).total_seconds() < ttl:
                return entry['data']

    # Fallback: extract POST API actual data directly from ENTSO-E cache
    post_data = _entsoe_cache.get(date_str, {}).get('data', {})
    if post_data:
        fallback = {
            'actual_wind': {},  # Only available from XML API
            'actual_solar': {},
            'actual_production': post_data.get('post_actual_gen', {}),
            'actual_consumption': post_data.get('post_actual_load', {}),
        }
        # Merge with XML API data if available
        if date_str in _actual_cache:
            cached = _actual_cache[date_str]['data']
            for k in ['actual_wind', 'actual_solar']:
                if cached.get(k):
                    fallback[k] = cached[k]
        if any(len(v) > 0 for v in fallback.values() if isinstance(v, dict)):
            return fallback

    # Last resort: on-demand fetch for historical dates
    if not is_today:
        try:
            data = entsoe.get_actual_generation(date_str, post_api_data=post_data)
            has_data = any(len(v) > 0 for v in data.values() if isinstance(v, dict))
            if has_data:
                _actual_cache[date_str] = {'data': data, 'time': now}
                return data
        except Exception as e:
            log.error(f"Actual generation error for {date_str}: {e}")

    if date_str in _actual_cache:
        return _actual_cache[date_str]['data']
    return {}


@app.route('/entsoe_generation')
def api_entsoe_generation():
    """
    ENTSO-E forecast data at 15-min intervals.
    Accepts ?date=YYYY-MM-DD (defaults to today) and ?country=RO|HU (defaults RO).
    Returns: {wind, solar, consumption, total_production, total_generation,
              yesterday_*, actual_wind, actual_solar, actual_consumption, actual_production}
    """
    date_str = request.args.get('date', _today())
    country = _country_from_request()

    if country == countries.DEFAULT_COUNTRY:
        # RO: use the rich cache maintained by background threads + XML API actuals.
        data = _get_entsoe_cached(date_str)
        actual = _get_actual_cached(date_str)
        try:
            prev_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        except ValueError:
            prev_date = _yesterday()
        ydata = _entsoe_cache.get(prev_date, {}).get('data', {})
    else:
        # Foreign country: on-demand fetch, no XML actuals, no yesterday comparison for now.
        data = _get_foreign_entsoe_cached(country, date_str)
        try:
            prev_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        except ValueError:
            prev_date = _yesterday()
        ydata = _foreign_entsoe_cache.get((country, prev_date), {}).get('data', {})
        # For foreign countries we only have POST-API actuals (embedded in `data`)
        actual = {
            'actual_wind': {},
            'actual_solar': {},
            'actual_production': data.get('post_actual_gen', {}),
            'actual_consumption': data.get('post_actual_load', {}),
        }

    result = {
        'wind': data.get('wind', {}),
        'solar': data.get('solar', {}),
        'consumption': data.get('consumption', {}),         # Load forecast = Consumption
        'total_production': data.get('total_generation', {}),  # Generation forecast = Production
        'total_generation': data.get('total_generation', {}),
        'yesterday_consumption': ydata.get('consumption', {}),
        'yesterday_production': ydata.get('total_generation', {}),
        'yesterday_wind': ydata.get('wind', {}),
        'yesterday_solar': ydata.get('solar', {}),
        # Actual generation data (from ENTSO-E XML API)
        'actual_wind': actual.get('actual_wind', {}),
        'actual_solar': actual.get('actual_solar', {}),
        'actual_consumption': actual.get('actual_consumption', {}),
        'actual_production': actual.get('actual_production', {}),
    }
    return jsonify(result)


@app.route('/transelectrica_data')
def api_transelectrica_data():
    """SEN readings table (last N entries).
    With ?hours=24, also includes historical data from DB for today."""
    hours = request.args.get('hours', None)
    if hours:
        rows = te_tracker.get_history_hours(int(hours))
        # Supplement with DB data for today if in-memory history is sparse
        if len(rows) < 20:
            try:
                db_rows = db.get_sen_today()
                # Merge: DB data first (older), then in-memory (fresher, may overlap)
                seen_ts = {r['timestamp'] for r in rows}
                for dr in db_rows:
                    if dr.get('timestamp') and dr['timestamp'] not in seen_ts:
                        rows.append(dr)
                        seen_ts.add(dr['timestamp'])
                rows.sort(key=lambda r: r.get('timestamp', ''), reverse=True)
            except Exception as e:
                log.debug(f"DB SEN load: {e}")
    else:
        rows = te_tracker.get_history(15)

    if not rows:
        return jsonify({'error': 'No data', 'rows': []})

    return jsonify({'rows': rows})


@app.route('/transelectrica_live')
def api_transelectrica_live():
    """Current live SEN reading."""
    live = te_tracker.get_live()
    if not live:
        return jsonify({'error': 'No live data'})
    return jsonify({'live': live})


@app.route('/imbalance_prices')
def api_imbalance_prices():
    """Estimated imbalance prices. RO: Transelectrica DAMAS. HU: MAVIR XLSX (~2-week lag)."""
    date_str = request.args.get('date', _today())
    country = _country_from_request()
    items = _damas_cached('imbalance', damas.get_estimated_imbalance_prices, date_str, country=country)
    rows = [i for i in items if i.get('id')]
    source = 'damas' if country == 'RO' else 'mavir'
    return jsonify({'rows': rows, 'date': date_str, 'source': source, 'country': country})


@app.route('/marginal_prices')
def api_marginal_prices():
    """Marginal prices (aFRR/mFRR). RO: DAMAS. HU: MAVIR (derived from fee/energy ratio)."""
    date_str = request.args.get('date', _today())
    country = _country_from_request()
    items = _damas_cached('marginal', damas.get_marginal_prices_overview, date_str, country=country)
    rows = [i for i in items if i.get('id')]
    source = 'damas' if country == 'RO' else 'mavir'
    return jsonify({'rows': rows, 'date': date_str, 'source': source, 'country': country})


_entsoe_bal_cache = {}


@app.route('/forex/huf_eur')
def api_forex_huf_eur():
    """ECB official HUF-per-EUR reference rate (daily, ~16:00 CET).

    Returns the latest rate, the ECB publication date it applies to, the
    source tier (ECB / ECB-hist / fallback), and a staleness flag. Cached
    in-process for 1 hour to match ECB's publication cadence.
    """
    force = request.args.get('refresh', '').lower() in ('1', 'true', 'yes')
    return jsonify(forex_client.get_huf_per_eur(force_refresh=force))


_tso_forecast_cache = {}


@app.route('/tso_forecast')
def api_tso_forecast():
    """TSO-direct forecast: wind, solar, load, generation per 15-min ISP.

    Bypasses ENTSO-E and pulls straight from each country's TSO public
    data feed. Currently wired:
        HU — MAVIR RTDW (charts 11840 wind, 19240+19260 PV, 7678 load, 4401 gen)

    For countries without a native TSO forecast API, returns empty series
    with a `sources._unsupported` note so the frontend can show a clear
    fallback message instead of silently fetching from ENTSO-E.
    """
    date_str = request.args.get('date', _today())
    # Accept any 2-letter country code, not just the validated RO/HU set —
    # the dispatcher itself knows which TSOs are wired and returns an
    # `_unsupported` source note for the rest, so the frontend can show a
    # proper "no TSO forecast wired" hint without a wrong-country fallback.
    raw_country = (request.args.get('country') or countries.DEFAULT_COUNTRY).upper()
    country = raw_country if raw_country.isalpha() and len(raw_country) == 2 else countries.DEFAULT_COUNTRY
    is_today = (date_str == _today())
    ttl = 120 if is_today else 24 * 3600
    now = datetime.now()
    key = (country, date_str)
    if key in _tso_forecast_cache:
        e = _tso_forecast_cache[key]
        if (now - e['time']).total_seconds() < ttl:
            return jsonify(e['data'])
    try:
        data = tso_forecast_client.get_forecast(country, date_str)
    except Exception as e:
        log.error(f"tso_forecast {country} {date_str}: {e}")
        return jsonify({'error': str(e), 'date': date_str, 'country': country}), 500
    _tso_forecast_cache[key] = {'data': data, 'time': now}
    return jsonify(data)


_entsoe_bids_cache = {}

@app.route('/entsoe_bids_totals')
def api_entsoe_bids_totals():
    """ENTSO-E Aggregated Balancing Energy Bids (A24) — total bid volume per 15-min MTU.

    Unlike activated volumes (what was used), this is total volume OFFERED —
    the "Total: X MW" figure shown on ENTSO-E's bids page. Coverage:
      RO: aFRR only.  HU: aFRR + mFRR.
    """
    date_str = request.args.get('date', _today())
    country = _country_from_request()
    cache_key = (country, date_str)
    is_today = (date_str == _today())
    ttl = 600 if is_today else 24 * 3600
    now = datetime.now()
    if cache_key in _entsoe_bids_cache:
        e = _entsoe_bids_cache[cache_key]
        if (now - e['time']).total_seconds() < ttl:
            return jsonify(e['data'])
    try:
        data = entsoe.get_aggregated_balancing_bids(date_str, country=country)
    except Exception as e:
        log.error(f"entsoe_bids_totals ({country}, {date_str}): {e}")
        return jsonify({'error': str(e), 'date': date_str}), 500
    payload = {'date': date_str, 'country': country, 'unit': data.get('unit'),
               'bids': {k: data.get(k, {}) for k in ('aFRR_Up','aFRR_Down','mFRR_Up','mFRR_Down')},
               'source': 'entsoe_a24'}
    _entsoe_bids_cache[cache_key] = {'data': payload, 'time': now}
    return jsonify(payload)

@app.route('/entsoe_balancing_prices')
def api_entsoe_balancing_prices():
    """ENTSO-E Accepted Balancing Energy prices (A84) — RO aFRR + RR per 15-min MTU.

    Returns prices in the native ENTSO-E currency for the zone (RON for RO).
    mFRR is NOT published to ENTSO-E for Romania — we surface only aFRR + RR.
    For mFRR clearing prices, the DAMAS `/marginal_prices` endpoint is the source.
    """
    date_str = request.args.get('date', _today())
    country = _country_from_request()
    if country != 'RO':
        return jsonify({'prices': {}, 'date': date_str, 'country': country,
                        'currency': None, 'note': 'A84 balancing energy prices: RO only'})

    cache_key = (country, date_str)
    is_today = (date_str == _today())
    # ENTSO-E publishes new MTU prices every ~15 min during the day and the
    # full historical day around 16:00 CET. A 2-min cache for today catches
    # fresh publications quickly while keeping load minimal.
    ttl = 120 if is_today else 24 * 3600
    now = datetime.now()
    if cache_key in _entsoe_bal_cache:
        entry = _entsoe_bal_cache[cache_key]
        if (now - entry['time']).total_seconds() < ttl:
            return jsonify(entry['data'])

    try:
        data = entsoe.get_balancing_activation_prices(date_str, country=country)
    except Exception as e:
        log.error(f"entsoe_balancing_prices ({country}, {date_str}): {e}")
        return jsonify({'error': str(e), 'date': date_str}), 500

    payload = {
        'date': date_str,
        'country': country,
        'currency': data.get('currency'),
        'prices': {
            'aFRR_Up':   data.get('aFRR_Up', {}),
            'aFRR_Down': data.get('aFRR_Down', {}),
            'RR_Up':     data.get('RR_Up', {}),
            'RR_Down':   data.get('RR_Down', {}),
        },
        'source': 'entsoe_a84',
    }
    _entsoe_bal_cache[cache_key] = {'data': payload, 'time': now}
    return jsonify(payload)


@app.route('/activated_energy')
def api_activated_energy():
    """Activated balancing energy. RO: DAMAS. HU: MAVIR."""
    date_str = request.args.get('date', _today())
    country = _country_from_request()
    items = _damas_cached('activated', damas.get_activated_balancing_energy_overview, date_str, country=country)
    rows = [i for i in items if i.get('id')]
    source = 'damas' if country == 'RO' else 'mavir'
    return jsonify({'rows': rows, 'date': date_str, 'source': source, 'country': country})


@app.route('/system_imbalance')
def api_system_imbalance():
    """Estimated power system imbalance from DAMAS API."""
    date_str = request.args.get('date', _today())
    items = _damas_cached('sysimbal', damas.get_estimated_power_system_imbalance, date_str)
    rows = [i for i in items if i.get('id')]
    return jsonify({'rows': rows, 'date': date_str})


@app.route('/consumption_overview')
def api_consumption_overview():
    """Daily consumption forecast vs realized from DAMAS API."""
    date_str = request.args.get('date', _today())
    items = _damas_cached('consumption', damas.get_daily_consumption_overview, date_str)
    rows = [i for i in items if i.get('id')]
    return jsonify({'rows': rows, 'date': date_str})


@app.route('/scheduled_exchanges')
def api_scheduled_exchanges():
    """Scheduled cross-border exchanges from DAMAS API."""
    date_str = request.args.get('date', _today())
    items = _damas_cached('exchanges', damas.get_scheduled_exchanges, date_str)
    return jsonify({'rows': items, 'date': date_str})


@app.route('/hu_live_grid')
def api_hu_live_grid():
    """Live Hungarian grid snapshot from MAVIR's RTDW (rtdwweb.mavir.hu).

    Returns a compact snapshot suitable for a dashboard card: current
    generation plan/actual, load plan/actual, frequency, aFRR reserve band,
    generation mix by fuel type, and 15-min balancing activations.

    Forces country=HU — this endpoint has no Romanian equivalent.
    """
    date_str = request.args.get('date', _today())
    try:
        snap = mavir_rtdw.live_snapshot(date_str)
        return jsonify(snap)
    except Exception as e:
        log.error(f"hu_live_grid error for {date_str}: {e}")
        return jsonify({'error': str(e), 'date': date_str}), 500


@app.route('/api/countries')
def api_countries():
    """List available countries and which panels each one supports.
    Used by the frontend to hide panels that have no data source for HU, etc."""
    return jsonify({
        'default': countries.DEFAULT_COUNTRY,
        'countries': {
            code: {
                'label': cfg['label'],
                'currency': cfg['currency'],
                'panels': cfg['panels'],
            }
            for code, cfg in countries.COUNTRIES.items()
        },
    })


@app.route('/dam_prices')
def api_dam_prices():
    """DAM prices for a date (EUR/MWh). Accepts ?country=RO|HU (default RO)."""
    date = request.args.get('date', _today())
    country = _country_from_request()
    key = (country, date)

    if key in _dam_cache:
        cached, cache_time = _dam_cache[key]
        if (datetime.now() - cache_time).total_seconds() < 3600:
            return jsonify({'prices': cached, 'date': date, 'country': country})

    try:
        prices = entsoe.get_dam_prices(date, country=country)
        if prices:
            _dam_cache[key] = (prices, datetime.now())
            return jsonify({'prices': prices, 'date': date, 'country': country})
    except Exception as e:
        log.error(f"DAM prices error ({country}): {e}")

    if key in _dam_cache:
        return jsonify({'prices': _dam_cache[key][0], 'date': date, 'country': country})
    return jsonify({'prices': {}, 'date': date, 'country': country})


# ── Forecast Hub: wind & solar D-1 forecast for HU + neighbors ────────
#
# Centralizes wind+solar forecasts from MAVIR (HU TSO native) and ENTSO-E
# (everyone else). Powers the standalone /forecast page, which lets the
# user pick any one country to chart and (optionally) overlay a second
# country for side-by-side comparison.

import entsoe_client as _entsoe_mod
import open_meteo_client as _open_meteo

FORECAST_HUB_COUNTRIES = {
    'HU': {'label': 'Hungary',  'flag': '🇭🇺', 'tz': 'Europe/Budapest'},
    'AT': {'label': 'Austria',  'flag': '🇦🇹', 'tz': 'Europe/Vienna'},
    'HR': {'label': 'Croatia',  'flag': '🇭🇷', 'tz': 'Europe/Zagreb'},
    'RO': {'label': 'Romania',  'flag': '🇷🇴', 'tz': 'Europe/Bucharest'},
    'RS': {'label': 'Serbia',   'flag': '🇷🇸', 'tz': 'Europe/Belgrade'},
    'SI': {'label': 'Slovenia', 'flag': '🇸🇮', 'tz': 'Europe/Ljubljana'},
    'SK': {'label': 'Slovakia', 'flag': '🇸🇰', 'tz': 'Europe/Bratislava'},
}

_forecast_hub_cache = {}  # {(country, date): {'data', 'time'}}


def _forecast_hub_fetch(country, date_str):
    """Return all available wind+solar forecast sources for one country/date.

    Sources combined:
      - ENTSO-E: 4 columns (DAY_AHEAD, INTRADAY, CURRENT, ACTUAL) per series
      - MAVIR (HU only): wind forecast from RTDW chart 11840
      - Open-Meteo: weather-derived wind & solar (independent cross-check)

    Empty maps are returned (not None) so the frontend can distinguish
    "fetched, no data" from "not fetched." `meta.published` flags tell
    the UI which lines to draw and which to mark as "not yet published."
    """
    now = datetime.now()
    key = (country, date_str)
    is_today = (date_str == _today())
    is_future = date_str > _today()
    ttl = 600 if (is_today or is_future) else 3600
    if key in _forecast_hub_cache:
        entry = _forecast_hub_cache[key]
        if (now - entry['time']).total_seconds() < ttl:
            return entry['data']

    out = {
        'entsoe': {'wind': {}, 'solar': {}, 'meta': {}},
        'mavir': None,
        'open_meteo': None,
        'errors': [],
    }

    # ── ENTSO-E ──
    try:
        es = entsoe.get_wind_solar_all_columns(date_str, country=country)
        out['entsoe'] = es
    except Exception as e:
        msg = f"ENTSO-E {country} {date_str}: {e}"
        log.warning(msg)
        out['errors'].append(msg)

    # ── MAVIR (HU only) ──
    if country == 'HU':
        try:
            series = mavir_rtdw.extract_series(11840, date_str)
            wf = series.get('wind_forecast', {}) or {}
            wa = series.get('wind_actual',   {}) or {}
            def _hm(d):
                out_d = {}
                for ts, v in d.items():
                    parts = str(ts).split(' ')
                    if len(parts) == 2 and len(parts[1]) >= 5:
                        out_d[parts[1][:5]] = v
                return out_d
            wf_hm = _hm(wf)
            wa_hm = _hm(wa)
            if wf_hm or wa_hm:
                out['mavir'] = {
                    'wind_forecast': wf_hm,
                    'wind_actual':   wa_hm,
                    'meta': {
                        'source': 'MAVIR RTDW chart 11840',
                        'resolution_min': 1,
                        'fetched_at': datetime.utcnow().isoformat() + 'Z',
                    },
                }
        except Exception as e:
            log.debug(f"forecast_hub MAVIR {date_str}: {e}")

    # ── Open-Meteo ──
    try:
        om = _open_meteo.get_forecast(country, date_str)
        if om:
            out['open_meteo'] = om
    except Exception as e:
        log.debug(f"forecast_hub Open-Meteo {country} {date_str}: {e}")

    # ── Compute derived freshness flags ──
    es = out['entsoe']
    def _has(col, kind):
        return bool((es.get(kind) or {}).get(col))
    out['summary'] = {
        'entsoe_day_ahead_wind':  _has('DAY_AHEAD', 'wind'),
        'entsoe_day_ahead_solar': _has('DAY_AHEAD', 'solar'),
        'entsoe_intraday_wind':   _has('INTRADAY', 'wind'),
        'entsoe_intraday_solar':  _has('INTRADAY', 'solar'),
        'entsoe_current_wind':    _has('CURRENT', 'wind'),
        'entsoe_current_solar':   _has('CURRENT', 'solar'),
        'entsoe_actual_wind':     _has('ACTUAL', 'wind'),
        'entsoe_actual_solar':    _has('ACTUAL', 'solar'),
        'mavir_available':        out['mavir'] is not None,
        'open_meteo_available':   out['open_meteo'] is not None,
    }

    _forecast_hub_cache[key] = {'data': out, 'time': now}
    return out


@app.route('/forecast')
def forecast_page():
    """Standalone wind + solar forecast hub (Hungary)."""
    from flask import make_response
    resp = make_response(render_template('forecast.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/api/forecast_hub')
def api_forecast_hub():
    """Wind + solar forecast for one country/date.

    Query params:
      country : HU|AT|SK|RO|CZ|PL|RS|HR|SI  (default HU)
      date    : YYYY-MM-DD                   (default today, accepts tomorrow)
    """
    country = (request.args.get('country', 'HU') or 'HU').upper()
    if country not in FORECAST_HUB_COUNTRIES:
        return jsonify({'error': f'unsupported country {country}',
                        'supported': list(FORECAST_HUB_COUNTRIES.keys())}), 400
    date = request.args.get('date', _today())
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'date must be YYYY-MM-DD'}), 400

    data = _forecast_hub_fetch(country, date)
    meta = FORECAST_HUB_COUNTRIES[country]
    return jsonify({
        'country':    country,
        'label':      meta['label'],
        'flag':       meta['flag'],
        'tz':         meta['tz'],
        'date':       date,
        'entsoe':     data['entsoe'],
        'mavir':      data['mavir'],
        'open_meteo': data['open_meteo'],
        'summary':    data['summary'],
        'errors':     data.get('errors', []),
    })


@app.route('/api/forecast_hub/countries')
def api_forecast_hub_countries():
    """List of countries available in the forecast hub (for the picker)."""
    return jsonify(FORECAST_HUB_COUNTRIES)


@app.route('/api/debug_cache')
def api_debug_cache():
    """Debug: show what's in the ENTSO-E cache."""
    result = {}
    for date_str, entry in _entsoe_cache.items():
        data = entry.get('data', {})
        cache_time = entry.get('time', '')
        result[date_str] = {
            'cache_time': str(cache_time),
            'keys': {k: len(v) if isinstance(v, dict) else str(v) for k, v in data.items()},
        }
    # Include actual generation cache info
    for date_str, entry in _actual_cache.items():
        key = date_str + '_actual'
        data = entry.get('data', {})
        cache_time = entry.get('time', '')
        result[key] = {
            'cache_time': str(cache_time),
            'keys': {k: len(v) if isinstance(v, dict) else str(v) for k, v in data.items()},
        }
    return jsonify(result)


@app.route('/api/status')
def api_status():
    # Check if we have any cached ENTSO-E data
    has_entsoe = any(
        entry.get('data') and any(v for v in entry['data'].values() if v)
        for entry in _entsoe_cache.values()
    ) if _entsoe_cache else False
    return jsonify({
        'ok': True,
        'entsoe_key': bool(ENTSOE_KEY),
        'entsoe_urls': ['https://transparency.entsoe.eu (new POST API)'],
        'entsoe_has_data': has_entsoe,
        'sen_history': len(te_tracker.history),
        'live': te_tracker.get_live() is not None,
    })


# ── ML Prediction API Endpoints ──────────────────────────────

@app.route('/api/prediction')
def api_prediction():
    """Current real-time negative price prediction."""
    pred = predictor.latest_prediction
    if pred:
        return jsonify(pred)
    return jsonify({
        'model_ready': predictor.model_ready,
        'training_in_progress': predictor.training_in_progress,
        'historical_days_loaded': predictor.historical_days_loaded,
        'probability': None,
        'risk_level': None,
    })


@app.route('/api/prediction_history')
def api_prediction_history():
    """Recent predictions with accuracy stats."""
    hours = int(request.args.get('hours', 6))
    preds = db.get_recent_predictions(hours=hours)
    accuracy = predictor.get_accuracy_stats()
    return jsonify({'predictions': preds, 'accuracy': accuracy})


@app.route('/api/prediction_status')
def api_prediction_status():
    """Model training status."""
    meta = db.get_latest_model_metadata()
    return jsonify({
        'model_ready': predictor.model_ready,
        'training_in_progress': predictor.training_in_progress,
        'historical_days_loaded': predictor.historical_days_loaded,
        'model_version': predictor.model_version,
        'thresholds': {str(k): v for k, v in predictor.thresholds.items()},
        'model_metadata': meta,
    })


@app.route('/api/market_analysis')
def api_market_analysis():
    """Full archive analysis — hourly profiles, risk factors, streaks etc."""
    analysis = market_analysis.get_full_analysis()
    return jsonify(analysis)


@app.route('/api/market_suggestion')
def api_market_suggestion():
    """Current SAFE/NORMAL/RISKY suggestion based on live data."""
    date_str = _today()
    imb_items = _damas_cached('imbalance', damas.get_estimated_imbalance_prices, date_str)
    marg_items = _damas_cached('marginal', damas.get_marginal_prices_overview, date_str)

    # Get latest ISP with real data
    imb_valid = [r for r in imb_items if r.get('id')
                 and isinstance(r.get('estimatedPriceNegativeImbalance'), (int, float))]
    marg_valid = {r.get('ISP', 0): r for r in marg_items if r.get('id')}

    if not imb_valid:
        return jsonify({'error': 'No live data'})

    latest = imb_valid[-1]
    latest_marg = marg_valid.get(latest.get('ISP', 0), {})

    suggestion = market_analysis.get_current_suggestion(latest, latest_marg)
    return jsonify(suggestion)


if __name__ == '__main__':
    log.info("Energy Prediction Platform starting on port 8084")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8084')), debug=False)
