"""MAVIR Real-Time Data Warehouse (RTDW) client.

MAVIR exposes live Hungarian grid data via a chart-export endpoint on
`rtdwweb.mavir.hu`. Each chart has a numeric ID and returns an XLSX with
minute-resolution columns. This module wraps the endpoint for a handful
of high-value charts and returns normalized dicts.

Endpoint pattern (discovered by inspecting the public iframe pages on
mavir.hu):

    GET https://rtdwweb.mavir.hu/rtdwweb/webuser/chart/{chart_id}/export
        ?exportType=xlsx
        &fromTime={epoch_ms}
        &toTime={epoch_ms}
        &periodType=min   (or hour)
        &period=1

The SSL cert on rtdwweb is not in our trust store, so we disable cert
verification — the data is public and read-only.

Rate limit: ~1 req / 2 seconds. Multiple fetches are paced by the client.

Chart IDs and the specific column names below come from the public
iframe at mavir.hu/web/mavir/igenybeveheto-kiegy-szab-kapacitasok-* and
mavir.hu/web/mavir/szabalyozasi-adatok-*.
"""
import io
import logging
import time
import datetime as dt
from threading import Lock
from zoneinfo import ZoneInfo

import requests
import urllib3
import urllib.parse as _up

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
log = logging.getLogger(__name__)

TZ = ZoneInfo('Europe/Budapest')
BASE = 'https://rtdwweb.mavir.hu/rtdwweb/webuser/chart'

# Time-based in-memory cache per (chart_id, date). Parsed records are big-ish
# so we don't want to refetch on every request; live data refreshes every
# minute server-side anyway.
_CACHE_TTL_LIVE = 60         # today's data
_CACHE_TTL_HIST = 3600       # past days (immutable)
_cache = {}
_cache_lock = Lock()

# Minimum spacing between outbound requests — MAVIR returns 429 at ~1 req/s
_last_fetch_ts = 0.0
_pace_lock = Lock()
_MIN_GAP_SECS = 5.0  # MAVIR throttles at ~1 req / 2.5s and 429s are sticky;
                     # use a generous gap so live_snapshot doesn't lose charts.

_session = None
_session_lock = Lock()


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            s = requests.Session()
            s.verify = False
            s.headers.update({
                'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                               'AppleWebKit/537.36 (KHTML, like Gecko) '
                               'Chrome/131.0.0.0 Safari/537.36'),
                'Accept': '*/*',
                'Accept-Language': 'hu-HU,hu;q=0.9,en;q=0.8',
            })
            _session = s
        return _session


def _pace():
    """Sleep enough to maintain a minimum gap between outbound MAVIR calls."""
    global _last_fetch_ts
    with _pace_lock:
        now = time.time()
        gap = now - _last_fetch_ts
        if gap < _MIN_GAP_SECS:
            time.sleep(_MIN_GAP_SECS - gap)
        _last_fetch_ts = time.time()


def _fetch_chart_xlsx(chart_id, from_dt, to_dt, period_type='min', period=1):
    """Download the XLSX for one chart over one Budapest-local day window."""
    from_ms = int(from_dt.timestamp() * 1000)
    to_ms = int(to_dt.timestamp() * 1000)
    url = f'{BASE}/{chart_id}/export?' + _up.urlencode({
        'exportType': 'xlsx',
        'fromTime': str(from_ms),
        'toTime': str(to_ms),
        'periodType': period_type,
        'period': str(period),
    })
    _pace()
    r = _get_session().get(url, timeout=30)
    # Retry on 429 with exponential backoff (the throttle stays in effect briefly).
    for attempt in (1, 2):
        if r.status_code != 429:
            break
        time.sleep(8 * attempt)
        _pace()
        r = _get_session().get(url, timeout=30)
    if r.status_code != 200 or 'sheet' not in r.headers.get('Content-Type', ''):
        raise RuntimeError(f"MAVIR rtdw chart {chart_id}: HTTP {r.status_code}")
    return r.content


def _parse_xlsx(content):
    """Return (headers, list_of_row_dicts) from MAVIR-format XLSX.

    MAVIR XLSX: row 1 = headers, row 2+ = data. First column is always
    "Időpont" (timestamp) as a string like "2026.04.21 00:01:00 +0200".
    """
    import openpyxl  # import locally — openpyxl isn't cheap
    wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True, read_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    headers = [str(h) if h is not None else '' for h in (next(it, []) or [])]
    rows = []
    for row in it:
        if not row:
            continue
        rec = dict(zip(headers, row))
        rows.append(rec)
    wb.close()
    return headers, rows


def get_chart(chart_id, date_str):
    """Fetch + parse one chart for one Budapest local day. Cached in memory.

    Returns:
        {'headers': [col names], 'rows': [{col: value, ...}, ...]}
    """
    today = dt.datetime.now(TZ).strftime('%Y-%m-%d')
    key = (chart_id, date_str)
    is_today = (date_str == today)
    ttl = _CACHE_TTL_LIVE if is_today else _CACHE_TTL_HIST
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and now - entry['time'] < ttl:
            return entry['data']
    try:
        y, m, d = date_str.split('-')
        from_dt = dt.datetime(int(y), int(m), int(d), 0, 0, tzinfo=TZ)
        to_dt = from_dt + dt.timedelta(days=1)
        content = _fetch_chart_xlsx(chart_id, from_dt, to_dt)
        headers, rows = _parse_xlsx(content)
        data = {'headers': headers, 'rows': rows}
        with _cache_lock:
            _cache[key] = {'data': data, 'time': now}
        log.info(f"MAVIR rtdw chart {chart_id} ({date_str}): {len(rows)} rows")
        return data
    except Exception as e:
        log.error(f"MAVIR rtdw chart {chart_id} ({date_str}) failed: {e}")
        with _cache_lock:
            entry = _cache.get(key)
            if entry:
                return entry['data']
        return {'headers': [], 'rows': []}


# ── Named chart helpers (field name → Hungarian column on source) ───

CHART_MAP = {
    # chart_id: {semantic_name: Hungarian column name in the XLSX header row}
    4401: {
        'plan_gross':   'Bruttó terv erőművi termelés',
        'actual_gross': 'Bruttó tény erőművi termelés',
        'plan_net':     'Nettó terv erőművi termelés',
        'actual_net':   'Nettó hazai termelés tény',
    },
    4444: {
        'hz':           'Hálózati frekvencia',
    },
    7678: {
        'actual_gross':        'Bruttó tény rendszerterhelés',
        'plan_gross':          'Bruttó terv rendszerterhelés',
        'dayahead_gross':      'Bruttó rendszerterhelés becslés (dayahead)',
        'actual_net':          'Nettó tény rendszerterhelés - net.ker.elsz.meres',
    },
    9404: {
        # Generation by fuel/primary source — 14 types
        'total':        'Hazai termelés (erőművi szumma)',
        'nuclear':      'Nukleáris erőművek',
        'lignite':      'Barnakőszén-lignit erőművek',
        'gas':          'Gáz (fosszilis) erőművek',
        'hard_coal':    'Feketekőszén erőművek',
        'oil':          'Olaj (fosszilis) erőművek',
        'wind':         'Szárazföldi szélerőművek',
        'biomass':      'Biomassza erőművek',
        'pv':           'Ipari PV',
        'waste':        'Szemétégető erőművek',
        'hydro_ror':    'Folyóvizes erőművek',
        'hydro_res':    'Víztározós vízerőművek',
        'other_renew':  'Egyéb megújuló erőművek',
        'other':        'Egyéb erőművek',
    },
    10260: {
        'load_actual':       'Bruttó tény rendszerterhelés',
        'load_estimate':     'Bruttó rendszer terhelésbecslés MAVIR',
        'afrr_avail_min':    '15 perc alatt igénybe vehető aFRR kiegy.szab.kap. (forgó tartalék) MIN',
        'afrr_avail_max':    '15 perc alatt igénybe vehető aFRR kiegy.szab.kap. (forgó tartalék) MAX',
    },
    11326: {
        # 15-min-resolution balancing activations
        'afrr_up':       'aFRR (Automatikus) szabályozás FEL (15p)',
        'afrr_down':     'aFRR (Automatikus) szabályozás LE (15p)',
        'afrr_hu_up':    'Hazai aFRR (aut.) szab. FEL (15p)',
        'afrr_hu_down':  'Hazai aFRR (aut.) szab. LE (15p)',
        'igcc_up':       'IGCC szabályozás FEL (15p)',
        'igcc_down':     'IGCC szabályozás LE (15p)',
        'mfrr_up':       'Nem automatikus szabályozás mértéke fel (balancing)',
        'mfrr_down':     'Nem automatikus szabályozás mértéke le (balancing)',
    },
    11322: {
        # Non-balancing-purpose regulation (redispatch-style actions). Separate
        # chart from 11326 because the source page's XLSX export merges the two.
        'nonbal_up':     'Nem kiegyenlítő célú szabályozás mértéke fel (15p)',
        'nonbal_down':   'Nem kiegyenlítő célú szabályozás mértéke le (15p)',
    },
    1000684: {
        # Renewable share (%) in domestic gross generation (actual, incl. small PV)
        'renew_share':   'Megújuló részarány a hazai bruttó termelésben - tény - HMKE és SCTE-vel',
    },
    1000688: {
        # Total power-plant CO2 emissions (actual), in t/h
        'co2_actual':    'Összes erőművi CO2 kibocsátás - tény',
    },
    19782: {
        'total_pv_feed': 'VER összes PV - Mért hálózati kitáplálás',
    },
    11840: {
        'wind_actual':   'Szélerőművek tény - nettó üzemirányítási',
        'wind_forecast': 'Szélerőművek becsült termelése (aktuális)',
    },
}


def _parse_ts(ts):
    """'2026.04.21 00:01:00 +0200' -> 'YYYY-MM-DD HH:MM:SS' (drop TZ suffix)."""
    try:
        s = str(ts)
        parts = s.split(' ')
        date = parts[0].replace('.', '-')
        return f"{date} {parts[1]}"
    except Exception:
        return str(ts)


def extract_series(chart_id, date_str):
    """Return {semantic_field: {timestamp_str: value}} for a mapped chart."""
    if chart_id not in CHART_MAP:
        raise ValueError(f"chart {chart_id} not mapped in CHART_MAP")
    data = get_chart(chart_id, date_str)
    field_map = CHART_MAP[chart_id]
    out = {f: {} for f in field_map}
    for rec in data['rows']:
        ts = _parse_ts(rec.get('Időpont'))
        for field, col_name in field_map.items():
            val = rec.get(col_name)
            if val is None:
                continue
            try:
                out[field][ts] = float(val)
            except (TypeError, ValueError):
                continue
    return out


def latest_values(chart_id, date_str, prefer_fields=None):
    """Return the most recent row's values as {timestamp, field: val, ...}.

    Searches from the bottom up. If `prefer_fields` is given, finds the
    last row that has at least one non-null value in THAT preferred subset
    (typically the "actual" columns). Otherwise, the last row with any
    non-null mapped value.

    The MAVIR XLSX commonly includes the next-day midnight row populated
    with plan-only values; preferring `actual_*` fields skips it.
    """
    if chart_id not in CHART_MAP:
        raise ValueError(f"chart {chart_id} not mapped")
    data = get_chart(chart_id, date_str)
    field_map = CHART_MAP[chart_id]
    target_fields = prefer_fields if prefer_fields else list(field_map.keys())
    for rec in reversed(data['rows']):
        vals = {f: rec.get(cn) for f, cn in field_map.items()}
        if any(vals.get(f) is not None for f in target_fields):
            return {'timestamp': _parse_ts(rec.get('Időpont')), **vals}
    # Fallback: any non-null
    for rec in reversed(data['rows']):
        vals = {f: rec.get(cn) for f, cn in field_map.items()}
        if any(v is not None for v in vals.values()):
            return {'timestamp': _parse_ts(rec.get('Időpont')), **vals}
    return None


# ── High-level aggregate: one call returns all key live metrics ─────

def live_snapshot(date_str):
    """Fetch several charts and return a compact snapshot for the HU dashboard.

    Rate-limited to ~1 call per 2s, so the first uncached call takes ~12-15s
    for the full set. Subsequent calls for today hit the 60s cache.
    """
    snap = {
        'date': date_str,
        'timestamp': None,
        'generation': None,   # chart 4401
        'load': None,         # chart 7678
        'frequency': None,    # chart 4444
        'reserves': None,     # chart 10260
        'fuel_mix': None,     # chart 9404 — latest row only
        'balancing': None,    # chart 11326 — latest 15-min row
        'nonbalancing': None, # chart 11322 — redispatch-style regulation
        'renew_share': None,  # chart 1000684 — % renewable in gross generation
        'co2': None,          # chart 1000688 — total plant CO2 (t/h)
        # Full intraday series for charting (trimmed to needed fields)
        'series': {},
    }

    try:
        snap['generation'] = latest_values(4401, date_str, prefer_fields=['actual_gross', 'actual_net'])
        gen_series = extract_series(4401, date_str)
        snap['series']['generation_actual'] = gen_series.get('actual_gross', {})
        snap['series']['generation_plan'] = gen_series.get('plan_gross', {})
    except Exception as e:
        log.debug(f"live_snapshot gen: {e}")

    try:
        snap['load'] = latest_values(7678, date_str, prefer_fields=['actual_gross', 'actual_net'])
        load_series = extract_series(7678, date_str)
        snap['series']['load_actual'] = load_series.get('actual_gross', {})
        snap['series']['load_plan'] = load_series.get('plan_gross', {})
    except Exception as e:
        log.debug(f"live_snapshot load: {e}")

    try:
        freq_series = extract_series(4444, date_str)
        hz = freq_series.get('hz', {})
        snap['series']['frequency'] = hz
        if hz:
            last_ts = sorted(hz.keys())[-1]
            snap['frequency'] = {'timestamp': last_ts, 'hz': hz[last_ts]}
    except Exception as e:
        log.debug(f"live_snapshot freq: {e}")

    try:
        snap['reserves'] = latest_values(10260, date_str,
                                         prefer_fields=['afrr_avail_min', 'afrr_avail_max'])
    except Exception as e:
        log.debug(f"live_snapshot reserves: {e}")

    try:
        snap['fuel_mix'] = latest_values(9404, date_str,
                                         prefer_fields=['nuclear', 'gas', 'lignite', 'wind', 'pv'])
        # Pull minute-resolution wind + total-PV series from the fuel-mix chart
        # so the HU production graph can render them alongside generation/load.
        fuel_series = extract_series(9404, date_str)
        snap['series']['wind'] = fuel_series.get('wind', {})
        snap['series']['pv']   = fuel_series.get('pv', {})
    except Exception as e:
        log.debug(f"live_snapshot fuel_mix: {e}")

    try:
        snap['balancing'] = latest_values(11326, date_str,
                                          prefer_fields=['afrr_up', 'afrr_down', 'afrr_hu_up', 'afrr_hu_down'])
    except Exception as e:
        log.debug(f"live_snapshot balancing: {e}")

    # Non-balancing-purpose regulation (redispatch). Usually zero — surface the
    # latest row anyway so users can see that it's tracked.
    try:
        snap['nonbalancing'] = latest_values(11322, date_str,
                                             prefer_fields=['nonbal_up', 'nonbal_down'])
    except Exception as e:
        log.debug(f"live_snapshot nonbalancing: {e}")

    # Renewable share + CO2 emissions — HU-specific extras available on RTDW.
    try:
        snap['renew_share'] = latest_values(1000684, date_str,
                                            prefer_fields=['renew_share'])
    except Exception as e:
        log.debug(f"live_snapshot renew_share: {e}")

    try:
        snap['co2'] = latest_values(1000688, date_str,
                                    prefer_fields=['co2_actual'])
    except Exception as e:
        log.debug(f"live_snapshot co2: {e}")

    # Pick a unified snapshot timestamp (prefer frequency's — 1-min res)
    snap['timestamp'] = ((snap.get('frequency') or {}).get('timestamp')
                        or (snap.get('generation') or {}).get('timestamp'))
    return snap


# ── DAMAS-shape adapters (for dashboard fallback when XLSX is unavailable) ──
#
# The monthly MAVIR XLSX reports carry the authoritative balancing data but
# publish with a ~2-week lag. For the current month we have no XLSX rows, so
# the dashboard's Merit Order / Activated Energy cards render empty. RTDW
# chart 11326 publishes the same activation volumes live (minute-resolution),
# so we synthesize DAMAS-shape records from it for today and the tail of the
# current month.

def _rtdw_utc_iso(date_str, hh, mm):
    """Build a Budapest-local ISO timestamp with a DST-aware offset.

    Previously this hardcoded '+01:00' which gave the wrong UTC anchor on
    summer dates (Budapest is CEST/+02:00 from the last Sunday of March
    through the last Sunday of October). A fixed offset also caused the
    frontend to bucket the same ISP into two different keys when different
    adapters disagreed on the end-of-day format (ISP 96 at 23:45 → next-day
    00:00 vs 23:59). Using proper ISO with tzinfo fixes both issues.
    """
    y, mo, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    hh_c, mm_c = hh, mm
    day_off = 0
    if hh_c == 23 and mm_c == 59:  # legacy "end-of-day" callers
        hh_c, mm_c = 0, 0
        day_off = 1
    elif hh_c >= 24:
        hh_c -= 24
        day_off = 1
    base = dt.datetime(y, mo, d, hh_c, mm_c, tzinfo=TZ) + dt.timedelta(days=day_off)
    return base.isoformat(timespec='milliseconds')


def _aggregate_chart_by_isp(chart_id, date_str):
    """Parse a minute-resolution chart into 96 × 15-min ISP averages.

    MAVIR's minute timestamps are END-labeled: the sample at "HH:MM" covers
    the minute ending at HH:MM, and the samples at HH:01..HH:15 together
    cover ISP 1 for that hour. A naïve `minute // 15` puts the boundary
    minute (e.g. 00:15) into the next ISP, biasing values at transitions —
    so we subtract one minute before bucketing.
    """
    data = get_chart(chart_id, date_str)
    field_map = CHART_MAP[chart_id]
    buckets = {i: {f: [] for f in field_map} for i in range(1, 97)}
    for rec in data['rows']:
        ts = rec.get('Időpont')
        if not ts:
            continue
        try:
            parts = str(ts).split(' ')
            hm = parts[1].split(':')
            h, mi = int(hm[0]), int(hm[1])
        except Exception:
            continue
        if h >= 24 or (h == 0 and mi == 0):
            # MAVIR emits a 00:00:00 row at the start whose value belongs to
            # the prior day's final ISP; skip it to avoid contaminating ISP 1.
            continue
        # END-labeled minute → bucket into the ISP containing (HH:MM-1).
        total_min = h * 60 + mi - 1
        if total_min < 0:
            continue
        isp = total_min // 15 + 1
        if isp < 1 or isp > 96:
            continue
        for field, col in field_map.items():
            v = rec.get(col)
            if v is None:
                continue
            try:
                buckets[isp][field].append(float(v))
            except (TypeError, ValueError):
                pass

    def _avg(xs):
        return (sum(xs) / len(xs)) if xs else None

    out = {}
    for isp in range(1, 97):
        if not any(buckets[isp][f] for f in field_map):
            continue
        out[isp] = {f: _avg(buckets[isp][f]) for f in field_map}
    return out


def _aggregate_11326_by_isp(date_str):
    """Backwards-compat alias: chart 11326 balancing activations by ISP."""
    return _aggregate_chart_by_isp(11326, date_str)


def get_activated_energy_damas(date_str):
    """DAMAS-shape activated balancing energy for HU, sourced live from RTDW.

    Returns a list of 15-min records with aFRR_Up/Down, mFRR_Up/Down (balancing)
    and nonBalancing_Up/Down (redispatch-style, chart 11322) in MW. Down
    volumes are returned as positive magnitudes (MAVIR signs them negative).
    Used as a live fallback when the monthly XLSX hasn't published yet.
    """
    try:
        bins = _aggregate_chart_by_isp(11326, date_str)
    except Exception as e:
        log.error(f"activated_energy (11326) failed for {date_str}: {e}")
        return []
    # Non-balancing regulation is on a separate chart. Its absence must not
    # break the main balancing output, so fetch it best-effort.
    try:
        nb_bins = _aggregate_chart_by_isp(11322, date_str)
    except Exception as e:
        log.warning(f"activated_energy (11322 non-balancing) failed for {date_str}: {e}")
        nb_bins = {}

    out = []
    for isp in sorted(bins):
        vals = bins[isp]
        nb = nb_bins.get(isp, {})
        start_min = (isp - 1) * 15
        h1, m1 = divmod(start_min, 60)
        end_min = start_min + 15
        h2, m2 = divmod(end_min, 60)
        afrr_up = vals.get('afrr_up')
        afrr_dn = vals.get('afrr_down')
        mfrr_up = vals.get('mfrr_up')
        mfrr_dn = vals.get('mfrr_down')
        nb_up = nb.get('nonbal_up')
        nb_dn = nb.get('nonbal_down')
        rec = {
            'id': f"hu_rtdw_act_{date_str}_{isp:03d}",
            'ISP': isp,
            'timeInterval': {
                'from': _rtdw_utc_iso(date_str, h1, m1),
                'to':   _rtdw_utc_iso(date_str, min(h2, 23), m2) if h2 < 24
                        else _rtdw_utc_iso(date_str, 23, 59),
            },
            'aFRR_Up':   None if afrr_up is None else round(abs(afrr_up), 3),
            'aFRR_Down': None if afrr_dn is None else round(abs(afrr_dn), 3),
            'mFRR_Up':   None if mfrr_up is None else round(abs(mfrr_up), 3),
            'mFRR_Down': None if mfrr_dn is None else round(abs(mfrr_dn), 3),
            'nonBalancing_Up':   None if nb_up is None else round(abs(nb_up), 3),
            'nonBalancing_Down': None if nb_dn is None else round(abs(nb_dn), 3),
        }
        out.append(rec)
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import json
    today_str = dt.datetime.now(TZ).strftime('%Y-%m-%d')
    print(f"Fetching live snapshot for {today_str}...")
    snap = live_snapshot(today_str)
    # Drop full series from print output (too noisy)
    preview = {k: v for k, v in snap.items() if k != 'series'}
    preview['series_lengths'] = {k: len(v) for k, v in snap['series'].items()}
    print(json.dumps(preview, indent=2, default=str))
