"""Open-Meteo client — weather-derived wind & solar forecast.

Free, no API key, 7-day hourly forecasts updated 4× daily. Used as an
independent cross-check alongside ENTSO-E (TSO data) and MAVIR (TSO native).

Approach:
  - Fetch GHI (global horizontal irradiance) at the country's PV-weighted
    centroid, scale by installed PV capacity × performance ratio → MW
  - Fetch wind speed at 100 m hub height, run through a generic IEC-class-II
    power curve, scale by installed wind capacity → MW
  - Resample to 15-min by linear interpolation so it aligns with ENTSO-E

This is not as accurate as a real wind/solar power forecast service (Solcast,
Meteologica) — it's intended as a sanity check, not a production estimate.
A ±20 % discrepancy with ENTSO-E is normal and informative ("MAVIR thinks
the wind dies down faster than weather alone would suggest").
"""

import logging
import math
import time
from datetime import datetime, timedelta
from threading import Lock
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger(__name__)

# Country PV-fleet centroid (weighted toward where panels actually are)
# and installed capacity in MW (end-2025 figures, refreshed annually).
COUNTRY_CONFIG = {
    'HU': {'lat': 47.16, 'lon': 19.50, 'pv_mw': 8400, 'wind_mw': 330,  'tz': 'Europe/Budapest'},
    'AT': {'lat': 48.20, 'lon': 14.30, 'pv_mw': 4200, 'wind_mw': 3900, 'tz': 'Europe/Vienna'},
    'SK': {'lat': 48.40, 'lon': 19.30, 'pv_mw': 1700, 'wind_mw': 3,    'tz': 'Europe/Bratislava'},
    'RO': {'lat': 45.90, 'lon': 25.00, 'pv_mw': 2500, 'wind_mw': 3000, 'tz': 'Europe/Bucharest'},
    'CZ': {'lat': 49.80, 'lon': 15.50, 'pv_mw': 2700, 'wind_mw': 340,  'tz': 'Europe/Prague'},
    'PL': {'lat': 52.00, 'lon': 19.00, 'pv_mw': 18000, 'wind_mw': 9500, 'tz': 'Europe/Warsaw'},
    'RS': {'lat': 44.00, 'lon': 21.00, 'pv_mw': 200,  'wind_mw': 400,  'tz': 'Europe/Belgrade'},
    'HR': {'lat': 45.50, 'lon': 16.30, 'pv_mw': 350,  'wind_mw': 1000, 'tz': 'Europe/Zagreb'},
    'SI': {'lat': 46.10, 'lon': 14.80, 'pv_mw': 600,  'wind_mw': 6,    'tz': 'Europe/Ljubljana'},
}

# PV system performance ratio: average ratio of AC output to (GHI × DC peak).
# 0.78 = typical fleet-wide value accounting for inverter losses, soiling,
# temperature derating, and tilt/azimuth mismatch.
PV_PERFORMANCE_RATIO = 0.78

# Standard test irradiance (W/m²) used to define DC peak.
PV_STC_IRRADIANCE = 1000.0


def wind_power_curve(speed_ms):
    """Generic IEC class-II turbine power curve, returns capacity factor [0,1].

    cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s. Cubic ramp between cut-in
    and rated. Scaled per-turbine, applied to total installed capacity.
    """
    if speed_ms is None or speed_ms < 3:
        return 0.0
    if speed_ms >= 25:
        return 0.0
    if speed_ms >= 12:
        return 1.0
    return ((speed_ms - 3) / 9.0) ** 3


_cache = {}
_cache_lock = Lock()
_CACHE_TTL = 1800  # 30 min — Open-Meteo refreshes 4× daily


def _fetch_raw(country, date_str):
    """Hit Open-Meteo for a 1-day window in the country's local TZ."""
    cfg = COUNTRY_CONFIG.get(country)
    if not cfg:
        return None
    params = {
        'latitude':  cfg['lat'],
        'longitude': cfg['lon'],
        'hourly':    'shortwave_radiation,wind_speed_100m',
        'timezone':  cfg['tz'],
        'start_date': date_str,
        'end_date':   date_str,
    }
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast',
                         params=params, timeout=15)
        if r.status_code != 200:
            log.warning(f"Open-Meteo {country} {date_str}: HTTP {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        log.warning(f"Open-Meteo {country} {date_str}: {e}")
        return None


def get_forecast(country, date_str):
    """Return {'wind': {HH:MM: MW}, 'solar': {HH:MM: MW}, 'meta': {...}}.

    Hourly resolution from Open-Meteo, expanded to 15-min by holding each
    hourly value across its 4 quarters (no interpolation — keeps the data
    honest; ENTSO-E does the same when it publishes hourly values).
    """
    if country not in COUNTRY_CONFIG:
        return None
    key = (country, date_str)
    now = time.time()
    with _cache_lock:
        if key in _cache and now - _cache[key]['t'] < _CACHE_TTL:
            return _cache[key]['data']

    raw = _fetch_raw(country, date_str)
    if not raw or 'hourly' not in raw:
        return None
    cfg = COUNTRY_CONFIG[country]
    times = raw['hourly'].get('time', [])
    ghi   = raw['hourly'].get('shortwave_radiation', [])
    ws100 = raw['hourly'].get('wind_speed_100m', [])

    solar_mw = {}
    wind_mw  = {}
    for i, t in enumerate(times):
        # t is "YYYY-MM-DDTHH:MM" in country-local time
        try:
            hh_mm = t.split('T')[1][:5]
        except IndexError:
            continue
        # Solar: GHI(W/m²) / 1000 × installed_MW × performance_ratio
        if i < len(ghi) and ghi[i] is not None:
            mw = (float(ghi[i]) / PV_STC_IRRADIANCE) * cfg['pv_mw'] * PV_PERFORMANCE_RATIO
            solar_mw[hh_mm] = round(max(0.0, mw), 1)
        if i < len(ws100) and ws100[i] is not None:
            cf = wind_power_curve(float(ws100[i]) / 3.6)  # km/h → m/s
            wind_mw[hh_mm] = round(cf * cfg['wind_mw'], 1)

    # Expand hourly → 15-min by holding the hourly value across each quarter.
    # Keeps the data honest (no fake interpolation) and aligns with ENTSO-E keys.
    def _expand(hourly):
        out = {}
        for hm, v in hourly.items():
            h = hm[:2]
            for mm in ('00', '15', '30', '45'):
                out[f'{h}:{mm}'] = v
        return out

    out = {
        'wind':  _expand(wind_mw),
        'solar': _expand(solar_mw),
        'meta':  {
            'lat': cfg['lat'], 'lon': cfg['lon'],
            'pv_capacity_mw':   cfg['pv_mw'],
            'wind_capacity_mw': cfg['wind_mw'],
            'tz': cfg['tz'],
            'fetched_at': datetime.utcnow().isoformat() + 'Z',
            'method': 'GHI × capacity × PR + IEC II wind curve',
        },
    }
    with _cache_lock:
        _cache[key] = {'data': out, 't': now}
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import sys
    c = sys.argv[1] if len(sys.argv) > 1 else 'HU'
    d = sys.argv[2] if len(sys.argv) > 2 else datetime.utcnow().date().isoformat()
    res = get_forecast(c, d)
    if not res:
        print(f"no data for {c} {d}")
    else:
        print(f"{c} {d}: {len(res['solar'])} solar, {len(res['wind'])} wind pts")
        print(f"meta: {res['meta']}")
        midday = '12:00'
        print(f"  solar @ {midday}: {res['solar'].get(midday)} MW")
        print(f"  wind  @ {midday}: {res['wind'].get(midday)} MW")
