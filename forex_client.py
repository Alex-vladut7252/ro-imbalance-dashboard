"""ECB (European Central Bank) HUF/EUR reference rate client.

ECB is the authoritative free source for EUR exchange rates:
- Published once per TARGET working day around 16:00 CET
- No auth, no rate limits
- Used by EU institutions, banks, and regulatory reporting as the reference

We pull the daily XML, cache the parsed rate in memory for an hour, and
fall back to the most recent 90-day history if today's publication is
missing (weekends, holidays, ECB publication delay). A hardcoded last-
resort value prevents the dashboard from blanking when ECB is unreachable.
"""
import logging
import re
import time
from datetime import datetime
from threading import Lock

import requests

log = logging.getLogger(__name__)

DAILY_URL   = 'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml'
HIST_90_URL = 'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml'

# Fallback if ECB is unreachable AND we have no cached value. Matches the
# older hardcoded constant, updated here in one place if ever needed.
_FALLBACK_RATE = 400.0

_cache = {'rate': None, 'date': None, 'time': 0.0, 'source': None}
_cache_lock = Lock()
_CACHE_TTL_SECS = 60 * 60  # 1 hour — ECB publishes at most once per day


def _parse_daily_xml(text: str):
    """Return (rate, iso_date) from the single-day XML, or (None, None)."""
    date_m = re.search(r"<Cube\s+time='([^']+)'", text)
    rate_m = re.search(r"Cube\s+currency='HUF'\s+rate='([^']+)'", text)
    if not (date_m and rate_m):
        return None, None
    try:
        return float(rate_m.group(1)), date_m.group(1)
    except ValueError:
        return None, None


def _parse_hist_xml(text: str):
    """Return (rate, iso_date) for the most recent day with a HUF entry."""
    # The 90-day file is grouped by <Cube time='YYYY-MM-DD'>...</Cube>; for each
    # day block, find HUF rate. Return the latest (first) matching block.
    blocks = re.findall(
        r"<Cube\s+time='([^']+)'>([\s\S]*?)</Cube>", text)
    for date, body in blocks:
        m = re.search(r"currency='HUF'\s+rate='([^']+)'", body)
        if m:
            try:
                return float(m.group(1)), date
            except ValueError:
                continue
    return None, None


def get_huf_per_eur(force_refresh: bool = False) -> dict:
    """Return the latest HUF-per-EUR reference rate from ECB.

    Result shape:
        { 'rate': 364.89, 'date': '2026-04-23',
          'source': 'ECB',  # or 'ECB-hist' / 'cache' / 'fallback'
          'fetched_at': 1777412345, 'stale': False }
    """
    now = time.time()
    with _cache_lock:
        cached = dict(_cache) if _cache['rate'] is not None else None
    if cached and not force_refresh and (now - cached['time']) < _CACHE_TTL_SECS:
        return {'rate': cached['rate'], 'date': cached['date'],
                'source': cached['source'], 'fetched_at': cached['time'],
                'stale': False}

    # Try daily first
    try:
        r = requests.get(DAILY_URL, timeout=10)
        if r.status_code == 200:
            rate, date = _parse_daily_xml(r.text)
            if rate:
                with _cache_lock:
                    _cache.update(rate=rate, date=date, time=now, source='ECB')
                log.info(f"ECB HUF/EUR: {rate} ({date})")
                return {'rate': rate, 'date': date, 'source': 'ECB',
                        'fetched_at': now, 'stale': False}
    except Exception as e:
        log.warning(f"ECB daily fetch failed: {e}")

    # Fall back to 90-day history (weekends/holidays when daily may lag)
    try:
        r = requests.get(HIST_90_URL, timeout=10)
        if r.status_code == 200:
            rate, date = _parse_hist_xml(r.text)
            if rate:
                with _cache_lock:
                    _cache.update(rate=rate, date=date, time=now,
                                  source='ECB-hist')
                log.info(f"ECB HUF/EUR (hist): {rate} ({date})")
                return {'rate': rate, 'date': date, 'source': 'ECB-hist',
                        'fetched_at': now, 'stale': False}
    except Exception as e:
        log.warning(f"ECB hist fetch failed: {e}")

    # Last resort: return stale cache if we have one, else hardcoded fallback
    with _cache_lock:
        cached = dict(_cache) if _cache['rate'] is not None else None
    if cached:
        return {'rate': cached['rate'], 'date': cached['date'],
                'source': cached['source'] + '-stale',
                'fetched_at': cached['time'], 'stale': True}
    today = datetime.now().strftime('%Y-%m-%d')
    log.warning(f"ECB unreachable, falling back to {_FALLBACK_RATE} HUF/EUR")
    return {'rate': _FALLBACK_RATE, 'date': today, 'source': 'fallback',
            'fetched_at': now, 'stale': True}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import json
    print(json.dumps(get_huf_per_eur(), indent=2, default=str))
