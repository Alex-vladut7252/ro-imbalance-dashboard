"""MAVIR (Hungarian TSO) balancing-data client.

Hungary doesn't publish a real-time JSON API for balancing prices the way
Romania's DAMAS does. Instead, MAVIR posts monthly XLSX reports to its
public Liferay portal under the "Settlement Unit Prices of Balance Energy"
section. This module:

  1. Discovers the per-month folder list from the public portal page.
  2. Downloads the two XLSX files per month (Up_* and Dw_* directions).
  3. Parses them into a DAMAS-compatible per-ISP structure so the
     existing frontend renderers can show Hungarian data with no JS changes.

Reference page:
    https://www.mavir.hu/web/riportok/settlement-unit-prices-of-balance-energy

URL pattern:
    https://www.mavir.hu/documents/187408084/{folderId}/{filename}.xlsx
    where filename = {Up|Dw}_BE_unitprice_{YYYYMMDD}_{YYYYMMDD}.xlsx

Currency: HUF/kWh in the source files. We convert to EUR/MWh (the unit the
Romanian dashboard uses) so the rendered cards stay comparable.
"""

import io
import logging
import re
import time
from datetime import datetime
from threading import Lock

import requests

log = logging.getLogger(__name__)

# Public portal page that lists all month-folders.
SETTLEMENT_PAGE = (
    'https://www.mavir.hu/web/riportok/settlement-unit-prices-of-balance-energy'
)
# Direct XLSX URLs hang off this base (repository id is constant for MAVIR).
DOC_BASE = 'https://www.mavir.hu/documents/187408084'

# Crude EUR/HUF rate. Updates monthly are fine — the MAVIR data itself
# already has weeks of lag, and absolute precision is not the goal.
# 1 EUR ~ 400 HUF (April 2026 ballpark). Override via MAVIR_HUF_PER_EUR env var.
import os as _os
HUF_PER_EUR = float(_os.environ.get('MAVIR_HUF_PER_EUR', '400'))


def _huf_per_kwh_to_eur_per_mwh(v):
    """HUF/kWh -> EUR/MWh. Returns None for None/NaN inputs."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # HUF/kWh × 1000 = HUF/MWh; ÷ HUF_PER_EUR = EUR/MWh
    return round(f * 1000.0 / HUF_PER_EUR, 4)


def _kwh_to_mwh(v):
    if v is None:
        return None
    try:
        return round(float(v) / 1000.0, 4)
    except (TypeError, ValueError):
        return None


_session_lock = Lock()
_session = None


def _get_session():
    """Lazy session with browser-like headers and cookie warm-up.

    MAVIR's CDN/firewall TLS-resets aggressive crawlers; a normal browser
    UA + a single warm-up GET to the homepage gets us a JSESSIONID and we
    can then download files cleanly.
    """
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        s = requests.Session()
        s.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/131.0.0.0 Safari/537.36'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,hu;q=0.8',
        })
        try:
            s.get('https://www.mavir.hu/', timeout=20)
        except Exception as e:
            log.warning(f"MAVIR session warm-up failed: {e}")
        _session = s
        return s


# ── File discovery ────────────────────────────────────────────────────

# Cache discovered folder map for an hour — MAVIR uploads at most monthly.
_folder_cache = {'data': None, 'time': 0}
_FOLDER_TTL = 3600


def list_month_folders():
    """Return dict {(year, month): folder_id} from the settlement portal page.

    Each month folder on MAVIR contains the Up_* and Dw_* XLSX files for
    that delivery month.
    """
    now = time.time()
    if _folder_cache['data'] and (now - _folder_cache['time']) < _FOLDER_TTL:
        return _folder_cache['data']

    s = _get_session()
    try:
        r = s.get(SETTLEMENT_PAGE, timeout=25)
        r.raise_for_status()
    except Exception as e:
        log.error(f"MAVIR settlement page fetch failed: {e}")
        return _folder_cache['data'] or {}

    body = r.text
    # The portal renders subfolder links like:
    #   .../document_library/hmRdMIdRT1ar/view/{folderId}? ...
    # with the folder name as the link text "{YYYYMM}".
    # Grab folderId + label by walking the document_library/.../view/<id> matches.
    out = {}
    pattern = re.compile(
        r'/document_library/hmRdMIdRT1ar/view/(\d+)[^>]*>\s*'
        r'<[^>]+>\s*</[^>]+>\s*</[^>]*>\s*(\d{6})\s*</a>'
    )
    # Fall back: simpler — walk all <a ...>YYYYMM</a> links and pair with folderId by URL.
    for m in re.finditer(
        r'href="([^"]*?/document_library/hmRdMIdRT1ar/view/(\d+)[^"]*?)"[^>]*>'
        r'(?:[^<]|<(?!/a>))*?(\d{6})(?:[^<]|<(?!/a>))*?</a>',
        body,
    ):
        folder_id = int(m.group(2))
        yyyymm = m.group(3)
        try:
            year = int(yyyymm[:4])
            month = int(yyyymm[4:6])
        except ValueError:
            continue
        out[(year, month)] = folder_id

    _folder_cache['data'] = out
    _folder_cache['time'] = now
    log.info(f"MAVIR: discovered {len(out)} month folders "
             f"(latest: {max(out) if out else 'none'})")
    return out


def list_files_in_folder(folder_id):
    """Given a month folder id, return a dict {direction: download_url}.

    direction is 'Up' or 'Dw'. Url is the fully-qualified document URL.
    """
    s = _get_session()
    url = (f'{SETTLEMENT_PAGE}/-/document_library/hmRdMIdRT1ar/view/{folder_id}')
    try:
        r = s.get(url, timeout=25)
        r.raise_for_status()
    except Exception as e:
        log.error(f"MAVIR folder {folder_id} fetch failed: {e}")
        return {}

    out = {}
    # Filenames look like: Up_BE_unitprice_YYYYMMDD_YYYYMMDD.xlsx
    # Pick the one with the latest end-date in case there's both a partial and final upload.
    candidates = {'Up': [], 'Dw': []}
    for m in re.finditer(
        r'(/documents/187408084/' + str(folder_id)
        + r'/(Up|Dw)_BE_unitprice_(\d{8})_(\d{8})\.xlsx)',
        r.text,
    ):
        path, direction, dfrom, dto = m.group(1), m.group(2), m.group(3), m.group(4)
        candidates[direction].append((dto, dfrom, f'https://www.mavir.hu{path}'))

    for direction, lst in candidates.items():
        if lst:
            lst.sort(reverse=True)  # latest dto first
            out[direction] = lst[0][2]
    return out


# ── Download + parse ──────────────────────────────────────────────────

# Cache parsed XLSX content keyed by url. Files are immutable once published
# so we never need to refetch within a session.
_xlsx_cache = {}


def fetch_and_parse(url):
    """Download an XLSX and parse into [{date, isp_str, ...}] rows."""
    if url in _xlsx_cache:
        return _xlsx_cache[url]
    s = _get_session()
    try:
        r = s.get(url, timeout=60)
        if r.status_code != 200 or len(r.content) < 1000:
            log.warning(f"MAVIR XLSX fetch bad: {url} -> HTTP {r.status_code}, {len(r.content)}B")
            return []
    except Exception as e:
        log.error(f"MAVIR XLSX fetch error: {url}: {e}")
        return []

    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(r.content), data_only=True, read_only=True)
        ws = wb.active
        # Header is at row 7. Data starts at row 9 (row 8 is blank).
        rows = []
        for row in ws.iter_rows(min_row=9, values_only=True):
            if not row or row[0] is None:
                continue
            date_val = row[0]
            isp_str = row[1]
            if not isp_str:
                continue
            # Normalize date to YYYY-MM-DD string
            if isinstance(date_val, datetime):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val).strip()[:10]
            # Cells (1-indexed in spec; 0-indexed here):
            #   col 4 (idx 3): mFRR+RR energy (kWh)
            #   col 5 (idx 4): mFRR+RR fee (HUF)
            #   col 6 (idx 5): aFRR energy (kWh)
            #   col 7 (idx 6): aFRR fee (HUF)
            #   col 8 (idx 7): Requested aFRR (kWh)
            #   col 13 (idx 12): Market incentive price (HUF/kWh)
            #   col 16 (idx 15): Price for BRP net energy balance (HUF/kWh)
            rows.append({
                'date': date_str,
                'isp_str': str(isp_str),
                'mfrr_rr_kwh': row[3],
                'mfrr_rr_huf': row[4],
                'afrr_kwh': row[5],
                'afrr_huf': row[6],
                'requested_afrr_kwh': row[7],
                'market_incentive_huf_kwh': row[12] if len(row) > 12 else None,
                'brp_price_huf_kwh': row[15] if len(row) > 15 else None,
                'system_state_kwh': row[14] if len(row) > 14 else None,
            })
        wb.close()
        _xlsx_cache[url] = rows
        log.info(f"MAVIR XLSX parsed: {url.split('/')[-1]} -> {len(rows)} rows")
        return rows
    except Exception as e:
        log.error(f"MAVIR XLSX parse error for {url}: {e}", exc_info=True)
        return []


# ── DAMAS-compatible adapters (consumed by app routes) ────────────────


def _isp_index(isp_str):
    """Convert 'HH:MM - HH:MM' to a 1-96 ISP number for the day."""
    m = re.match(r'(\d{1,2}):(\d{2})\s*-\s*\d{1,2}:\d{2}', isp_str or '')
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return h * 4 + mi // 15 + 1


_BUDAPEST_TZ = None
def _budapest_tz():
    global _BUDAPEST_TZ
    if _BUDAPEST_TZ is None:
        from zoneinfo import ZoneInfo
        _BUDAPEST_TZ = ZoneInfo('Europe/Budapest')
    return _BUDAPEST_TZ


def _utc_iso(date_str, hh, mm):
    """Budapest-local ISO stamp with DST-aware offset; rolls to next day for hh==24.

    Matches the format produced by the ENTSO-E and RTDW adapters so the
    frontend key-merge (moMakeKey) treats the same MTU as a single row.
    """
    from datetime import datetime, timedelta
    y, mo, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    day_off = 0
    hh_c = hh
    if hh_c >= 24:
        hh_c -= 24
        day_off = 1
    base = datetime(y, mo, d, hh_c, mm, tzinfo=_budapest_tz()) + timedelta(days=day_off)
    return base.isoformat(timespec='milliseconds')


def _row_to_damas_shape(row, direction_up):
    """Convert one parsed MAVIR row into a DAMAS-marginal-prices-shape dict.

    The frontend expects fields like aFRR_Up, aFRR_Down, mFRR_Up, mFRR_Down
    plus a `timeInterval` and stable `id`. We populate the matching direction
    and leave the opposite side null — the renderer already handles nulls.
    """
    isp = _isp_index(row['isp_str'])
    if isp is None:
        return None
    # Build start/end datetime strings for the frontend's utcToLocal helper.
    m = re.match(r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})', row['isp_str'])
    if not m:
        return None
    h1, m1, h2, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    # Stable record id keys the frontend's "has data" filter.
    rec_id = f"hu_{row['date']}_{isp:03d}_{('up' if direction_up else 'dw')}"

    afrr_kwh = row['afrr_kwh']
    afrr_huf = row['afrr_huf']
    mfrr_kwh = row['mfrr_rr_kwh']
    mfrr_huf = row['mfrr_rr_huf']

    # Marginal prices: HUF/kWh = (fee / kWh). Convert to EUR/MWh for the UI.
    def _to_eur_mwh(huf, kwh):
        if not huf or not kwh:
            return None
        try:
            ratio = abs(float(huf) / float(kwh))  # |HUF/kWh|
            return round(ratio * 1000.0 / HUF_PER_EUR, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    afrr_marg = _to_eur_mwh(afrr_huf, afrr_kwh)
    mfrr_marg = _to_eur_mwh(mfrr_huf, mfrr_kwh)

    interval = {
        'from': _utc_iso(row['date'], h1, m1),
        'to':   _utc_iso(row['date'], h2 + (24 if (h2, m2) <= (h1, m1) else 0), m2),
    }
    return {
        'id': rec_id,
        'ISP': isp,
        'timeInterval': interval,
        'aFRR_Up':         afrr_marg if direction_up else None,
        'aFRR_Down':       None      if direction_up else afrr_marg,
        'mFRR_Up':         mfrr_marg if direction_up else None,
        'mFRR_Down':       None      if direction_up else mfrr_marg,
        'mFRR_Up_Scheduled':   mfrr_marg if direction_up else None,
        'mFRR_Down_Scheduled': None      if direction_up else mfrr_marg,
        'mFRR_Up_Direct':      None,
        'mFRR_Down_Direct':    None,
    }


def _row_to_activated_shape(row_up, row_dw):
    """Combine a (UP, DOWN) row pair into a DAMAS activated-energy-shape dict."""
    base = row_up or row_dw
    if base is None:
        return None
    isp = _isp_index(base['isp_str'])
    if isp is None:
        return None
    m = re.match(r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})', base['isp_str'])
    if not m:
        return None
    h1, mi1, h2, mi2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))

    return {
        'id': f"hu_act_{base['date']}_{isp:03d}",
        'ISP': isp,
        'timeInterval': {
            'from': _utc_iso(base['date'], h1, mi1),
            # Midnight wrap: XLSX ISP 96 is "23:45 - 00:00"; pass hh+24 so
            # `_utc_iso` rolls the date forward.
            'to':   _utc_iso(base['date'], h2 + (24 if (h2, mi2) <= (h1, mi1) else 0), mi2),
        },
        'aFRR_Up':   _kwh_to_mwh((row_up or {}).get('afrr_kwh')),
        'aFRR_Down': _kwh_to_mwh((row_dw or {}).get('afrr_kwh')),
        'mFRR_Up':   _kwh_to_mwh((row_up or {}).get('mfrr_rr_kwh')),
        'mFRR_Down': _kwh_to_mwh((row_dw or {}).get('mfrr_rr_kwh')),
    }


def _row_to_imbalance_shape(row_up, row_dw):
    """Combine UP+DOWN rows into a DAMAS imbalance-prices-shape dict."""
    base = row_up or row_dw
    if base is None:
        return None
    isp = _isp_index(base['isp_str'])
    if isp is None:
        return None
    m = re.match(r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})', base['isp_str'])
    if not m:
        return None
    h1, mi1, h2, mi2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))

    pos_price = _huf_per_kwh_to_eur_per_mwh((row_up or {}).get('brp_price_huf_kwh'))
    neg_price = _huf_per_kwh_to_eur_per_mwh((row_dw or {}).get('brp_price_huf_kwh'))
    sys_state = (row_up or row_dw or {}).get('system_state_kwh')
    sys_imb_mw = _kwh_to_mwh(sys_state)

    return {
        'id': f"hu_imb_{base['date']}_{isp:03d}",
        'ISP': isp,
        'timeInterval': {
            'from': _utc_iso(base['date'], h1, mi1),
            # Midnight wrap: XLSX ISP 96 is "23:45 - 00:00"; pass hh+24 so
            # `_utc_iso` rolls the date forward.
            'to':   _utc_iso(base['date'], h2 + (24 if (h2, mi2) <= (h1, mi1) else 0), mi2),
        },
        'estimatedPriceNegativeImbalance': neg_price,
        'estimatedPricePositiveImbalance': pos_price,
        'estimatedSystemImbalance': sys_imb_mw,
        'realizedConsumption': None,
        'sumQup':       _kwh_to_mwh((row_up or {}).get('afrr_kwh')),
        'sumQdn':       _kwh_to_mwh((row_dw or {}).get('afrr_kwh')),
        'sumQupPup':    None,
        'sumQdownPdn':  None,
        'imbalanceNettingImport': None,
        'imbalanceNettingExport': None,
        'fcr': None,
        'type': 'Single',
    }


# ── Public API consumed by app.py ─────────────────────────────────────


def _get_month_rows(year, month):
    """Return ([up_rows], [dw_rows]) for the given month, downloading files if needed."""
    folders = list_month_folders()
    folder_id = folders.get((year, month))
    if not folder_id:
        log.warning(f"MAVIR: no folder for {year}-{month:02d}")
        return [], []
    files = list_files_in_folder(folder_id)
    up_rows = fetch_and_parse(files['Up']) if files.get('Up') else []
    dw_rows = fetch_and_parse(files['Dw']) if files.get('Dw') else []
    return up_rows, dw_rows


def _filter_to_date(rows, date_str):
    return [r for r in rows if r['date'] == date_str]


def _pair_by_isp(up_rows, dw_rows):
    """Yield (up_row, dw_row) tuples aligned by ISP, with None for missing sides."""
    by_isp = {}
    for r in up_rows:
        idx = _isp_index(r['isp_str'])
        if idx: by_isp[idx] = [r, None]
    for r in dw_rows:
        idx = _isp_index(r['isp_str'])
        if idx:
            if idx in by_isp: by_isp[idx][1] = r
            else: by_isp[idx] = [None, r]
    for idx in sorted(by_isp):
        yield by_isp[idx][0], by_isp[idx][1]


def _normalize_date(date_arg):
    """Accept either a YYYY-MM-DD local date or a YYYY-MM-DDTHH... ISO timestamp.

    Always returns the local Budapest delivery-day key. Important: callers
    must pass the local date, NOT a UTC-converted one — otherwise the date
    rolls back by 1 day at midnight local time.
    """
    return (date_arg or '')[:10]


def get_marginal_prices_overview(date_arg, _date_to=None):
    """DAMAS-shape marginal prices for one Budapest local delivery day."""
    date_str = _normalize_date(date_arg)
    y, m = int(date_str[:4]), int(date_str[5:7])
    up_rows, dw_rows = _get_month_rows(y, m)
    up_today = _filter_to_date(up_rows, date_str)
    dw_today = _filter_to_date(dw_rows, date_str)
    out = []
    for up, dw in _pair_by_isp(up_today, dw_today):
        if up is not None:
            r = _row_to_damas_shape(up, direction_up=True)
            if r: out.append(r)
        if dw is not None:
            r = _row_to_damas_shape(dw, direction_up=False)
            if r: out.append(r)
    return out


def get_activated_balancing_energy_overview(date_arg, _date_to=None):
    """DAMAS-shape activated balancing energy for one Budapest local day."""
    date_str = _normalize_date(date_arg)
    y, m = int(date_str[:4]), int(date_str[5:7])
    up_rows, dw_rows = _get_month_rows(y, m)
    up_today = _filter_to_date(up_rows, date_str)
    dw_today = _filter_to_date(dw_rows, date_str)
    out = []
    for up, dw in _pair_by_isp(up_today, dw_today):
        rec = _row_to_activated_shape(up, dw)
        if rec: out.append(rec)
    return out


def get_estimated_imbalance_prices(date_arg, _date_to=None):
    """DAMAS-shape imbalance prices + system imbalance for one Budapest local day."""
    date_str = _normalize_date(date_arg)
    y, m = int(date_str[:4]), int(date_str[5:7])
    up_rows, dw_rows = _get_month_rows(y, m)
    up_today = _filter_to_date(up_rows, date_str)
    dw_today = _filter_to_date(dw_rows, date_str)
    out = []
    for up, dw in _pair_by_isp(up_today, dw_today):
        rec = _row_to_imbalance_shape(up, dw)
        if rec: out.append(rec)
    return out


# ── Smoke test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    folders = list_month_folders()
    print(f"Found {len(folders)} folders. Latest: {max(folders) if folders else None}")
    if folders:
        year, month = max(folders)
        files = list_files_in_folder(folders[(year, month)])
        print(f"Files in latest folder: {files}")
        # Try fetching marginal prices for the most recent day this folder covers
        from datetime import date
        test_date = date(year, month, 1).isoformat()
        items = get_marginal_prices_overview(f"{test_date}T00:00:00.000Z", f"{test_date}T23:59:59.000Z")
        print(f"\nMarginal prices for {test_date}: {len(items)} records")
        if items:
            print(f"  first: {items[0]}")
            non_zero = [r for r in items if any(v for k, v in r.items() if k.startswith(('aFRR','mFRR')) and v)]
            print(f"  non-zero records: {len(non_zero)}")
            if non_zero:
                print(f"  sample non-zero: {non_zero[len(non_zero)//2]}")
