"""
Transelectrica live data client.
Fetches real-time SEN data, maintains history, and fetches imbalance prices.
"""

import requests
import re
import logging
from datetime import datetime
from collections import deque
import threading

log = logging.getLogger(__name__)

SEN_URL = 'https://www.transelectrica.ro/sen-filter'
IMBALANCE_URL = 'https://web.transelectrica.ro/energie_imp_exp/zilnic.html'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'ro-RO,ro;q=0.9,en;q=0.8',
}

INTERCONNECTIONS = {
    'HU': ['BEKE1', 'BEKE2', 'SAND'],
    'BG': ['VARN', 'DOBR', 'KOZL1', 'KOZL2'],
    'RS': ['DJER', 'PANCEVO21', 'PANCEVO22', 'KUSJ', 'SIP_', 'KIKI'],
    'MD': ['VULC', 'COSE', 'UNGE', 'CIOA', 'GOTE', 'IAS2', 'IS'],
    'UA': ['MUKA'],
}


def _sf(val):
    """Safe float."""
    if val is None or val == '' or val == 'N/A':
        return None
    val = str(val).replace(',', '.')
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class TranselectricaTracker:
    """Tracks SEN readings and provides history."""

    def __init__(self, max_history=200):
        self.history = deque(maxlen=max_history)
        self.lock = threading.Lock()
        self.last_live = None

    def fetch_and_store(self):
        """Fetch current SEN data and add to history."""
        data = self._fetch_sen()
        if data:
            with self.lock:
                self.last_live = data
                # Only add if timestamp changed (avoid duplicates)
                if not self.history or self.history[0]['timestamp'] != data['timestamp']:
                    self.history.appendleft(data)
        return data

    def get_live(self):
        return self.last_live

    def get_history(self, count=15):
        with self.lock:
            return list(self.history)[:count]

    def get_history_hours(self, hours=24):
        """Get all history within last N hours."""
        with self.lock:
            return list(self.history)

    def _fetch_sen(self):
        try:
            r = requests.get(SEN_URL, headers={
                **HEADERS,
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Referer': 'https://www.transelectrica.ro/widget/web/guest/sen-grafic/',
                'X-Requested-With': 'XMLHttpRequest',
            }, timeout=15, verify=False)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.error(f"SEN fetch failed: {e}")
            return None

        if not isinstance(data, list) or not data:
            return None

        entry = {}
        for item in data:
            if isinstance(item, dict):
                entry.update(item)

        if not entry:
            return None

        return self._parse(entry)

    def _parse(self, e):
        # Cross-border aggregation
        cross_border = {}
        for country, points in INTERCONNECTIONS.items():
            total = 0
            has_any = False
            for code in points:
                val = _sf(e.get(code))
                if val is not None:
                    total += val
                    has_any = True
            cross_border[country] = round(total, 1) if has_any else None

        now = datetime.now()
        return {
            'timestamp': now.strftime('%Y-%m-%d %H:%M'),
            'coal': _sf(e.get('CARB')),
            'storage': _sf(e.get('ISPOZ')),
            'hydrocarbon': _sf(e.get('GAZE')),
            'hydro': _sf(e.get('APE')),
            'nuclear': _sf(e.get('NUCL')),
            'wind': _sf(e.get('EOLIAN')),
            'solar': _sf(e.get('FOTO')),
            'biomass': _sf(e.get('BMASA')),
            'consumption': _sf(e.get('CONS')),
            'production': _sf(e.get('PROD')),
            'exchange': _sf(e.get('SOLD')),
            'hungary': cross_border.get('HU'),
            'bulgaria': cross_border.get('BG'),
            'serbia': cross_border.get('RS'),
            'ukraine': cross_border.get('UA'),
            'moldova': cross_border.get('MD'),
        }


def fetch_imbalance_prices():
    """Fetch imbalance prices from Transelectrica."""
    try:
        r = requests.get(IMBALANCE_URL, headers={
            **HEADERS, 'Accept': 'text/html,*/*',
        }, timeout=15, verify=False)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Imbalance fetch failed: {e}")
        return None

    return _parse_imbalance_html(r.text)


def _parse_imbalance_html(html):
    rows_raw = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S | re.I)
    headers = [
        'Time interval',
        'Estimated price negative imbalance [Lei/MWh]',
        'Estimated price positive imbalance [Lei/MWh]',
        'aFRR Up [MWh]',
        'aFRR Dn [MWh]',
        'mFRR Up [MWh]',
        'mFRR Dn [MWh]',
        'P aFRR Up [LEI]',
        'P aFRR Dn [LEI]',
        'P mFRR Up [LEI]',
        'P mFRR Dn [LEI]',
        'Estimated system imbalance [MWh]',
        'Realized consumption [MWh]',
        'Imbalance netting import [MWh]',
        'Imbalance netting export [MWh]',
    ]

    rows = []
    timestamp = None

    for row_html in rows_raw:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.S | re.I)
        if len(cells) < 6:
            continue
        texts = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]

        if not re.match(r'\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}', texts[0]):
            continue

        # Parse time interval
        m_from = re.match(r'(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})', texts[0])
        m_to = re.match(r'(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})', texts[1]) if len(texts) > 1 else None

        if m_from:
            day_from = int(m_from.group(1))
            mon_from = int(m_from.group(2))
            h_from = m_from.group(4)
            min_from = m_from.group(5)

            months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            date_label = f'{day_from:02d} {months[mon_from]}'

            time_from = f'{h_from}:{min_from}'
            time_to = ''
            if m_to:
                time_to = f'{m_to.group(4)}:{m_to.group(5)}'

            interval_label = f'{date_label} {time_from} - {time_to}'

            if timestamp is None:
                timestamp = f'{m_from.group(3)}-{m_from.group(2)}-{m_from.group(1)}T{h_from}:{min_from}:00'

        # Parse numeric values
        vals = []
        for t in texts[2:]:
            t = t.strip().replace(',', '')
            try:
                vals.append(float(t))
            except (ValueError, TypeError):
                vals.append(0.0)

        # Build row: [interval_label, neg_price, pos_price, ...]
        row = [interval_label]
        # The basic page has: export_energy, export_price, import_energy, import_price
        # Map to our expected format
        if len(vals) >= 4:
            export_energy = vals[0]
            export_price = vals[1]
            import_energy = vals[2]
            import_price = vals[3]

            # Neg imbalance price = import price (penalty for being short)
            # Pos imbalance price = export price (penalty for being long)
            neg_price = import_price
            pos_price = export_price

            # System imbalance estimate
            sys_imbalance = export_energy - import_energy

            row.extend([
                neg_price,       # Neg price
                pos_price,       # Pos price
                import_energy,   # aFRR Up (approx)
                export_energy,   # aFRR Dn (approx)
                0.0,             # mFRR Up
                0.0,             # mFRR Dn
                import_energy * import_price if import_price else 0,  # P aFRR Up
                export_energy * export_price if export_price else 0,  # P aFRR Dn
                0.0,             # P mFRR Up
                0.0,             # P mFRR Dn
                round(sys_imbalance, 3),  # System imbalance
                0.0,             # Realized consumption (N/A from this source)
                import_energy,   # Netting import
                export_energy,   # Netting export
            ])
        else:
            row.extend([0.0] * 14)

        rows.append(row)

    return {
        'headers': headers,
        'rows': rows,
        'timestamp': timestamp,
    }
