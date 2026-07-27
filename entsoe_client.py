"""
ENTSO-E Transparency Platform API client.
Uses the new POST-based API (transparency.entsoe.eu) for forecasts,
and the old XML REST API (web-api.tp.entsoe.eu) for actual generation data.
Returns 15-min interval data mapped as {"HH:MM": value} dicts.
"""

import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging

log = logging.getLogger(__name__)

RO_TZ = ZoneInfo('Europe/Bucharest')
UTC_TZ = ZoneInfo('UTC')

BASE_URL = 'https://transparency.entsoe.eu'
XML_API_URL = 'https://web-api.tp.entsoe.eu/api'
RO_DOMAIN = '10YRO-TEL------P'

# Per-country ENTSO-E config. Add entries here for new countries.
# `tz` is the delivery-day local timezone; `area` is the ENTSO-E BZN code.
COUNTRY_CONFIG = {
    'RO': {'area': 'BZN|10YRO-TEL------P', 'domain': '10YRO-TEL------P',
           'tz': ZoneInfo('Europe/Bucharest')},
    'HU': {'area': 'BZN|10YHU-MAVIR----U', 'domain': '10YHU-MAVIR----U',
           'tz': ZoneInfo('Europe/Budapest')},
    # Hungary's neighbors / same synchronous block — used by the Forecast Hub
    'AT': {'area': 'BZN|10YAT-APG------L', 'domain': '10YAT-APG------L',
           'tz': ZoneInfo('Europe/Vienna')},
    'SK': {'area': 'BZN|10YSK-SEPS-----K', 'domain': '10YSK-SEPS-----K',
           'tz': ZoneInfo('Europe/Bratislava')},
    'CZ': {'area': 'BZN|10YCZ-CEPS-----N', 'domain': '10YCZ-CEPS-----N',
           'tz': ZoneInfo('Europe/Prague')},
    'PL': {'area': 'BZN|10YPL-AREA-----S', 'domain': '10YPL-AREA-----S',
           'tz': ZoneInfo('Europe/Warsaw')},
    'RS': {'area': 'BZN|10YCS-SERBIATSOV', 'domain': '10YCS-SERBIATSOV',
           'tz': ZoneInfo('Europe/Belgrade')},
    'HR': {'area': 'BZN|10YHR-HEP------M', 'domain': '10YHR-HEP------M',
           'tz': ZoneInfo('Europe/Zagreb')},
    'SI': {'area': 'BZN|10YSI-ELES-----O', 'domain': '10YSI-ELES-----O',
           'tz': ZoneInfo('Europe/Ljubljana')},
}

# ENTSO-E PSR type codes
PSR_TYPES = {
    'B01': 'biomass', 'B02': 'lignite', 'B04': 'gas', 'B05': 'hard_coal',
    'B06': 'oil', 'B09': 'geothermal', 'B10': 'hydro_pumped',
    'B11': 'hydro_ror', 'B12': 'hydro_reservoir', 'B14': 'nuclear',
    'B15': 'other_renewable', 'B16': 'solar', 'B17': 'waste',
    'B18': 'wind_offshore', 'B19': 'wind_onshore', 'B20': 'other',
}


class EntsoeClient:
    RO = 'BZN|10YRO-TEL------P'

    # Keep old URLS for status page compatibility
    URLS = [BASE_URL]

    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Origin': 'https://transparency.entsoe.eu',
            'Referer': 'https://transparency.entsoe.eu/',
        })

    def _ro_day_utc_range(self, date_str):
        """Convert a Romanian date string to UTC start/end covering the full local day."""
        return self._day_utc_range(date_str, 'RO')

    def _day_utc_range(self, date_str, country='RO'):
        """Convert a date string to UTC start/end covering the country's local delivery day."""
        tz = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])['tz']
        local_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=tz)
        local_end = local_date + timedelta(days=1)
        return local_date.astimezone(UTC_TZ), local_end.astimezone(UTC_TZ)

    def _post(self, endpoint, date_str, timeout=15, country='RO'):
        """POST to the new ENTSO-E API with retry on transient errors."""
        import time as _time
        utc_start, utc_end = self._day_utc_range(date_str, country)
        date_from = utc_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        date_to = utc_end.strftime('%Y-%m-%dT%H:%M:%SZ')

        area = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])['area']
        body = {
            'dateTimeRange': {'from': date_from, 'to': date_to},
            'areaList': [area],
        }

        url = f"{BASE_URL}/{endpoint}"
        last_err = None

        for attempt in range(3):
            try:
                r = self.session.post(url, json=body, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    return data
                last_err = f"HTTP {r.status_code}"
                if r.status_code in (500, 502, 503):
                    _time.sleep(2)  # Brief wait before retry on server error
                    continue
                break  # Don't retry client errors
            except requests.exceptions.Timeout:
                last_err = "timeout"
                continue
            except requests.exceptions.ConnectionError:
                last_err = "connection error"
                _time.sleep(1)
                continue
            except Exception as e:
                last_err = str(e)
                break

        if last_err:
            log.warning(f"ENTSO-E {endpoint}: {last_err}")
        return None

    def _parse_points(self, data, col_name, country='RO'):
        """Parse new API response into {"HH:MM": value} dict.

        The new API returns data with:
        - instanceList[0].curveData.periodList[0].pointMap: {"0": [...], "1": [...]}
        - metaData: [{"code": "COL_NAME"}, ...]
        - Points are 0-indexed, values array matches metaData order.
        - timeInterval.from gives the start time of the period.
        """
        if not data:
            return {}

        instances = data.get('instanceList', [])
        meta = data.get('metaData', [])
        if not instances:
            return {}

        # Find column index
        col_idx = None
        for i, m in enumerate(meta):
            if m.get('code') == col_name:
                col_idx = i
                break
        if col_idx is None:
            return {}

        result = {}
        for inst in instances:
            time_interval = inst.get('timeInterval', {})
            start_str = time_interval.get('from', '')
            storage_tz = inst.get('storageInfo', {}).get('timeZone', 'EET')

            # Parse start time
            if not start_str:
                continue
            start_str = start_str.replace('Z', '+00:00')
            try:
                start_utc = datetime.fromisoformat(start_str)
            except ValueError:
                continue
            if start_utc.tzinfo is None:
                start_utc = start_utc.replace(tzinfo=UTC_TZ)

            curve = inst.get('curveData', {})
            for period in curve.get('periodList', []):
                resolution = period.get('resolution', 'PT60M')
                if 'PT15M' in resolution:
                    delta = timedelta(minutes=15)
                elif 'PT30M' in resolution:
                    delta = timedelta(minutes=30)
                else:
                    delta = timedelta(hours=1)

                point_map = period.get('pointMap', {})
                for idx_str, values in point_map.items():
                    idx = int(idx_str)
                    if col_idx >= len(values) or values[col_idx] is None:
                        continue
                    try:
                        val = float(values[col_idx])
                    except (ValueError, TypeError):
                        continue

                    ts_utc = start_utc + delta * idx
                    local_tz = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])['tz']
                    ts_local = ts_utc.astimezone(local_tz)
                    key = ts_local.strftime('%H:%M')
                    result[key] = val

        return result

    # ── Public API ──────────────────────────────────────────

    def get_wind_solar_all_columns(self, date_str, country='RO'):
        """Return ALL forecast columns ENTSO-E publishes, not just DAY_AHEAD.

        ENTSO-E's wind & solar endpoints return 4 columns per timestamp:
          - DAY_AHEAD: official D-1 forecast submitted by ~18:00 CET
          - INTRADAY:  refreshed during the delivery day
          - CURRENT:   most recent update (often == INTRADAY)
          - ACTUAL:    measured production (only past hours)
        Different public sites show different columns, which is why "the data
        looks different everywhere." This method returns the full picture so
        the frontend can show all four side-by-side and label provenance.

        Returns:
            {
              'wind':  {'DAY_AHEAD': {HH:MM: MW}, 'INTRADAY': {...}, ...},
              'solar': {...},
              'meta':  {'resolution_min': 15, 'columns': [...], 'fetched_at': iso}
            }
        """
        from datetime import datetime as _dt
        out = {
            'wind':  {'DAY_AHEAD': {}, 'INTRADAY': {}, 'CURRENT': {}, 'ACTUAL': {}},
            'solar': {'DAY_AHEAD': {}, 'INTRADAY': {}, 'CURRENT': {}, 'ACTUAL': {}},
            'meta':  {'resolution_min': 15, 'fetched_at': _dt.utcnow().isoformat() + 'Z',
                      'country': country, 'date': date_str},
        }
        for kind, key in [('onshore', 'wind'), ('solar', 'solar')]:
            data = self._post(f'generation/forecast/windAndSolar/{kind}/load',
                              date_str, country=country)
            for col in ('DAY_AHEAD', 'INTRADAY', 'CURRENT', 'ACTUAL'):
                out[key][col] = self._parse_points(data, col, country=country)
        return out

    def get_forecasts(self, date_str, country='RO'):
        """Get all forecast data for the SEN chart."""
        result = {
            'wind': {}, 'solar': {},
            'consumption': {},       # A65 = load forecast (+ actual where available)
            'total_production': {},   # same as total_generation (compat)
            'total_generation': {},   # A71 = generation forecast (+ actual where available)
            'post_actual_load': {},   # Actual load from POST API (separate)
            'post_actual_gen': {},    # Actual generation from POST API (separate)
        }

        # Wind onshore forecast (day-ahead)
        wind_data = self._post('generation/forecast/windAndSolar/onshore/load', date_str, country=country)
        result['wind'] = self._parse_points(wind_data, 'DAY_AHEAD', country=country)

        # Solar forecast (day-ahead)
        solar_data = self._post('generation/forecast/windAndSolar/solar/load', date_str, country=country)
        result['solar'] = self._parse_points(solar_data, 'DAY_AHEAD', country=country)

        # Load forecast (= consumption)
        load_data = self._post('load/total/dayAhead/load', date_str, country=country)
        consumption = self._parse_points(load_data, 'TOTAL_LOAD_FORECAST', country=country)
        actual_load = self._parse_points(load_data, 'TOTAL_LOAD_ACTUAL', country=country)
        result['post_actual_load'] = dict(actual_load)  # keep separate copy
        for k, v in actual_load.items():
            consumption[k] = v
        result['consumption'] = consumption
        result['total_production'] = consumption  # compat

        # Generation forecast (= production)
        gen_data = self._post('generation/forecast/dayAhead/load', date_str, country=country)
        generation = self._parse_points(gen_data, 'GENERATION_FORECAST', country=country)
        actual_gen = self._parse_points(gen_data, 'ACTUAL_GENERATION', country=country)
        result['post_actual_gen'] = dict(actual_gen)  # keep separate copy
        for k, v in actual_gen.items():
            generation[k] = v
        result['total_generation'] = generation
        result['total_production'] = generation  # alias for compat

        total_pts = sum(len(v) for v in result.values() if isinstance(v, dict))
        log.info(f"ENTSO-E new API ({country}): {total_pts} total points for {date_str} "
                 f"(wind={len(result['wind'])}, solar={len(result['solar'])}, "
                 f"cons={len(result['consumption'])}, gen={len(result['total_generation'])}, "
                 f"act_load={len(result['post_actual_load'])}, act_gen={len(result['post_actual_gen'])})")

        return result

    # Alias for background thread compat
    def get_forecasts_aggressive(self, date_str, country='RO'):
        return self.get_forecasts(date_str, country=country)

    def get_dam_prices(self, date_str, country='RO'):
        """DAM prices mapped to {"HH:MM": EUR/MWh} in the country's local timezone."""
        data = self._post('market/energyPrices/load', date_str, country=country)
        if not data:
            return {}

        # The energy prices endpoint may return multiple instances (different areas/currencies)
        # Find the one with prices
        instances = data.get('instanceList', [])
        meta = data.get('metaData', [])

        prices = {}
        col_idx = 0  # Usually the first (and only) column 'CAPACITY' = price

        for inst in instances:
            time_interval = inst.get('timeInterval', {})
            start_str = time_interval.get('from', '').replace('Z', '+00:00')
            if not start_str:
                continue
            try:
                start_utc = datetime.fromisoformat(start_str)
            except ValueError:
                continue
            if start_utc.tzinfo is None:
                start_utc = start_utc.replace(tzinfo=UTC_TZ)

            curve = inst.get('curveData', {})
            for period in curve.get('periodList', []):
                resolution = period.get('resolution', 'PT60M')
                if 'PT15M' in resolution:
                    delta = timedelta(minutes=15)
                elif 'PT30M' in resolution:
                    delta = timedelta(minutes=30)
                else:
                    delta = timedelta(hours=1)

                point_map = period.get('pointMap', {})
                for idx_str, values in point_map.items():
                    idx = int(idx_str)
                    if col_idx >= len(values) or values[col_idx] is None:
                        continue
                    try:
                        val = float(values[col_idx])
                    except (ValueError, TypeError):
                        continue

                    ts_utc = start_utc + delta * idx
                    local_tz = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])['tz']
                    ts_local = ts_utc.astimezone(local_tz)
                    key = ts_local.strftime('%H:%M')
                    prices[key] = val

        return prices

    def get_dam_prices_hourly(self, date_str, country='RO'):
        """DAM prices as list of {hour, price_eur}."""
        prices_map = self.get_dam_prices(date_str, country=country)
        result = []
        for h in range(24):
            key = f'{h:02d}:00'
            price = prices_map.get(key)
            if price is not None:
                result.append({'hour': h, 'price_eur': price})
        return result

    # ── XML REST API (actual generation data) ──────────────────

    def _xml_get(self, params, timeout=45):
        """GET from the old XML REST API. Single attempt."""
        params['securityToken'] = self.api_key
        try:
            r = requests.get(XML_API_URL, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.text
            log.warning(f"ENTSO-E XML API: HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            log.warning("ENTSO-E XML API: timeout")
        except requests.exceptions.ConnectionError:
            log.warning("ENTSO-E XML API: connection error")
        except Exception as e:
            log.warning(f"ENTSO-E XML API: {e}")
        return None

    def _parse_xml_timeseries(self, xml_text, target_date_str, country='RO'):
        """Parse XML timeseries into {psr_type: {"HH:MM": value}} dict.

        Filters to only include points that fall within the target Romanian date.
        """
        if not xml_text:
            return {}

        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            log.error(f"XML parse error: {e}")
            return {}

        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        result = {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            # Get PSR type
            psr_el = ts.find('.//ns:MktPSRType/ns:psrType', ns)
            psr_code = psr_el.text if psr_el is not None else 'unknown'
            psr_name = PSR_TYPES.get(psr_code, psr_code)

            for period in ts.findall('.//ns:Period', ns):
                ti = period.find('ns:timeInterval', ns)
                if ti is None:
                    continue
                start_el = ti.find('ns:start', ns)
                if start_el is None:
                    continue

                start_str = start_el.text.replace('Z', '+00:00')
                try:
                    start_utc = datetime.fromisoformat(start_str)
                except ValueError:
                    continue
                if start_utc.tzinfo is None:
                    start_utc = start_utc.replace(tzinfo=UTC_TZ)

                res_el = period.find('ns:resolution', ns)
                resolution = res_el.text if res_el is not None else 'PT60M'
                if 'PT15M' in resolution:
                    delta = timedelta(minutes=15)
                elif 'PT30M' in resolution:
                    delta = timedelta(minutes=30)
                else:
                    delta = timedelta(hours=1)

                if psr_name not in result:
                    result[psr_name] = {}

                for point in period.findall('ns:Point', ns):
                    pos_el = point.find('ns:position', ns)
                    qty_el = point.find('ns:quantity', ns)
                    if pos_el is None or qty_el is None:
                        continue
                    try:
                        pos = int(pos_el.text)
                        qty = float(qty_el.text)
                    except (ValueError, TypeError):
                        continue

                    ts_utc = start_utc + delta * (pos - 1)
                    local_tz = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])['tz']
                    ts_local = ts_utc.astimezone(local_tz)

                    # Only include points for the target date
                    if ts_local.date() != target_date:
                        continue

                    key = ts_local.strftime('%H:%M')
                    result[psr_name][key] = qty

        return result

    def _parse_xml_load(self, xml_text, target_date_str, country='RO'):
        """Parse XML load timeseries into {"HH:MM": value} dict."""
        if not xml_text:
            return {}

        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            log.error(f"XML parse error: {e}")
            return {}

        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        result = {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            for period in ts.findall('.//ns:Period', ns):
                ti = period.find('ns:timeInterval', ns)
                if ti is None:
                    continue
                start_el = ti.find('ns:start', ns)
                if start_el is None:
                    continue

                start_str = start_el.text.replace('Z', '+00:00')
                try:
                    start_utc = datetime.fromisoformat(start_str)
                except ValueError:
                    continue
                if start_utc.tzinfo is None:
                    start_utc = start_utc.replace(tzinfo=UTC_TZ)

                res_el = period.find('ns:resolution', ns)
                resolution = res_el.text if res_el is not None else 'PT60M'
                if 'PT15M' in resolution:
                    delta = timedelta(minutes=15)
                elif 'PT30M' in resolution:
                    delta = timedelta(minutes=30)
                else:
                    delta = timedelta(hours=1)

                for point in period.findall('ns:Point', ns):
                    pos_el = point.find('ns:position', ns)
                    qty_el = point.find('ns:quantity', ns)
                    if pos_el is None or qty_el is None:
                        continue
                    try:
                        pos = int(pos_el.text)
                        qty = float(qty_el.text)
                    except (ValueError, TypeError):
                        continue

                    ts_utc = start_utc + delta * (pos - 1)
                    local_tz = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])['tz']
                    ts_local = ts_utc.astimezone(local_tz)

                    if ts_local.date() != target_date:
                        continue

                    key = ts_local.strftime('%H:%M')
                    result[key] = qty

        return result

    def get_actual_generation(self, date_str, post_api_data=None, country='RO'):
        """Get actual generation data combining XML API (per-type) and POST API (totals).

        Uses XML A75 for actual wind/solar (reliable per-type data).
        Uses POST API ACTUAL_GENERATION/TOTAL_LOAD_ACTUAL for totals (reliable aggregates).

        Returns dict with keys:
            actual_wind, actual_solar, actual_production, actual_consumption,
            gen_by_type (detailed breakdown)
        """
        # XML API requires midnight-aligned UTC single-day ranges.
        # Romania is UTC+2/3, so local 00:00-02:59 falls in the previous UTC day.
        # We use single-day UTC range (covers 03:00+ local); 00:00-02:59 gap is minor.
        local_date = datetime.strptime(date_str, '%Y-%m-%d')
        next_day = local_date + timedelta(days=1)

        result = {
            'actual_wind': {},
            'actual_solar': {},
            'actual_production': {},
            'actual_consumption': {},
            'gen_by_type': {},
        }

        # A75: Actual generation per type (for wind/solar breakdown)
        period_start = local_date.strftime('%Y%m%d') + '0000'
        period_end = next_day.strftime('%Y%m%d') + '0000'
        country_domain = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])['domain']
        xml = self._xml_get({
            'documentType': 'A75',
            'processType': 'A16',
            'in_Domain': country_domain,
            'periodStart': period_start,
            'periodEnd': period_end,
        })
        if xml:
            gen_data = self._parse_xml_timeseries(xml, date_str, country=country)
            result['gen_by_type'] = gen_data
            result['actual_wind'] = gen_data.get('wind_onshore', {})
            result['actual_solar'] = gen_data.get('solar', {})
            log.info(f"ENTSO-E XML A75 ({country}): wind={len(result['actual_wind'])}, "
                     f"solar={len(result['actual_solar'])} pts for {date_str}")

        # For total production and consumption, use POST API actuals (more reliable than A75 sum)
        if post_api_data:
            result['actual_production'] = post_api_data.get('post_actual_gen', {})
            result['actual_consumption'] = post_api_data.get('post_actual_load', {})
            log.info(f"POST API actuals: prod={len(result['actual_production'])}, "
                     f"cons={len(result['actual_consumption'])} pts for {date_str}")
        else:
            # Fallback: fetch from A65 XML
            xml = self._xml_get({
                'documentType': 'A65',
                'processType': 'A16',
                'outBiddingZone_Domain': country_domain,
                'periodStart': period_start,
                'periodEnd': period_end,
            })
            if xml:
                result['actual_consumption'] = self._parse_xml_load(xml, date_str, country=country)
                log.info(f"ENTSO-E XML A65 ({country}): consumption={len(result['actual_consumption'])} pts")

        return result

    def get_imbalance_prices(self, date_str, country='HU'):
        """Fetch imbalance prices (A85) per 15-min MTU, in native currency.

        ENTSO-E zips A85 responses, so we decompress. Two TimeSeries are
        returned per day for HU — category A04 (positive imbalance price) and
        A05 (negative imbalance price). Currency: HUF for HU.

        Returns:
            {
                'pos': {'HH:MM': price, ...},   # category A04 — paid to BRP on surplus
                'neg': {'HH:MM': price, ...},   # category A05 — paid by BRP on deficit
                'currency': 'HUF',
                'date': date_str,
            }
        """
        import io, zipfile
        out = {'pos': {}, 'neg': {}, 'currency': None, 'date': date_str}
        cfg = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['HU'])
        domain = cfg['domain']
        tz = cfg['tz']

        local_d = datetime.strptime(date_str, '%Y-%m-%d')
        start_local = local_d.replace(tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        period_start = start_local.astimezone(ZoneInfo('UTC')).strftime('%Y%m%d%H%M')
        period_end = end_local.astimezone(ZoneInfo('UTC')).strftime('%Y%m%d%H%M')

        try:
            params = {
                'documentType': 'A85',
                'controlArea_Domain': domain,
                'periodStart': period_start,
                'periodEnd': period_end,
                'securityToken': self.api_key,
            }
            r = requests.get(XML_API_URL, params=params, timeout=45)
            if r.status_code != 200 or len(r.content) < 200:
                log.warning(f"A85 {country} {date_str}: HTTP {r.status_code}")
                return out
            # ENTSO-E returns a ZIP archive for A85
            try:
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                xml = zf.read(zf.namelist()[0]).decode('utf-8')
            except zipfile.BadZipFile:
                # Sometimes returned as plain XML
                xml = r.text
        except Exception as e:
            log.error(f"A85 fetch {country} {date_str}: {e}")
            return out

        if '<TimeSeries>' not in xml:
            return out

        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:4'}
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as e:
            # Some doc versions use an older namespace; fall back by stripping NS.
            log.warning(f"A85 namespace parse retry: {e}")
            xml_no_ns = re.sub(r'\sxmlns="[^"]+"', '', xml, count=1)
            root = ET.fromstring(xml_no_ns)
            ns = {'ns': ''}

        def _find(el, path):
            if ns.get('ns'):
                return el.find('ns:' + path, ns)
            return el.find(path)

        def _findall(el, path):
            if ns.get('ns'):
                return el.findall('ns:' + path, ns)
            return el.findall(path)

        def _text(el, path):
            node = _find(el, path)
            return node.text if node is not None else None

        for ts in _findall(root, 'TimeSeries'):
            currency = _text(ts, 'currency_Unit.name')
            if currency:
                out['currency'] = currency
            per = _find(ts, 'Period')
            if per is None:
                continue
            resolution = _text(per, 'resolution') or 'PT15M'
            step_min = int(''.join(c for c in resolution if c.isdigit()) or '15')
            start_iso = _text(_find(per, 'timeInterval'), 'start')
            period_start_utc = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
            for p in _findall(per, 'Point'):
                pos_s = _text(p, 'position')
                price_s = _text(p, 'imbalance_Price.amount')
                cat = _text(p, 'imbalance_Price.category')
                if pos_s is None or price_s is None:
                    continue
                try:
                    price = float(price_s)
                    pos = int(pos_s)
                except ValueError:
                    continue
                pt_utc = period_start_utc + timedelta(minutes=step_min * (pos - 1))
                pt_local = pt_utc.astimezone(tz)
                if pt_local.date() != local_d.date():
                    continue
                hm = pt_local.strftime('%H:%M')
                bucket = 'pos' if cat == 'A04' else ('neg' if cat == 'A05' else None)
                if bucket:
                    out[bucket][hm] = price

        log.info(f"ENTSO-E A85 ({country}) {date_str}: "
                 f"pos={len(out['pos'])}, neg={len(out['neg'])} pts currency={out['currency']}")
        return out

    def get_aggregated_balancing_bids(self, date_str, country='RO'):
        """Fetch Aggregated Balancing Energy Bids (A24) per 15-min MTU.

        Returns the total bid VOLUME offered per MTU per reserve type and direction.
        This is what ENTSO-E's transparency page shows as "Total: X MW" — the
        offered capacity before acceptance. Individual bid prices are not
        exposed via the public REST API (those require an authenticated session
        on transparency.entsoe.eu), so VWAP and bid count can't be derived here.

        Availability:
          - HU: aFRR (A51) + mFRR (A47), both directions
          - RO: aFRR only (A47 not published for RO)

        Returns:
            {
                'aFRR_Up':   {'HH:MM': mw, ...},
                'aFRR_Down': {'HH:MM': mw, ...},
                'mFRR_Up':   {'HH:MM': mw, ...},
                'mFRR_Down': {'HH:MM': mw, ...},
                'unit':      'MW',
                'date':      date_str,
            }
        """
        import io, zipfile
        out = {'aFRR_Up': {}, 'aFRR_Down': {}, 'mFRR_Up': {}, 'mFRR_Down': {},
               'unit': 'MW', 'date': date_str}
        cfg = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])
        domain = cfg['domain']
        tz = cfg['tz']

        local_d = datetime.strptime(date_str, '%Y-%m-%d')
        start_local = local_d.replace(tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        period_start = start_local.astimezone(ZoneInfo('UTC')).strftime('%Y%m%d%H%M')
        period_end = end_local.astimezone(ZoneInfo('UTC')).strftime('%Y%m%d%H%M')

        pt_map = {'A51': 'aFRR', 'A47': 'mFRR'}
        dir_map = {'A01': 'Up', 'A02': 'Down'}

        for pt, reserve in pt_map.items():
            try:
                r = requests.get(XML_API_URL, params={
                    'securityToken': self.api_key,
                    'documentType': 'A24', 'processType': pt,
                    'area_Domain': domain,
                    'periodStart': period_start, 'periodEnd': period_end,
                }, timeout=45)
                if r.status_code != 200 or len(r.content) < 200:
                    continue
                # Some A-doc responses are ZIP, some plain XML
                if r.content[:2] == b'PK':
                    try:
                        zf = zipfile.ZipFile(io.BytesIO(r.content))
                        xml = zf.read(zf.namelist()[0]).decode('utf-8')
                    except zipfile.BadZipFile:
                        xml = r.text
                else:
                    xml = r.text
            except Exception as e:
                log.warning(f"A24 {country}/{pt} fetch: {e}")
                continue

            if '<TimeSeries>' not in xml:
                continue

            try:
                # Strip namespace for simpler traversal (doc uses balancingdocument:4:1)
                xml_ns = re.sub(r'\sxmlns="[^"]+"', '', xml, count=1)
                root = ET.fromstring(xml_ns)
            except ET.ParseError as e:
                log.warning(f"A24 {country}/{pt} parse: {e}")
                continue

            for ts in root.findall('TimeSeries'):
                direction = ts.findtext('flowDirection.direction')
                dir_label = dir_map.get(direction)
                if not dir_label:
                    continue
                key = f'{reserve}_{dir_label}'
                per = ts.find('Period')
                if per is None:
                    continue
                resolution = per.findtext('resolution') or 'PT15M'
                step_min = int(''.join(c for c in resolution if c.isdigit()) or '15')
                start_iso = per.find('timeInterval').findtext('start')
                period_start_utc = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
                for p in per.findall('Point'):
                    pos_s = p.findtext('position')
                    qty_s = p.findtext('quantity')
                    if pos_s is None or qty_s is None:
                        continue
                    try:
                        pos = int(pos_s); qty = float(qty_s)
                    except ValueError:
                        continue
                    pt_utc = period_start_utc + timedelta(minutes=step_min * (pos - 1))
                    pt_local = pt_utc.astimezone(tz)
                    if pt_local.date() != local_d.date():
                        continue
                    hm = pt_local.strftime('%H:%M')
                    out[key][hm] = qty

        log.info(f"ENTSO-E A24 ({country}) {date_str}: "
                 f"aFRR Up/Dn={len(out['aFRR_Up'])}/{len(out['aFRR_Down'])}, "
                 f"mFRR Up/Dn={len(out['mFRR_Up'])}/{len(out['mFRR_Down'])}")
        return out

    def get_balancing_activation_prices(self, date_str, country='RO'):
        """Fetch prices of activated balancing energy (A84) per 15-min MTU.

        ENTSO-E publishes aFRR (businessType A96) and RR (A98) activation prices
        for Romania. mFRR (A97) is NOT published for RO — use DAMAS for that.

        Returns:
            {
                'aFRR_Up':   {'HH:MM': price_ron_per_mwh, ...},   # direction A01
                'aFRR_Down': {'HH:MM': price_ron_per_mwh, ...},   # direction A02
                'RR_Up':     {'HH:MM': ...},
                'RR_Down':   {'HH:MM': ...},
                'currency':  'RON',
                'date':      'YYYY-MM-DD',
            }

        Prices are timestamped by the local HH:MM of each 15-min MTU start.
        """
        out = {'aFRR_Up': {}, 'aFRR_Down': {}, 'RR_Up': {}, 'RR_Down': {},
               'currency': None, 'date': date_str}
        cfg = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG['RO'])
        domain = cfg['domain']
        tz = cfg['tz']

        local_d = datetime.strptime(date_str, '%Y-%m-%d')
        start_local = local_d.replace(tzinfo=tz)
        end_local = start_local + timedelta(days=1)
        period_start = start_local.astimezone(ZoneInfo('UTC')).strftime('%Y%m%d%H%M')
        period_end = end_local.astimezone(ZoneInfo('UTC')).strftime('%Y%m%d%H%M')

        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1'}
        dir_map = {
            ('A96', 'A01'): 'aFRR_Up',   ('A96', 'A02'): 'aFRR_Down',
            ('A98', 'A01'): 'RR_Up',     ('A98', 'A02'): 'RR_Down',
        }

        for bt in ('A96', 'A98'):
            xml = self._xml_get({
                'documentType': 'A84',
                'businessType': bt,
                'controlArea_Domain': domain,
                'periodStart': period_start,
                'periodEnd': period_end,
            }, timeout=45)
            if not xml or '<TimeSeries>' not in xml:
                continue
            try:
                root = ET.fromstring(xml)
            except ET.ParseError as e:
                log.warning(f"A84 bt={bt} parse error: {e}")
                continue

            for ts in root.findall('ns:TimeSeries', ns):
                direction = ts.findtext('ns:flowDirection.direction', namespaces=ns)
                key = dir_map.get((bt, direction))
                if not key:
                    continue
                currency = ts.findtext('ns:currency_Unit.name', namespaces=ns)
                if currency:
                    out['currency'] = currency
                per = ts.find('ns:Period', ns)
                if per is None:
                    continue
                resolution = per.findtext('ns:resolution', namespaces=ns) or 'PT15M'
                # Step per position — PT15M → 15 min, PT30M → 30 min, etc.
                step_min = int(''.join(c for c in resolution if c.isdigit()) or '15')
                start_iso = per.find('ns:timeInterval', ns).findtext('ns:start', namespaces=ns)
                period_start_utc = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
                for p in per.findall('ns:Point', ns):
                    pos = int(p.findtext('ns:position', namespaces=ns))
                    price_str = p.findtext('ns:activation_Price.amount', namespaces=ns)
                    if price_str is None:
                        continue
                    try:
                        price = float(price_str)
                    except ValueError:
                        continue
                    pt_utc = period_start_utc + timedelta(minutes=step_min * (pos - 1))
                    pt_local = pt_utc.astimezone(tz)
                    # Keep only points that fall on the requested local day
                    if pt_local.date() != local_d.date():
                        continue
                    hm = pt_local.strftime('%H:%M')
                    out[key][hm] = price
        log.info(f"ENTSO-E A84 ({country}) {date_str}: "
                 f"aFRR_Up={len(out['aFRR_Up'])}, aFRR_Down={len(out['aFRR_Down'])}, "
                 f"RR_Up={len(out['RR_Up'])}, RR_Down={len(out['RR_Down'])} pts "
                 f"currency={out['currency']}")
        return out
