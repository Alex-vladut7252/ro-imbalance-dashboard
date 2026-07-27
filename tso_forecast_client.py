"""TSO-direct forecast client.

Pulls wind / solar / load / generation forecasts straight from each
country's Transmission System Operator (TSO) public data, bypassing
ENTSO-E. Keeps the dashboard's forecast layer authoritative even when
ENTSO-E is delayed or off.

Currently wired:
    HU — MAVIR RTDW (charts 11840 wind, 19240/19260 PV, 7678 load, 4401 gen)

Returns a unified per-country shape:
    {
        'wind':       {'HH:MM': MW, ...},   # day-ahead forecast
        'solar':      {'HH:MM': MW, ...},
        'load':       {'HH:MM': MW, ...},   # day-ahead load forecast
        'generation': {'HH:MM': MW, ...},   # generation plan
        'sources':    {field: 'human-readable source label'},
        'date':       'YYYY-MM-DD',
        'country':    'XX',
    }

All values are in MW; timestamps are HH:MM in the country's local TZ.
"""
import logging

log = logging.getLogger(__name__)


def _empty_shape(country, date_str):
    return {
        'wind': {}, 'solar': {}, 'load': {}, 'generation': {},
        'sources': {}, 'date': date_str, 'country': country.upper(),
    }


def _resample_to_quarter(minute_series):
    """Resample a minute-resolution {timestamp: value} dict to 15-min HH:MM keys.

    MAVIR RTDW publishes minute-level data; the dashboard's forecast layer
    expects 15-min steps. We average the 15 minutes that fall inside each
    ISP. Empty bins are dropped.
    """
    buckets = {}
    for ts, v in minute_series.items():
        if v is None:
            continue
        try:
            hm = ts[11:16]   # "YYYY-MM-DD HH:MM:SS" -> "HH:MM"
            h, m = int(hm[:2]), int(hm[3:5])
            isp_start_m = (m // 15) * 15
            key = f"{h:02d}:{isp_start_m:02d}"
            buckets.setdefault(key, []).append(float(v))
        except (ValueError, IndexError, TypeError):
            continue
    return {k: round(sum(vs) / len(vs), 2) for k, vs in buckets.items() if vs}


def get_hu_forecast(date_str):
    """MAVIR RTDW forecasts for Hungary.

    Wind:  chart 11840 column 'Szélerőművek becsült termelése (dayahead)'
    Solar: chart 19240 (HMKE rooftop) + 19260 (SCTE industrial), DA columns
    Load:  chart 7678 column 'Bruttó rendszerterhelés becslés (dayahead)'
    Gen:   chart 4401 column 'Bruttó terv erőművi termelés'
    """
    import mavir_rtdw_client as rtdw
    out = _empty_shape('HU', date_str)

    chart_field_map = {
        # (chart_id, hungarian_column_name): output_field
        (11840, 'Szélerőművek becsült termelése (dayahead)'): 'wind',
        (7678,  'Bruttó rendszerterhelés becslés (dayahead)'): 'load',
        (4401,  'Bruttó terv erőművi termelés'):              'generation',
    }
    for (cid, col), field in chart_field_map.items():
        try:
            data = rtdw.get_chart(cid, date_str)
            minute = {}
            for row in data['rows']:
                ts = row.get('Időpont')
                v = row.get(col)
                if ts is not None and v is not None:
                    minute[str(ts)] = v
            out[field] = _resample_to_quarter(minute)
        except Exception as e:
            log.warning(f"HU forecast {cid}/{col!r}: {e}")

    # Solar = HMKE + SCTE day-ahead (sum of rooftop + industrial PV)
    try:
        hmke = rtdw.get_chart(19240, date_str)
        scte = rtdw.get_chart(19260, date_str)
        col_hmke = 'HMKE PV - Előre jelzett termelés - Day-Ahead (DA)'
        col_scte = 'SCTE ipari PV - Előre jelzett termelés - Day-Ahead (DA)'
        merged = {}
        for row in hmke['rows']:
            ts = row.get('Időpont')
            v = row.get(col_hmke)
            if ts is not None and v is not None:
                merged[str(ts)] = float(v)
        for row in scte['rows']:
            ts = row.get('Időpont')
            v = row.get(col_scte)
            if ts is not None and v is not None:
                merged[str(ts)] = merged.get(str(ts), 0.0) + float(v)
        out['solar'] = _resample_to_quarter(merged)
    except Exception as e:
        log.warning(f"HU solar forecast (19240+19260): {e}")

    out['sources'] = {
        'wind':       'MAVIR RTDW chart 11840 (day-ahead)',
        'solar':      'MAVIR RTDW charts 19240 + 19260 (HMKE + SCTE day-ahead)',
        'load':       'MAVIR RTDW chart 7678 (day-ahead)',
        'generation': 'MAVIR RTDW chart 4401 (gross plan)',
    }
    return out


# Country dispatcher. Each entry is the function that produces the unified
# shape. Add new TSOs here as they're wired.
_PROVIDERS = {
    'HU': get_hu_forecast,
}


def get_forecast(country, date_str):
    """Top-level entry point — dispatches to the country's TSO provider."""
    fn = _PROVIDERS.get((country or '').upper())
    if fn is None:
        out = _empty_shape(country, date_str)
        out['sources'] = {'_unsupported': f"No TSO-direct forecast wired for {country}"}
        return out
    return fn(date_str)


if __name__ == '__main__':
    import sys, json, datetime
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    country = sys.argv[1] if len(sys.argv) > 1 else 'HU'
    date    = sys.argv[2] if len(sys.argv) > 2 else datetime.date.today().isoformat()
    out = get_forecast(country, date)
    summary = {
        'country': out['country'], 'date': out['date'],
        'wind':       len(out['wind']),
        'solar':      len(out['solar']),
        'load':       len(out['load']),
        'generation': len(out['generation']),
        'sources':    out['sources'],
    }
    print(json.dumps(summary, indent=2))
