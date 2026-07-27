"""Country registry for the multi-country dashboard.

Single source of truth for bidding zones, timezones, and which data sources
supply each panel per country. Used by backend routes and (via the country
query param) by the frontend to decide what to render.

HU note: ENTSO-E publishes day-ahead prices and generation data for Hungary,
but NOT balancing/imbalance prices. Those require MAVIR (scraping) — which is
a follow-up phase. For now the HU dashboard is limited to DAM + generation.
"""

COUNTRIES = {
    'RO': {
        'label': 'Romania',
        'bidding_zone': '10YRO-TEL------P',
        'tz': 'Europe/Bucharest',
        'currency': 'EUR',
        'entsoe_area': 'BZN|10YRO-TEL------P',
        # Which panels have a data source available right now
        'panels': {
            'dam': True,
            'entsoe_generation': True,
            'sen_live': True,
            'balancing': True,       # DAMAS
            'prediction': True,      # ML trained on RO history
        },
    },
    'HU': {
        'label': 'Hungary',
        'bidding_zone': '10YHU-MAVIR----U',
        'tz': 'Europe/Budapest',
        'currency': 'EUR',
        'entsoe_area': 'BZN|10YHU-MAVIR----U',
        'panels': {
            'dam': True,
            'entsoe_generation': True,
            'sen_live': False,       # Transelectrica is RO-only; MAVIR live TBD
            'balancing': True,       # MAVIR XLSX (Phase 2) — ~2-week lag, historical only
            'prediction': False,     # No HU training data yet
        },
    },
}

DEFAULT_COUNTRY = 'RO'


def get(country_code):
    """Lookup country config. Falls back to DEFAULT_COUNTRY on unknown code."""
    return COUNTRIES.get((country_code or '').upper(), COUNTRIES[DEFAULT_COUNTRY])


def is_valid(country_code):
    return (country_code or '').upper() in COUNTRIES
