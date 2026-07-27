"""
Transelectrica New Markets (DAMAS) Public Report API Client
============================================================
Base URL: https://newmarkets.transelectrica.ro/usy-durom-publicreportg01/00121002500000000000000000000100

All endpoints are GET requests returning JSON with an "itemList" array.
Data is available from 2024-05-31 onwards, in 15-minute ISP (Imbalance Settlement Period) intervals.
96 ISPs per day. Dates use ISO 8601 format (UTC).

The delivery day in Romania runs from 22:00 UTC (D-1) to 22:00 UTC (D), which is midnight to midnight EET/EEST.

DISCOVERY:
- Platform: "Damas" built on Unicorn Universe (uu5) framework
- CMS: uu-webkit-maing02 (React SPA)
- Data backend: usy-durom-publicreportg01
- JS library: https://newmarkets.transelectrica.ro/cdn/usy-durom-publicreportg01/1.0.1/usy_durom_publicreportg01-uu5lib.min.js
"""

import requests
from datetime import datetime, timedelta
from typing import Optional


BASE_URL = "https://newmarkets.transelectrica.ro/usy-durom-publicreportg01/00121002500000000000000000000100"


def _fetch(endpoint: str, params: dict) -> dict:
    """Fetch data from the Transelectrica DAMAS API."""
    url = f"{BASE_URL}/{endpoint}"
    headers = {
        "Accept": "application/json",
        # DAMAS sits behind Cloudflare; a bare requests User-Agent is occasionally
        # served a challenge page (HTTP 200 + HTML) instead of JSON. A browser-like
        # UA avoids it — same trick as the energy_crm DamasImbalanceClient.
        "User-Agent": "Mozilla/5.0 (compatible; EnergyPrediction/1.0; imbalance-fetch)",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _time_interval_params(date_from: str, date_to: str) -> dict:
    """Build timeInterval query params. Dates should be ISO 8601 UTC strings."""
    return {
        "timeInterval.from": date_from,
        "timeInterval.to": date_to,
    }


def _day_params(date_str: str) -> dict:
    """Build params for a full delivery day (00:00 to 24:00 UTC on that date).
    For Romanian delivery day, use date_from = (D-1)T22:00Z, date_to = DT22:00Z.
    """
    return _time_interval_params(
        f"{date_str}T00:00:00.000Z",
        f"{date_str}T23:59:59.000Z",
    )


# =============================================================================
# PUBLIC ENDPOINTS (no authentication required)
# =============================================================================

def get_estimated_imbalance_prices(date_from: str, date_to: str) -> list:
    """
    Estimated imbalance prices per 15-min ISP.

    Fields per item:
    - timeInterval.from/to: ISP boundaries (UTC)
    - ISP: Imbalance Settlement Period number (1-96)
    - hour: Hour of delivery day (1-24, Romanian convention)
    - estimatedPriceNegativeImbalance: EUR/MWh for negative (short) imbalance
    - estimatedPricePositiveImbalance: EUR/MWh for positive (long) imbalance
    - estimatedSystemImbalance: MW system imbalance
    - realizedConsumption: MW actual consumption
    - sumQup / sumQdn: Activated balancing energy up/down (MW)
    - sumQupPup / sumQdownPdn: Activated balancing energy cost up/down
    - imbalanceNettingImport / imbalanceNettingExport: MW
    - estimatedUnintendedDeviationInArea / OutArea: MW
    - type: "Single" (single pricing) or "Dual" (dual pricing)
    - fcr: Frequency Containment Reserve (MW)
    """
    data = _fetch("publicReport/estimatedImbalancePrices",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_marginal_prices_overview(date_from: str, date_to: str) -> list:
    """
    Marginal prices for balancing energy per 15-min ISP.

    Fields per item:
    - aFRR_Up / aFRR_Down: automatic FRR marginal prices (EUR/MWh)
    - mFRR_Up / mFRR_Down: manual FRR marginal prices
    - mFRR_Up_Scheduled / mFRR_Down_Scheduled: scheduled mFRR
    - mFRR_Up_Direct / mFRR_Down_Direct: direct mFRR
    - rr_Up / rr_Down: Replacement Reserve marginal prices
    - fcr: Frequency Containment Reserve
    """
    data = _fetch("publicReport/marginalPricesOverview",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_estimated_power_system_imbalance(date_from: str, date_to: str) -> list:
    """
    Estimated power system imbalance per 15-min ISP.

    Fields per item:
    - estimatedSystemImbalance: MW (positive = surplus, negative = deficit)
    - type: "Surplus" or "Shortage"
    - estimatedUnintendedDeviationINArea / OUTArea: MW
    - contractedBMVolumeUp / Down: MW balancing market volume
    - imbalanceNettingImport / Export: MW
    - balancingExchange: MW
    - frequencyBiasFactorImport / Export: MW
    - activatedReserve: MW total activated reserve
    """
    data = _fetch("publicReport/estimatedPowerSystemImbalance",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_daily_consumption_overview(date_from: str, date_to: str) -> list:
    """
    Daily consumption forecast vs realized, per 15-min ISP.

    Fields per item:
    - hourlyInterval: Hour number
    - grossForecastConsumption: MW forecast
    - grossRealizedConsumption: MW actual
    - lastUpdate: timestamp of last data update
    """
    data = _fetch("publicReport/dailyConsumptionOverview",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_activated_balancing_energy_overview(date_from: str, date_to: str) -> list:
    """
    Activated balancing energy per 15-min ISP.

    Fields per item:
    - aFRR_Up / aFRR_Down: automatic FRR activated energy (MW)
    - mFRR_Up / mFRR_Down: manual FRR activated energy (MW)
    - rr_Up / rr_Down: Replacement Reserve (MW)
    - fcr: FCR (MW)
    """
    data = _fetch("publicReport/activatedBalancingEnergyOverview",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_scheduled_exchanges(date_from: str, date_to: str) -> list:
    """
    Scheduled cross-border exchanges per 15-min ISP.

    Fields per item - each border direction has sub-fields:
    - huro: Hungary -> Romania
    - rohu: Romania -> Hungary
    - mdro: Moldova -> Romania
    - romd: Romania -> Moldova
    - rors: Romania -> Serbia
    - rsro: Serbia -> Romania
    - bgro: Bulgaria -> Romania
    - robg: Romania -> Bulgaria
    - roua: Romania -> Ukraine
    - uaro: Ukraine -> Romania

    Each direction has: yearly, monthly, dayAhead, intraday, emergencyHelp,
    longTerm, commercial, operational (all in MW)
    """
    data = _fetch("publicReport/scheduledExchanges",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_final_unintentional_deviations(date_from: str, date_to: str) -> list:
    """
    Final unintentional deviations per 15-min ISP.
    Note: Data may not always be populated (fields may be null).
    """
    data = _fetch("publicReport/finalUnintentionalDeviations",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_generation_schedules(date_from: str, date_to: str) -> list:
    """
    NonDU Energy Delivery & Total Generation Notified, per 15-min ISP.

    Fields per item:
    - brpsConsumption: BRP total scheduled consumption (MW)
    - brpsProduction: BRP total scheduled production (MW)
    - nonDuConsumption: Non-DU consumption (MW)
    - nonDuProduction: Non-DU production (MW)
    """
    data = _fetch("publicReport/generationSchedules",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_capacity_limits(date_from: str, date_to: str) -> list:
    """
    Cross-border capacity limits (hourly, not 15-min).

    Fields per item:
    - borderDirectionCode: e.g. "romd", "huro", "robg", etc.
    - borderDirection: e.g. "ROMANIA-MOLDOVA"
    - ntclongterm: Long-term NTC (MW)
    - ntcActual: Actual NTC (MW)
    - ntcDayAhead: Day-ahead NTC (MW)
    - ocDayAhead: Day-ahead offered capacity (MW)
    - ocIntraday: Intraday offered capacity (MW)
    - aacIntraday: Already Allocated Capacity intraday (MW)
    - ntcIntraday: Intraday NTC (MW)
    """
    data = _fetch("publicReport/capacityLimits",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_auction_statistics(date_from: str, date_to: str) -> list:
    """
    Transfer capacity auction statistics list.

    Fields per item:
    - code: Auction code (e.g. "ROMD-D-01042026-020311")
    - borderDirectionCode / borderDirection
    - capacityContractTypeCode: "dayAhead", "monthly", "yearly"
    - capacityContractType: "Daily", "Monthly", "Yearly"
    - auctionState: e.g. "finalResultsPublished"
    - auctionProduct: e.g. "HOURLY"
    """
    data = _fetch("publicReport/auctionStatistics",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_auction_statistics_detail(code: str, auction_id: str) -> dict:
    """
    Detail for a specific auction (from get_auction_statistics).

    Returns hourly statistics with:
    - offeredCapacity, totalRequestedCapacity, totalAllocatedCapacity (MW)
    - auctionPrice (EUR/MW)
    - numberOfCapacityTraders, numberOfCapacityHolders, numberOfAuctionBids
    """
    data = _fetch("publicReport/auctionStatisticsDetail",
                   {"code": code, "id": auction_id})
    return data


def get_tender_statistics(date_from: str, date_to: str) -> list:
    """
    Ancillary services tender statistics.

    Returns tenders with service codes: aFRRDown, aFRRUp, mFRRDown, mFRRUp, FCR
    Each with hourly timeIntervalList containing:
    - tenderDemand, tenderSatisfiedDemand (MW)
    - tenderPrice, averageOfferedPrice, averageAcceptedPrice (EUR/MW)
    """
    data = _fetch("publicReport/tenderStatistics",
                   _time_interval_params(date_from, date_to))
    return data.get("itemList", [])


def get_brp_list_report(date_from: str, date_to: str, brp_codes: list) -> list:
    """
    BRP (Balance Responsible Party) list report.
    Requires authentication for actual BRP codes.

    Args:
        date_from/date_to: YYYY-MM-DD format
        brp_codes: list of BRP code strings
    """
    params = {
        "dateInterval.from": date_from,
        "dateInterval.to": date_to,
    }
    # Repeated params for array
    for code in brp_codes:
        params.setdefault("brpCodeList", []).append(code)
    data = _fetch("publicReport/brpListReport", params)
    return data.get("itemList", [])


# =============================================================================
# UTILITY: Filter initialization (used by the frontend)
# =============================================================================

def init_list_filter(use_case: str) -> dict:
    """
    Get filter configuration for a specific report.
    use_case: one of the report names (e.g. "estimatedImbalancePrices")
    """
    return _fetch("common/initListFilter", {"useCase": use_case})


# =============================================================================
# ALL AVAILABLE PAGES ON THE DAMAS PLATFORM
# =============================================================================
"""
Complete site map (from loadWebsite API):
Base: https://newmarkets.transelectrica.ro/uu-webkit-maing02/00121011300000000000000000000100/

PUBLIC (no auth required):
1.  /home                                    - Home page
2.  /publicReports                           - Public Reports (category header)
3.  /auctionStatisticList                    - Transfer Capacity Auction Statistics
4.  /capacityLimits                          - Capacity Limits
5.  /marginalPricesOverview                  - Marginal prices overview
6.  /finalUnintentionalDeviations            - Final unintentional deviations
7.  /estimatedImbalancePrices                - Estimated imbalance prices
8.  /estimatedPowerSystemImbalance           - Estimated power system imbalance
9.  /dailyConsumptionOverview                - Daily consumption overview
10. /activatedBalancingEnergyOverview        - Activated balancing energy overview
11. /scheduledExchanges                      - Scheduled exchanges
12. /scheduledExchangesAggregated            - Scheduled exchanges aggregated
13. /nonDUEnergyDeliveryTotalGenerationNotified - NonDU Energy Delivery & Total Generation
14. /tenderStatisticList                     - Ancillary Services Tender Statistics

HIDDEN (detail pages, accessible via parent):
15. /tenderStatistics                        - Tender Statistics Detail
16. /auctionStatistics                       - Auction Statistics Detail

AUTHENTICATED (require login):
17. /finalSystemImbalance                    - Final System Imbalance
18. /finalImbalancePrices                    - Final Imbalance Prices
19. /balanceChecking                         - Balance Checking
20. /mrnNetBalancingCosts                    - MRN Net Balancing Costs
21. /mrnCostsAndRevenues                     - MRN Costs and Revenues
22. /brpList                                 - BRP List
"""


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    # Test: Get yesterday's estimated imbalance prices
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    date_from = f"{yesterday}T00:00:00.000Z"
    date_to = f"{yesterday}T23:59:59.000Z"

    print(f"Fetching estimated imbalance prices for {yesterday}...")
    items = get_estimated_imbalance_prices(date_from, date_to)
    data_items = [i for i in items if i.get("id")]
    print(f"  Total ISPs: {len(items)}, With data: {len(data_items)}")

    if data_items:
        first = data_items[0]
        print(f"  First ISP: {first['timeInterval']['from']} - {first['timeInterval']['to']}")
        print(f"    Price (neg): {first.get('estimatedPriceNegativeImbalance')} EUR/MWh")
        print(f"    Price (pos): {first.get('estimatedPricePositiveImbalance')} EUR/MWh")
        print(f"    System imbalance: {first.get('estimatedSystemImbalance')} MW")
        print(f"    Consumption: {first.get('realizedConsumption')} MW")

    print(f"\nFetching marginal prices for {yesterday}...")
    items = get_marginal_prices_overview(date_from, date_to)
    data_items = [i for i in items if i.get("id")]
    print(f"  Total ISPs: {len(items)}, With data: {len(data_items)}")

    if data_items:
        first = data_items[0]
        print(f"  First ISP: {first['timeInterval']['from']}")
        print(f"    aFRR Up: {first.get('aFRR_Up')} EUR/MWh")
        print(f"    aFRR Down: {first.get('aFRR_Down')} EUR/MWh")

    print(f"\nFetching daily consumption for {yesterday}...")
    items = get_daily_consumption_overview(date_from, date_to)
    data_items = [i for i in items if i.get("id")]
    print(f"  Total ISPs: {len(items)}, With data: {len(data_items)}")

    if data_items:
        first = data_items[0]
        print(f"  First ISP: forecast={first.get('grossForecastConsumption')} MW, "
              f"realized={first.get('grossRealizedConsumption')} MW")

    print(f"\nFetching scheduled exchanges for {yesterday}...")
    items = get_scheduled_exchanges(date_from, date_to)
    print(f"  Total ISPs: {len(items)}")
    if items:
        first = items[0]
        borders = ['huro', 'rohu', 'robg', 'bgro', 'rors', 'rsro', 'romd', 'mdro', 'roua', 'uaro']
        for b in borders:
            val = first.get(b, {})
            if val and val.get('commercial'):
                print(f"    {b.upper()}: commercial={val['commercial']} MW")
