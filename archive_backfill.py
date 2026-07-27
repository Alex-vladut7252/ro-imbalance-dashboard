# -*- coding: utf-8 -*-
"""
Backfill the DAMAS imbalance archive (table: damas_imbalance_history) for the past year.

FACTUAL ONLY: pulls the three public Transelectrica DAMAS reports and stores their RAW
values (MWh / Lei, per 15-min ISP). No ML, no interpolation, no estimation on top — the
display layer applies the ×4 (MWh->MW) and the activation gate, so the stored numbers stay
exactly what DAMAS published.

Sources (one call each per day):
  estimatedImbalancePrices        -> prices, system imbalance, netting, sumQ, cost, consumption
  marginalPricesOverview          -> aFRR/mFRR marginal prices
  activatedBalancingEnergyOverview-> aFRR/mFRR activated volumes

Idempotent (INSERT OR REPLACE on PK date,isp). Skips days already complete (>=96 ISPs),
re-fetches missing / partial days. Responsible: 0.3s between days, retry on error.

Usage:
  python archive_backfill.py 366           # backfill last 366 days
  python archive_backfill.py 2025-09-01     # backfill one specific day (test)
"""
import sys, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import transelectrica_newmarkets_api as damas
import database as db

RO = ZoneInfo("Europe/Bucharest")
UTC = ZoneInfo("UTC")


def day_range(date_str):
    ls = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=RO)
    le = ls + timedelta(days=1)
    return (ls.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            le.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"))


def isp_from_interval(ti):
    try:
        start = datetime.fromisoformat(ti["from"].replace("Z", "+00:00"))
        loc = start.astimezone(RO)
        return loc.hour * 4 + loc.minute // 15 + 1
    except Exception:
        return None


def fetch_day(date_str):
    dfrom, dto = day_range(date_str)
    try:
        imb = damas.get_estimated_imbalance_prices(dfrom, dto)
    except Exception as e:
        print(f"  {date_str}: imbalance fetch failed: {e}", flush=True)
        return []
    try:
        marg = damas.get_marginal_prices_overview(dfrom, dto)
    except Exception:
        marg = []
    try:
        act = damas.get_activated_balancing_energy_overview(dfrom, dto)
    except Exception:
        act = []

    def by_isp(data):
        m = {}
        for r in data:
            if r.get("id"):
                isp = r.get("ISP") or isp_from_interval(r.get("timeInterval", {}))
                if isp:
                    m[isp] = r
        return m

    mbi, abi = by_isp(marg), by_isp(act)
    rows = []
    for r in imb:
        if not r.get("id"):
            continue
        isp = r.get("ISP", 0)
        if not isp:
            continue
        neg = r.get("estimatedPriceNegativeImbalance")
        if not isinstance(neg, (int, float)):
            continue  # "N/A" -> not settled yet, skip (never store a fake 0)
        mg, ac = mbi.get(isp, {}), abi.get(isp, {})
        ti = r.get("timeInterval", {})
        rows.append({
            "date": date_str, "isp": isp,
            "interval_from": ti.get("from", ""), "interval_to": ti.get("to", ""),
            "neg_price": neg, "pos_price": r.get("estimatedPricePositiveImbalance"),
            "system_imbalance": r.get("estimatedSystemImbalance"),
            "realized_consumption": r.get("realizedConsumption"),
            "sum_qup": r.get("sumQup"), "sum_qdn": r.get("sumQdn"),
            "sum_qup_pup": r.get("sumQupPup"), "sum_qdown_pdn": r.get("sumQdownPdn"),
            "netting_import": r.get("imbalanceNettingImport"),
            "netting_export": r.get("imbalanceNettingExport"),
            "deviation_in": r.get("estimatedUnintendedDeviationInArea"),
            "deviation_out": r.get("estimatedUnintendedDeviationOutArea"),
            "fcr": r.get("fcr"), "pricing_type": r.get("type"),
            "afrr_up_price": mg.get("aFRR_Up"), "afrr_down_price": mg.get("aFRR_Down"),
            "mfrr_up_price": mg.get("mFRR_Up") or mg.get("mFRR_Up_Scheduled"),
            "mfrr_down_price": mg.get("mFRR_Down") or mg.get("mFRR_Down_Scheduled"),
            "afrr_up_activated": ac.get("aFRR_Up"), "afrr_down_activated": ac.get("aFRR_Down"),
            "mfrr_up_activated": ac.get("mFRR_Up"), "mfrr_down_activated": ac.get("mFRR_Down"),
        })
    return rows


def existing_counts():
    conn = db.get_db()
    rows = conn.execute(
        "SELECT date, COUNT(*) FROM damas_imbalance_history GROUP BY date").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def backfill(days_back):
    counts = existing_counts()
    today = datetime.now(RO).strftime("%Y-%m-%d")
    new_rows = fetched = skipped = 0
    for d in range(days_back, -1, -1):
        date_str = (datetime.now(RO) - timedelta(days=d)).strftime("%Y-%m-%d")
        have = counts.get(date_str, 0)
        # >=96 = complete normal day (fall-DST days have 100). Partial/missing -> refetch.
        if date_str != today and have >= 96:
            skipped += 1
            continue
        try:
            rows = fetch_day(date_str)
            if rows:
                db.save_damas_history(rows)
                new_rows += len(rows); fetched += 1
                print(f"  {date_str}: {len(rows)} ISPs (had {have})", flush=True)
            else:
                print(f"  {date_str}: no data returned", flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"  {date_str} ERROR: {e}", flush=True)
            time.sleep(1.0)
    print(f"DONE: fetched {fetched} days ({new_rows} rows), skipped {skipped} complete days",
          flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "366"
    if "-" in arg:  # a single YYYY-MM-DD for testing
        rows = fetch_day(arg)
        if rows:
            db.save_damas_history(rows)
        print(f"{arg}: stored {len(rows)} ISPs", flush=True)
    else:
        backfill(int(arg))
