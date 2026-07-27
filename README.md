---
title: RO Imbalance Dashboard
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Romanian Balancing Market — Estimated Imbalance Prices

Live and historical Romanian imbalance data, sourced only from the public
Transelectrica **DAMAS** reports and **ENTSO-E** — no invented numbers, no
interpolation. Pick any day in the date picker to load that day's full tables
(imbalance prices, aFRR/mFRR merit order, generation chart, DAM prices).

## Data sources (per column)

| Column(s) | DAMAS report |
|---|---|
| Imbal price, Sys imbalance, netting, cost | `estimatedImbalancePrices` |
| aFRR / mFRR activated volumes | `activatedBalancingEnergyOverview` |
| P aFRR / mFRR (marginal prices) | `marginalPricesOverview` |

Volumes are shown in MW (DAMAS publishes MWh per 15-min ISP → ×4). Reserve
prices are shown only for the direction actually activated.

## Config

| Env var | Required | Purpose |
|---|---|---|
| `ENTSOE_API_KEY` | no | ENTSO-E generation chart + ML prediction card. Get your own free token from the ENTSO-E Transparency Platform — **none is shipped in this repo**. |
| `PORT` | no | HTTP port (default `8084`; the Dockerfile uses `7860`). |

The core imbalance tables use DAMAS only and need no key or account.

## Run locally

```bash
pip install -r requirements.txt
python app.py            # http://localhost:8084
```

## Deploy

```bash
docker build -t ro-imbalance .
docker run -p 7860:7860 -e ENTSOE_API_KEY=<your-token> ro-imbalance
```

Any Docker host works (Hugging Face Spaces, Fly.io, Railway, a VPS behind
nginx). Needs ~1 GB RAM — the ML stack (PyTorch + XGBoost + Prophet) does not
fit in a 512 MB tier. To run without ML, drop `torch` from the image; the DAMAS
tables and the archive are independent of it.

## Archive data

`energy_data.db` (SQLite, ~350 MB) is **not** in the repo — it is regenerable:

```bash
python archive_backfill.py 366     # backfills 1 year of DAMAS, idempotent
```

Only factual DAMAS values are stored; intervals DAMAS has not published are
left empty rather than filled in.
