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
| `DB_PATH` | no | Where the SQLite archive lives (default: next to the code; `/data/energy_data.db` in the image). Point it at a mounted volume on a server. |

The core imbalance tables use DAMAS only and need no key or account.

## Run locally

```bash
pip install -r requirements.txt
python app.py            # http://localhost:8084
```

## Deploy

```bash
echo "ENTSOE_API_KEY=<your-token>" > .env    # optional
docker compose up -d --build
```

Serves on port 80, restarts on crash and on reboot, and keeps the archive in a
named volume so a rebuild does not wipe it.

The image builds on **x86_64 and arm64** — the Dockerfile picks the right torch
wheel per architecture (PyTorch's CPU index on x86 to avoid the CUDA
dependencies; plain PyPI on ARM, where the wheel is already CPU-only).

Measured footprint with all models loaded: **~830 MB RSS**. A 512 MB tier will
not hold it. To run without ML, drop `torch` from the image — the DAMAS tables
and the archive do not depend on it.

### Oracle Cloud Ampere A1 (free)

The Always Free ARM shape (2 OCPU / 12 GB as of June 2026) fits this
comfortably. Two things bite on OCI specifically:

1. **Two firewalls.** Opening the port in the VCN Security List is not enough —
   the instance also ships local iptables rules that drop everything but SSH:
   ```bash
   sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
   sudo netfilter-persistent save        # Ubuntu
   ```
2. **Capacity.** ARM shapes are often "Out of Capacity" in a given region, and
   the home region is chosen once and cannot be changed later.

Backfill the archive once the container is up:

```bash
docker compose exec dashboard python archive_backfill.py 366
```

For a domain with HTTPS, put Caddy or nginx in front — the app speaks plain
HTTP and does not terminate TLS itself.

## Archive data

`energy_data.db` (SQLite, ~350 MB) is **not** in the repo — it is regenerable:

```bash
python archive_backfill.py 366     # backfills 1 year of DAMAS, idempotent
```

Only factual DAMAS values are stored; intervals DAMAS has not published are
left empty rather than filled in.
