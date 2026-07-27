# Energy Prediction — Romania

Flask app that forecasts Romanian electricity prices and imbalance direction using ENTSO-E, Transelectrica SEN, and DAMAS data + LSTM/logistic-regression/Prophet models.

## Run

```bash
PYTHONIOENCODING=utf-8 python app.py
```

Serves on `http://127.0.0.1:8084`. The `PYTHONIOENCODING=utf-8` is required on Windows for Romanian characters in logs.

## Stack

- Python 3.10 / Flask (dev server, not waitress)
- SQLite at `energy_data.db`
- Single-file SPA frontend at `templates/index.html` (Chart.js + Sortable.js via CDN)
- No `static/` assets — all CSS/JS is inline in the template

## Key files

- `app.py` — Flask routes + background polling threads (SEN 20s, ENTSO-E POST 60s, XML actuals 120s)
- `entsoe_client.py` — ENTSO-E POST API + XML REST API (A75 for actual generation per type)
- `transelectrica_client.py` — live SEN scraper from `transelectrica.ro/sen-filter`
- `transelectrica_newmarkets_api.py` — DAMAS API (imbalance prices, marginal prices, activated energy)
- `database.py` — SQLite schema + query helpers
- `lstm_predictor.py` / `negative_price_predictor.py` / `predictor.py` — ML models
- `market_analysis.py` / `sen_analysis.py` — analytical summaries for the UI

## Gotchas

- Romania bidding zone: `BZN|10YRO-TEL------P` (POST API), `10YRO-TEL------P` (XML API)
- XML API is rate-limited (503); code falls back to POST API actuals
- POST API uses Romanian-day UTC ranges; XML uses midnight-to-midnight UTC
- `app.config['TEMPLATES_AUTO_RELOAD']` is NOT set — template edits require a server restart
- Background ML training (LSTM, logreg, Prophet) runs on startup and can saturate CPU for ~30s

## Frontend

- 7 draggable cards (Sortable.js), order persisted in `localStorage`
- Inter + JetBrains Mono via Google Fonts
- Dark/light via `body.dark-mode`, persisted in `localStorage.ep_dark_mode`
- Polling intervals are staggered (0s/3s/6s/9s/12s/20s) to avoid synchronized DOM rebuilds
