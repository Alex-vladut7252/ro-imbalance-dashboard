"""
SQLite database for caching ENTSO-E and Transelectrica data.
"""

import sqlite3
import json
import os
import logging
from datetime import datetime

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'energy_data.db')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dam_prices (
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            price_eur REAL,
            volume REAL,
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (date, hour)
        );

        CREATE TABLE IF NOT EXISTS actual_load (
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            load_mw REAL,
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (date, hour)
        );

        CREATE TABLE IF NOT EXISTS load_forecast (
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            forecast_mw REAL,
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (date, hour)
        );

        CREATE TABLE IF NOT EXISTS generation (
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            psr_type TEXT NOT NULL,
            psr_name TEXT,
            value_mw REAL,
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (date, hour, psr_type)
        );

        CREATE TABLE IF NOT EXISTS imbalance_prices (
            date TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT,
            export_energy REAL,
            export_price REAL,
            import_energy REAL,
            import_price REAL,
            source TEXT DEFAULT 'transelectrica',
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (date, period_start)
        );

        CREATE TABLE IF NOT EXISTS sen_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL UNIQUE,
            production REAL,
            consumption REAL,
            sold REAL,
            generation_mix TEXT,
            cross_border TEXT,
            raw_json TEXT,
            fetched_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS cross_border_flows (
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            country TEXT NOT NULL,
            export_mw REAL,
            import_mw REAL,
            net_mw REAL,
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (date, hour, country)
        );

        CREATE INDEX IF NOT EXISTS idx_dam_date ON dam_prices(date);
        CREATE INDEX IF NOT EXISTS idx_load_date ON actual_load(date);
        CREATE INDEX IF NOT EXISTS idx_gen_date ON generation(date);
        CREATE INDEX IF NOT EXISTS idx_sen_ts ON sen_snapshots(timestamp);

        -- DAMAS historical data for ML training
        CREATE TABLE IF NOT EXISTS damas_imbalance_history (
            date TEXT NOT NULL,
            isp INTEGER NOT NULL,
            interval_from TEXT NOT NULL,
            interval_to TEXT NOT NULL,
            neg_price REAL,
            pos_price REAL,
            system_imbalance REAL,
            realized_consumption REAL,
            sum_qup REAL,
            sum_qdn REAL,
            sum_qup_pup REAL,
            sum_qdown_pdn REAL,
            netting_import REAL,
            netting_export REAL,
            deviation_in REAL,
            deviation_out REAL,
            fcr REAL,
            pricing_type TEXT,
            afrr_up_price REAL,
            afrr_down_price REAL,
            mfrr_up_price REAL,
            mfrr_down_price REAL,
            afrr_up_activated REAL,
            afrr_down_activated REAL,
            mfrr_up_activated REAL,
            mfrr_down_activated REAL,
            wind_forecast REAL,
            solar_forecast REAL,
            load_forecast REAL,
            dam_price REAL,
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (date, isp)
        );
        CREATE INDEX IF NOT EXISTS idx_damas_date ON damas_imbalance_history(date);

        -- SEN columns for combined DAMAS+SEN model
        -- (ALTER TABLE is idempotent with IF NOT EXISTS not supported in SQLite,
        --  so we do it in Python below)

        -- ML prediction log
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_at TEXT NOT NULL,
            target_date TEXT NOT NULL,
            target_isp INTEGER NOT NULL,
            target_interval TEXT,
            probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            predicted_negative INTEGER NOT NULL,
            actual_neg_price REAL,
            actual_was_negative INTEGER,
            model_version TEXT,
            features_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pred_target ON prediction_log(target_date, target_isp);

        -- Model metadata
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at TEXT NOT NULL,
            training_days INTEGER,
            training_samples INTEGER,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            f1 REAL,
            auc_roc REAL,
            threshold REAL,
            model_path TEXT,
            features_used TEXT
        );
    """)
    conn.commit()

    # Add SEN columns to damas_imbalance_history (safe to run multiple times)
    sen_cols = ['sen_solar', 'sen_wind', 'sen_consumption', 'sen_sold',
                'sen_surplus', 'sen_renewable_pct']
    for col in sen_cols:
        try:
            conn.execute(f"ALTER TABLE damas_imbalance_history ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()
    log.info("Database initialized")


def save_dam_prices(date_str, prices):
    conn = get_db()
    for p in prices:
        conn.execute("""
            INSERT OR REPLACE INTO dam_prices (date, hour, price_eur, volume)
            VALUES (?, ?, ?, ?)
        """, (date_str, p.get('hour', p.get('position', 0) - 1),
              p.get('price'), p.get('value')))
    conn.commit()
    conn.close()


def save_actual_load(date_str, load_data):
    conn = get_db()
    for d in load_data:
        if 'value' in d:
            conn.execute("""
                INSERT OR REPLACE INTO actual_load (date, hour, load_mw)
                VALUES (?, ?, ?)
            """, (date_str, d.get('hour', d.get('position', 1) - 1), d['value']))
    conn.commit()
    conn.close()


def save_load_forecast(date_str, forecast_data):
    conn = get_db()
    for d in forecast_data:
        if 'value' in d:
            conn.execute("""
                INSERT OR REPLACE INTO load_forecast (date, hour, forecast_mw)
                VALUES (?, ?, ?)
            """, (date_str, d.get('hour', d.get('position', 1) - 1), d['value']))
    conn.commit()
    conn.close()


def save_generation(date_str, gen_data):
    conn = get_db()
    for d in gen_data:
        if 'value' in d and d.get('psr_type'):
            conn.execute("""
                INSERT OR REPLACE INTO generation (date, hour, psr_type, psr_name, value_mw)
                VALUES (?, ?, ?, ?, ?)
            """, (date_str, d.get('hour', d.get('position', 1) - 1),
                  d['psr_type'], d.get('psr_name', ''), d['value']))
    conn.commit()
    conn.close()


def save_imbalance_prices(date_str, records):
    conn = get_db()
    for r in records:
        conn.execute("""
            INSERT OR REPLACE INTO imbalance_prices
            (date, period_start, period_end, export_energy, export_price, import_energy, import_price)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (date_str, r['period_start'], r.get('period_end'),
              r.get('export_energy'), r.get('export_price'),
              r.get('import_energy'), r.get('import_price')))
    conn.commit()
    conn.close()


def save_sen_snapshot(data):
    conn = get_db()
    conn.execute("""
        INSERT OR IGNORE INTO sen_snapshots (timestamp, production, consumption, sold, generation_mix, cross_border, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (data['timestamp'], data.get('production'), data.get('consumption'),
          data.get('sold'), json.dumps(data.get('generation_mix')),
          json.dumps(data.get('cross_border')), json.dumps(data)))
    conn.commit()
    conn.close()


def get_sen_today(date_str=None):
    """Get all SEN snapshots for a given date (default: today).
    Returns list of dicts with timestamp, production, consumption, wind, solar etc."""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, raw_json FROM sen_snapshots WHERE timestamp LIKE ? ORDER BY timestamp",
        (date_str + '%',)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        try:
            data = json.loads(r['raw_json'])
            result.append(data)
        except (json.JSONDecodeError, TypeError):
            pass
    return result


def get_dam_prices(date_str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM dam_prices WHERE date=? ORDER BY hour", (date_str,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_actual_load(date_str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM actual_load WHERE date=? ORDER BY hour", (date_str,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_load_forecast(date_str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM load_forecast WHERE date=? ORDER BY hour", (date_str,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_generation(date_str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM generation WHERE date=? ORDER BY hour, psr_type", (date_str,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_imbalance_prices(date_str):
    conn = get_db()
    rows = conn.execute("SELECT * FROM imbalance_prices WHERE date=? ORDER BY period_start", (date_str,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dam_prices_range(start_date, end_date):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM dam_prices WHERE date BETWEEN ? AND ? ORDER BY date, hour",
        (start_date, end_date)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_actual_load_range(start_date, end_date):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM actual_load WHERE date BETWEEN ? AND ? ORDER BY date, hour",
        (start_date, end_date)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_generation_range(start_date, end_date):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM generation WHERE date BETWEEN ? AND ? ORDER BY date, hour, psr_type",
        (start_date, end_date)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sen_snapshots(limit=100):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM sen_snapshots ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_cached_dates():
    """Return list of dates we have data for."""
    conn = get_db()
    dates = {}
    for table in ['dam_prices', 'actual_load', 'generation']:
        rows = conn.execute(f"SELECT DISTINCT date FROM {table} ORDER BY date DESC LIMIT 30").fetchall()
        dates[table] = [r['date'] for r in rows]
    conn.close()
    return dates


# ── DAMAS history for ML ──────────────────────────────────────

def save_damas_history(rows):
    """Bulk save DAMAS ISP rows. Each row is a dict with all fields."""
    conn = get_db()
    for r in rows:
        conn.execute("""
            INSERT OR REPLACE INTO damas_imbalance_history
            (date, isp, interval_from, interval_to, neg_price, pos_price,
             system_imbalance, realized_consumption, sum_qup, sum_qdn,
             sum_qup_pup, sum_qdown_pdn, netting_import, netting_export,
             deviation_in, deviation_out, fcr, pricing_type,
             afrr_up_price, afrr_down_price, mfrr_up_price, mfrr_down_price,
             afrr_up_activated, afrr_down_activated, mfrr_up_activated, mfrr_down_activated,
             wind_forecast, solar_forecast, load_forecast, dam_price)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (r.get('date'), r.get('isp'), r.get('interval_from'), r.get('interval_to'),
              r.get('neg_price'), r.get('pos_price'),
              r.get('system_imbalance'), r.get('realized_consumption'),
              r.get('sum_qup'), r.get('sum_qdn'),
              r.get('sum_qup_pup'), r.get('sum_qdown_pdn'),
              r.get('netting_import'), r.get('netting_export'),
              r.get('deviation_in'), r.get('deviation_out'),
              r.get('fcr'), r.get('pricing_type'),
              r.get('afrr_up_price'), r.get('afrr_down_price'),
              r.get('mfrr_up_price'), r.get('mfrr_down_price'),
              r.get('afrr_up_activated'), r.get('afrr_down_activated'),
              r.get('mfrr_up_activated'), r.get('mfrr_down_activated'),
              r.get('wind_forecast'), r.get('solar_forecast'),
              r.get('load_forecast'), r.get('dam_price')))
    conn.commit()
    conn.close()


def get_damas_history(start_date=None, end_date=None):
    """Get DAMAS history rows as list of dicts."""
    conn = get_db()
    if start_date and end_date:
        rows = conn.execute(
            "SELECT * FROM damas_imbalance_history WHERE date BETWEEN ? AND ? ORDER BY date, isp",
            (start_date, end_date)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM damas_imbalance_history ORDER BY date, isp").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_damas_dates():
    """Return set of dates already stored."""
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT date FROM damas_imbalance_history").fetchall()
    conn.close()
    return {r['date'] for r in rows}


def save_prediction(pred):
    conn = get_db()
    conn.execute("""
        INSERT INTO prediction_log
        (predicted_at, target_date, target_isp, target_interval, probability,
         risk_level, predicted_negative, model_version, features_json)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (pred['predicted_at'], pred['target_date'], pred['target_isp'],
          pred.get('target_interval'), pred['probability'],
          pred['risk_level'], pred['predicted_negative'],
          pred.get('model_version'), pred.get('features_json')))
    conn.commit()
    conn.close()


def get_recent_predictions(hours=6):
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM prediction_log
        WHERE predicted_at >= datetime('now', ?)
        ORDER BY predicted_at DESC
    """, (f'-{hours} hours',)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def backfill_prediction_actuals(date_str, isp, actual_price):
    conn = get_db()
    was_neg = 1 if actual_price is not None and actual_price < 0 else 0
    conn.execute("""
        UPDATE prediction_log SET actual_neg_price=?, actual_was_negative=?
        WHERE target_date=? AND target_isp=? AND actual_neg_price IS NULL
    """, (actual_price, was_neg, date_str, isp))
    conn.commit()
    conn.close()


def save_model_metadata(meta):
    conn = get_db()
    conn.execute("""
        INSERT INTO model_metadata
        (trained_at, training_days, training_samples, accuracy, precision_score,
         recall, f1, auc_roc, threshold, model_path, features_used)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (meta['trained_at'], meta.get('training_days'), meta.get('training_samples'),
          meta.get('accuracy'), meta.get('precision_score'),
          meta.get('recall'), meta.get('f1'), meta.get('auc_roc'),
          meta.get('threshold'), meta.get('model_path'), meta.get('features_used')))
    conn.commit()
    conn.close()


def update_damas_sen_columns(date, isp, sen_solar, sen_wind, sen_consumption,
                              sen_sold, sen_surplus, sen_renewable_pct):
    """Update SEN columns for an existing DAMAS history row."""
    conn = get_db()
    conn.execute("""
        UPDATE damas_imbalance_history
        SET sen_solar=?, sen_wind=?, sen_consumption=?, sen_sold=?,
            sen_surplus=?, sen_renewable_pct=?
        WHERE date=? AND isp=?
    """, (sen_solar, sen_wind, sen_consumption, sen_sold,
          sen_surplus, sen_renewable_pct, date, isp))
    conn.commit()
    conn.close()


def bulk_update_damas_sen(rows):
    """Bulk update SEN columns. rows = list of dicts with date, isp, sen_* fields."""
    conn = get_db()
    for r in rows:
        conn.execute("""
            UPDATE damas_imbalance_history
            SET sen_solar=?, sen_wind=?, sen_consumption=?, sen_sold=?,
                sen_surplus=?, sen_renewable_pct=?
            WHERE date=? AND isp=?
        """, (r.get('sen_solar'), r.get('sen_wind'), r.get('sen_consumption'),
              r.get('sen_sold'), r.get('sen_surplus'), r.get('sen_renewable_pct'),
              r['date'], r['isp']))
    conn.commit()
    conn.close()


def get_latest_model_metadata():
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM model_metadata ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None
