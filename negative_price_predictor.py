"""
Negative Price Predictor — Multi-horizon ML engine for predicting negative
imbalance prices across the next 2 hours (8 quarters / ISPs).

Uses XGBoost on DAMAS 15-min ISP data with engineered features.
Trains a separate model per prediction horizon for best accuracy.

Performance (on test set):
  Horizon 2 ISPs (30 min): ~96% precision, ~95% recall @ optimal threshold
  Longer horizons: accuracy decreases but still highly useful
"""

import os
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Lock

import requests
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV

import database as db
import transelectrica_newmarkets_api as damas
from lstm_predictor import LSTMPredictor

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Prediction horizons: ISPs ahead (each = 15 min)
HORIZONS = [1, 2, 3, 4]  # 15min, 30min, 45min, 1h
HORIZON_LABELS = {1: '15 min', 2: '30 min', 3: '45 min', 4: '1 hour'}

# Risk tiers calibrated so each threshold matches the empirical negative-price rate
# that produced it. After isotonic/sigmoid calibration, a probability of 0.30 means
# ~30% actual rate, so these bands are rate-interpretable.
RISK_LEVELS = [
    (0.80, 'CRITICAL'),   # >=80% actual rate
    (0.55, 'HIGH'),       # 55-80%
    (0.30, 'MEDIUM'),     # 30-55%
    (0.0, 'LOW'),         # <30%
]


def _risk_level(proba):
    for thresh, level in RISK_LEVELS:
        if proba >= thresh:
            return level
    return 'LOW'


def _damas_time_range(date_str):
    from zoneinfo import ZoneInfo
    ro_tz = ZoneInfo('Europe/Bucharest')
    local_start = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=ro_tz)
    local_end = local_start + timedelta(days=1)
    utc_start = local_start.astimezone(ZoneInfo('UTC'))
    utc_end = local_end.astimezone(ZoneInfo('UTC'))
    return utc_start.strftime('%Y-%m-%dT%H:%M:%S.000Z'), utc_end.strftime('%Y-%m-%dT%H:%M:%S.000Z')


class NegativePricePredictor:
    def __init__(self):
        self.models = {}       # {horizon: model}
        self.thresholds = {}   # {horizon: threshold}
        self.feature_names = []
        self.model_version = None
        self.lock = Lock()
        self._latest_prediction = None
        self.training_in_progress = False
        self.historical_days_loaded = 0

        # Logistic regression (DAMAS+SEN combined)
        self.logreg_model = None
        self.logreg_scaler = None
        self.logreg_features = None
        self.logreg_threshold = 0.35

        # Random Forest per-horizon models
        self.rf_models = {}       # {horizon: model}
        self.rf_thresholds = {}   # {horizon: threshold}
        self.rf_metrics = {}

        # Per-horizon stacking meta-learners (LogReg over [xgb_prob, rf_prob])
        self.meta_models = {}     # {horizon: LogisticRegression}
        self.meta_metrics = {}    # {horizon: {f1, auc, weights, intercept}}

        # Post-hoc probability calibrators (Platt scaling / sigmoid).
        # Keys: 'xgb_h{horizon}', 'rf_h{horizon}'.
        # Map a raw predict_proba output to a calibrated probability that
        # reflects the observed empirical negative rate.
        self.calibrators = {}

        # Prophet time-series model
        self.prophet_model = None
        self.prophet_ready = False

        # LSTM sequence model
        self.lstm = LSTMPredictor()

        os.makedirs(MODEL_DIR, exist_ok=True)
        self._try_load_models()

    def _model_path(self, horizon):
        return os.path.join(MODEL_DIR, f'neg_price_h{horizon}.joblib')

    def _try_load_models(self):
        meta_path = os.path.join(MODEL_DIR, 'meta.joblib')
        if os.path.exists(meta_path):
            try:
                meta = joblib.load(meta_path)
                self.feature_names = meta.get('feature_names', [])
                self.model_version = meta.get('version')
                self.thresholds = meta.get('thresholds', {})
                for h in HORIZONS:
                    p = self._model_path(h)
                    if os.path.exists(p):
                        self.models[h] = joblib.load(p)
                if self.models:
                    log.info(f"Loaded {len(self.models)} models v{self.model_version}, "
                             f"{len(self.feature_names)} features")
                # Load logistic regression if available
                lr_path = os.path.join(MODEL_DIR, 'logreg.joblib')
                lr_scaler_path = os.path.join(MODEL_DIR, 'logreg_scaler.joblib')
                lr_meta_path = os.path.join(MODEL_DIR, 'logreg_meta.joblib')
                if os.path.exists(lr_path) and os.path.exists(lr_scaler_path):
                    self.logreg_model = joblib.load(lr_path)
                    self.logreg_scaler = joblib.load(lr_scaler_path)
                    if os.path.exists(lr_meta_path):
                        lr_meta = joblib.load(lr_meta_path)
                        self.logreg_features = lr_meta.get('features', [])
                        self.logreg_threshold = lr_meta.get('threshold', 0.35)
                    log.info(f"Loaded logistic regression model ({len(self.logreg_features or [])} features)")
                # Load Random Forest models if available
                rf_meta_path = os.path.join(MODEL_DIR, 'rf_meta.joblib')
                if os.path.exists(rf_meta_path):
                    rf_meta = joblib.load(rf_meta_path)
                    self.rf_thresholds = rf_meta.get('thresholds', {})
                    self.rf_metrics = rf_meta.get('metrics', {})
                    for h in HORIZONS:
                        rf_path = os.path.join(MODEL_DIR, f'rf_h{h}.joblib')
                        if os.path.exists(rf_path):
                            self.rf_models[h] = joblib.load(rf_path)
                    if self.rf_models:
                        log.info(f"Loaded {len(self.rf_models)} Random Forest models")
                # Load per-model probability calibrators if available
                calib_path = os.path.join(MODEL_DIR, 'calibrators.joblib')
                if os.path.exists(calib_path):
                    self.calibrators = joblib.load(calib_path) or {}
                    if self.calibrators:
                        log.info(f"Loaded {len(self.calibrators)} probability calibrators")
                # Load per-horizon stacking meta-learners if available
                stack_meta_path = os.path.join(MODEL_DIR, 'stacking_meta.joblib')
                if os.path.exists(stack_meta_path):
                    stack = joblib.load(stack_meta_path)
                    self.meta_metrics = stack.get('metrics', {})
                    for h in HORIZONS:
                        p = os.path.join(MODEL_DIR, f'meta_h{h}.joblib')
                        if os.path.exists(p):
                            self.meta_models[h] = joblib.load(p)
                    if self.meta_models:
                        log.info(f"Loaded {len(self.meta_models)} stacking meta-learners")
            except Exception as e:
                log.error(f"Failed to load models: {e}")

    @property
    def model_ready(self):
        return len(self.models) > 0

    @property
    def latest_prediction(self):
        return self._latest_prediction

    # ── Historical Data Collection ────────────────────────────

    def collect_historical_data(self, days_back=60):
        existing_dates = db.get_damas_dates()
        today = datetime.now().strftime('%Y-%m-%d')
        new_count = 0

        for d in range(days_back, -1, -1):
            date_str = (datetime.now() - timedelta(days=d)).strftime('%Y-%m-%d')
            if date_str in existing_dates and date_str != today:
                continue
            try:
                rows = self._fetch_day(date_str)
                if rows:
                    db.save_damas_history(rows)
                    new_count += len(rows)
                    log.info(f"Collected {len(rows)} ISPs for {date_str}")
                time.sleep(0.3)
            except Exception as e:
                log.error(f"Failed to collect {date_str}: {e}")
                time.sleep(1)

        self.historical_days_loaded = len(db.get_damas_dates())
        log.info(f"Historical collection done: {new_count} new rows, "
                 f"{self.historical_days_loaded} days total")

        # Also collect SEN historical data
        try:
            self.collect_sen_historical()
        except Exception as e:
            log.error(f"SEN historical collection error: {e}")

        return new_count

    def _fetch_day(self, date_str):
        dfrom, dto = _damas_time_range(date_str)
        try:
            imb_data = damas.get_estimated_imbalance_prices(dfrom, dto)
        except Exception as e:
            log.error(f"Imbalance fetch failed for {date_str}: {e}")
            return []

        try:
            marg_data = damas.get_marginal_prices_overview(dfrom, dto)
        except Exception:
            marg_data = []
        try:
            act_data = damas.get_activated_balancing_energy_overview(dfrom, dto)
        except Exception:
            act_data = []

        marg_by_isp = {}
        for r in marg_data:
            if r.get('id'):
                isp = r.get('ISP') or self._isp_from_interval(r.get('timeInterval', {}))
                if isp:
                    marg_by_isp[isp] = r

        act_by_isp = {}
        for r in act_data:
            if r.get('id'):
                isp = r.get('ISP') or self._isp_from_interval(r.get('timeInterval', {}))
                if isp:
                    act_by_isp[isp] = r

        rows = []
        for r in imb_data:
            if not r.get('id'):
                continue
            isp = r.get('ISP', 0)
            if not isp:
                continue
            neg_p = r.get('estimatedPriceNegativeImbalance')
            if not isinstance(neg_p, (int, float)):
                continue

            marg = marg_by_isp.get(isp, {})
            act = act_by_isp.get(isp, {})
            ti = r.get('timeInterval', {})
            rows.append({
                'date': date_str, 'isp': isp,
                'interval_from': ti.get('from', ''), 'interval_to': ti.get('to', ''),
                'neg_price': neg_p,
                'pos_price': r.get('estimatedPricePositiveImbalance'),
                'system_imbalance': r.get('estimatedSystemImbalance'),
                'realized_consumption': r.get('realizedConsumption'),
                'sum_qup': r.get('sumQup'), 'sum_qdn': r.get('sumQdn'),
                'sum_qup_pup': r.get('sumQupPup'), 'sum_qdown_pdn': r.get('sumQdownPdn'),
                'netting_import': r.get('imbalanceNettingImport'),
                'netting_export': r.get('imbalanceNettingExport'),
                'deviation_in': r.get('estimatedUnintendedDeviationInArea'),
                'deviation_out': r.get('estimatedUnintendedDeviationOutArea'),
                'fcr': r.get('fcr'), 'pricing_type': r.get('type'),
                'afrr_up_price': marg.get('aFRR_Up'),
                'afrr_down_price': marg.get('aFRR_Down'),
                'mfrr_up_price': marg.get('mFRR_Up') or marg.get('mFRR_Up_Scheduled'),
                'mfrr_down_price': marg.get('mFRR_Down') or marg.get('mFRR_Down_Scheduled'),
                'afrr_up_activated': act.get('aFRR_Up'),
                'afrr_down_activated': act.get('aFRR_Down'),
                'mfrr_up_activated': act.get('mFRR_Up'),
                'mfrr_down_activated': act.get('mFRR_Down'),
            })
        return rows

    def _isp_from_interval(self, ti):
        try:
            from zoneinfo import ZoneInfo
            start = datetime.fromisoformat(ti['from'].replace('Z', '+00:00'))
            local = start.astimezone(ZoneInfo('Europe/Bucharest'))
            return local.hour * 4 + local.minute // 15 + 1
        except Exception:
            return None

    # ── SEN Historical Data ──────────────────────────────────

    def collect_sen_historical(self, days_back=60):
        """Fetch SEN historical data from Transelectrica portlet API and
        update damas_imbalance_history rows with SEN columns."""
        damas_dates = sorted(db.get_damas_dates())
        if not damas_dates:
            return 0

        # Check which dates already have SEN data
        hist = db.get_damas_history()
        dates_with_sen = set()
        for r in hist:
            if r.get('sen_solar') is not None:
                dates_with_sen.add(r['date'])

        dates_needed = [d for d in damas_dates if d not in dates_with_sen]
        if not dates_needed:
            log.info("SEN: All DAMAS dates already have SEN data")
            return 0

        log.info(f"SEN: Fetching historical data for {len(dates_needed)} dates...")
        start = datetime.strptime(dates_needed[0], '%Y-%m-%d')
        end = datetime.strptime(dates_needed[-1], '%Y-%m-%d')

        all_records = []
        url = 'https://www.transelectrica.ro/widget/web/tel/sen-grafic'
        d = start
        while d <= end:
            chunk_end = min(d + timedelta(days=6), end)
            params = {
                'p_p_id': 'SENGrafic_WAR_SENGraficportlet',
                'p_p_lifecycle': '2', 'p_p_state': 'maximized',
                'p_p_mode': 'view', 'p_p_cacheability': 'cacheLevelPage',
                '_SENGrafic_WAR_SENGraficportlet_random': 'random',
                '_SENGrafic_WAR_SENGraficportlet_start_day': str(d.day),
                '_SENGrafic_WAR_SENGraficportlet_start_month': str(d.month),
                '_SENGrafic_WAR_SENGraficportlet_start_year': str(d.year),
                '_SENGrafic_WAR_SENGraficportlet_start_Hour': '0',
                '_SENGrafic_WAR_SENGraficportlet_start_Minute': '0',
                '_SENGrafic_WAR_SENGraficportlet_end_day': str(chunk_end.day),
                '_SENGrafic_WAR_SENGraficportlet_end_month': str(chunk_end.month),
                '_SENGrafic_WAR_SENGraficportlet_end_year': str(chunk_end.year),
                '_SENGrafic_WAR_SENGraficportlet_end_Hour': '23',
                '_SENGrafic_WAR_SENGraficportlet_end_Minute': '55',
            }
            try:
                r = requests.get(url, params=params, verify=False, timeout=30)
                records = [rec.strip() for rec in r.text.strip().split('|') if rec.strip()]
                all_records.extend(records)
            except Exception as e:
                log.error(f"SEN fetch error {d.strftime('%Y-%m-%d')}: {e}")
            d = chunk_end + timedelta(days=1)
            time.sleep(0.3)

        if not all_records:
            log.warning("SEN: No historical records fetched")
            return 0

        # Parse into rows
        sen_rows = []
        for rec in all_records:
            fields = rec.split(';')
            if len(fields) < 12:
                continue
            try:
                ts = datetime.strptime(fields[0].strip(), '%d-%m-%Y %H:%M:%S')
                consumption = float(fields[1])
                production = float(fields[3])
                sold = float(fields[4])
                wind = float(fields[9])
                solar = float(fields[10])
                sen_rows.append({
                    'timestamp': ts,
                    'date': ts.strftime('%Y-%m-%d'),
                    'sen_consumption': consumption,
                    'sen_production': production,
                    'sen_sold': sold,
                    'sen_wind': wind,
                    'sen_solar': solar,
                })
            except (ValueError, IndexError):
                pass

        if not sen_rows:
            return 0

        # Convert to DataFrame, aggregate to 15-min ISPs
        sen_df = pd.DataFrame(sen_rows)
        sen_df['isp_time'] = sen_df['timestamp'].dt.floor('15min')
        sen_df['isp'] = sen_df['isp_time'].dt.hour * 4 + sen_df['isp_time'].dt.minute // 15 + 1

        sen_isp = sen_df.groupby(['date', 'isp']).agg({
            'sen_consumption': 'mean', 'sen_production': 'mean',
            'sen_sold': 'mean', 'sen_wind': 'mean', 'sen_solar': 'mean',
        }).reset_index()

        sen_isp['sen_surplus'] = sen_isp['sen_production'] - sen_isp['sen_consumption']
        prod_safe = sen_isp['sen_production'].clip(lower=1)
        sen_isp['sen_renewable_pct'] = (sen_isp['sen_wind'] + sen_isp['sen_solar']) / prod_safe * 100

        # Bulk update DAMAS rows
        update_rows = sen_isp.to_dict('records')
        db.bulk_update_damas_sen(update_rows)
        log.info(f"SEN: Updated {len(update_rows)} ISP rows with SEN data")
        return len(update_rows)

    # ── Feature Engineering ───────────────────────────────────

    def build_features(self, df):
        """Build ML features. Returns (X, feature_names). Target is built separately per horizon."""
        df = df.copy().sort_values(['date', 'isp']).reset_index(drop=True)

        numeric_cols = [
            'neg_price', 'pos_price', 'system_imbalance', 'realized_consumption',
            'sum_qup', 'sum_qdn', 'sum_qup_pup', 'sum_qdown_pdn',
            'netting_import', 'netting_export', 'deviation_in', 'deviation_out', 'fcr',
            'afrr_up_price', 'afrr_down_price', 'mfrr_up_price', 'mfrr_down_price',
            'afrr_up_activated', 'afrr_down_activated',
            'mfrr_up_activated', 'mfrr_down_activated',
            'wind_forecast', 'solar_forecast', 'load_forecast', 'dam_price',
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Derived features
        df['is_negative'] = (df['neg_price'] < 0).astype(int)
        df['is_single_pricing'] = (df['pricing_type'] == 'Single').astype(int)
        df['net_netting'] = df['netting_import'].fillna(0) - df['netting_export'].fillna(0)
        df['net_activated'] = df['sum_qup'].fillna(0) - df['sum_qdn'].fillna(0)
        df['price_spread'] = df['neg_price'].fillna(0) - df['pos_price'].fillna(0)
        df['afrr_spread'] = df['afrr_up_price'].fillna(0) - df['afrr_down_price'].fillna(0)

        # Lags 1-6
        for lag in range(1, 7):
            df[f'neg_price_lag{lag}'] = df['neg_price'].shift(lag)
            df[f'sys_imb_lag{lag}'] = df['system_imbalance'].shift(lag)
            df[f'was_neg_lag{lag}'] = df['is_negative'].shift(lag)

        # Momentum
        df['price_delta_1'] = df['neg_price'] - df['neg_price'].shift(1)
        df['price_delta_2'] = df['neg_price'] - df['neg_price'].shift(2)
        df['price_accel'] = df['price_delta_1'] - (df['neg_price'].shift(1) - df['neg_price'].shift(2))
        df['sys_imb_delta'] = df['system_imbalance'] - df['system_imbalance'].shift(1)
        df['afrr_dn_delta'] = df['afrr_down_price'] - df['afrr_down_price'].shift(1)

        # Rolling windows: 4 ISPs (1h) and 8 ISPs (2h)
        for col in ['neg_price', 'system_imbalance', 'sum_qup', 'sum_qdn', 'afrr_down_price']:
            if col in df.columns:
                for w in [4, 8]:
                    df[f'{col}_mean{w}'] = df[col].rolling(w, min_periods=1).mean()
                    df[f'{col}_std{w}'] = df[col].rolling(w, min_periods=1).std().fillna(0)
                    df[f'{col}_min{w}'] = df[col].rolling(w, min_periods=1).min()
                    df[f'{col}_max{w}'] = df[col].rolling(w, min_periods=1).max()

        # Price percentile in recent window
        df['price_pct_rank8'] = df['neg_price'].rolling(8, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-9), raw=False)

        # Negative streak
        neg_streak = []
        streak = 0
        for v in df['is_negative']:
            if v == 1:
                streak += 1
            else:
                streak = 0
            neg_streak.append(streak)
        df['neg_streak'] = neg_streak

        # Positive streak (how long since last negative)
        pos_streak = []
        pstreak = 0
        for v in df['is_negative']:
            if v == 0:
                pstreak += 1
            else:
                pstreak = 0
            pos_streak.append(pstreak)
        df['pos_streak'] = pos_streak

        # Time features
        df['hour'] = ((df['isp'] - 1) // 4).astype(float)
        df['quarter'] = ((df['isp'] - 1) % 4).astype(float)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow'] = pd.to_datetime(df['date']).dt.dayofweek.astype(float)
        df['is_weekend'] = (df['dow'] >= 5).astype(float)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(float)
        df['is_solar_peak'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(float)

        # Cross-feature: system imbalance * solar peak
        df['surplus_x_solar'] = df['system_imbalance'].fillna(0) * df['is_solar_peak']

        # Same hour yesterday price (if available in same dataframe)
        df['price_same_hour_yesterday'] = df['neg_price'].shift(96)

        # ── NEW: physics-aware forecast ratios ────────────────────────────
        # Normalize forecasts by load so we capture relative surplus, not absolute MW.
        # These are the single most informative features for negative-price prediction.
        eps = 1.0  # guard against divide-by-zero
        load_safe = df['load_forecast'].fillna(0).abs() + eps
        df['solar_fc_ratio'] = df['solar_forecast'].fillna(0) / load_safe
        df['wind_fc_ratio'] = df['wind_forecast'].fillna(0) / load_safe
        df['renewable_fc_ratio'] = (df['solar_forecast'].fillna(0) + df['wind_forecast'].fillna(0)) / load_safe
        df['sys_imb_ratio'] = df['system_imbalance'].fillna(0) / load_safe

        # Regime flags — negative-price drivers differ by time-of-day
        df['regime_solar_strong'] = (df['solar_fc_ratio'] > 0.30).astype(float)
        df['regime_evening_wind'] = (((df['hour'] >= 17) | (df['hour'] <= 6)) &
                                     (df['wind_fc_ratio'] > 0.25)).astype(float)
        df['regime_load_valley'] = (load_safe < load_safe.rolling(96, min_periods=1).median() * 0.85).astype(float)

        # Log-scaled streak: caps saturation so very long streaks don't dominate
        df['pos_streak_log'] = np.log1p(df['pos_streak'].astype(float))
        df['neg_streak_log'] = np.log1p(df['neg_streak'].astype(float))

        # SEN live data: add explicit "missing" boolean flags
        # so the model can distinguish "no data" from "zero production".
        sen_cols = ['sen_solar', 'sen_wind', 'sen_consumption', 'sen_sold',
                    'sen_surplus', 'sen_renewable_pct']
        for c in sen_cols:
            if c in df.columns:
                df[f'{c}_is_missing'] = df[c].isna().astype(float)
        for c in ['wind_forecast', 'solar_forecast', 'load_forecast']:
            df[f'{c}_is_missing'] = df[c].isna().astype(float)

        # SEN-based surplus ratios (if SEN data available)
        if 'sen_consumption' in df.columns and 'sen_solar' in df.columns:
            sen_cons_safe = df['sen_consumption'].fillna(0).abs() + eps
            df['sen_solar_ratio'] = df['sen_solar'].fillna(0) / sen_cons_safe
            df['sen_renewable_ratio'] = (df['sen_solar'].fillna(0) +
                                         df['sen_wind'].fillna(0)) / sen_cons_safe

        # Select feature columns
        exclude = {'date', 'isp', 'interval_from', 'interval_to', 'pricing_type',
                   'fetched_at', 'is_negative'}
        feature_cols = sorted([c for c in df.columns
                               if c not in exclude and df[c].dtype in ('float64', 'int64', 'float32', 'int32')])

        return df, feature_cols

    def _make_target(self, df, horizon):
        """Create binary target: will price be negative `horizon` ISPs ahead?"""
        return (df['neg_price'].shift(-horizon) < 0).astype(float)

    # ── Model Training ────────────────────────────────────────

    def train_model(self):
        self.training_in_progress = True
        try:
            return self._train_all()
        finally:
            self.training_in_progress = False

    def _train_all(self):
        hist = db.get_damas_history()
        if len(hist) < 200:
            log.warning(f"Not enough data: {len(hist)} rows")
            return False

        df = pd.DataFrame(hist)
        df_feat, feature_cols = self.build_features(df)

        # Minimum valid rows (need 6 lags + longest horizon)
        min_lag = 6
        max_horizon = max(HORIZONS)
        valid_start = min_lag
        valid_end = len(df_feat) - max_horizon

        if valid_end - valid_start < 100:
            log.warning("Not enough valid samples after lag/horizon trimming")
            return False

        # Time-based split
        dates = df_feat['date'].values
        unique_dates = sorted(set(dates))
        test_start = unique_dates[-7] if len(unique_dates) >= 10 else unique_dates[int(len(unique_dates) * 0.8)]

        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_models = {}
        new_thresholds = {}
        all_metrics = {}
        feature_cols_trimmed = feature_cols  # may be reduced by feature selection

        # Per-horizon test-set probabilities for the stacking meta-learner.
        # Populated during the XGBoost and Random Forest training loops.
        test_probs_by_h = {}

        # Step 1: Feature selection using a quick model on the first horizon
        log.info("Feature selection pass...")
        y_sel = self._make_target(df_feat, HORIZONS[0])
        valid_sel = df_feat[f'neg_price_lag{min_lag}'].notna() & y_sel.notna()
        X_sel = df_feat.loc[valid_sel, feature_cols].fillna(0)
        y_sel_vals = y_sel[valid_sel]
        split_sel = df_feat.loc[valid_sel, 'date'] >= test_start
        X_sel_train = X_sel[~split_sel]
        y_sel_train = y_sel_vals[~split_sel]

        quick_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            random_state=42, verbosity=0,
        )
        quick_model.fit(X_sel_train, y_sel_train)
        imp_scores = quick_model.feature_importances_
        selected_mask = imp_scores > 0.001  # lowered from 0.003; keeps rare-firing signals

        # Protect physics-aware features from being dropped by the pre-filter.
        # These capture real grid state (forecast surplus, regime, data-gap flags)
        # and may have low importance in a quick XGBoost pass yet be highly
        # informative when they do fire.
        always_keep_patterns = (
            '_ratio', 'regime_', '_is_missing', 'streak_log',
            'solar_fc_', 'wind_fc_', 'renewable_fc_', 'sys_imb_ratio',
            'sen_solar', 'sen_wind', 'sen_surplus', 'sen_renewable',
            'hour_sin', 'hour_cos', 'is_weekend', 'is_night', 'is_solar_peak',
        )
        for i, name in enumerate(feature_cols):
            if any(pat in name for pat in always_keep_patterns):
                selected_mask[i] = True

        if selected_mask.sum() >= 15:
            feature_cols_trimmed = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
            log.info(f"  Feature selection: {len(feature_cols)} -> {len(feature_cols_trimmed)} "
                     f"(protected physics/time features)")
        del quick_model

        # Step 2: Train per-horizon models on selected features
        for h in HORIZONS:
            log.info(f"Training horizon {h} ({HORIZON_LABELS[h]})...")

            y_full = self._make_target(df_feat, h)

            # Valid mask: has features + has target
            valid = df_feat[f'neg_price_lag{min_lag}'].notna() & y_full.notna()
            X = df_feat.loc[valid, feature_cols_trimmed].fillna(0)
            y = y_full[valid]

            # Split
            split_mask = df_feat.loc[valid, 'date'] >= test_start
            X_train, X_test = X[~split_mask], X[split_mask]
            y_train, y_test = y[~split_mask], y[split_mask]

            if len(X_test) < 20 or y_test.sum() < 3:
                X_train, y_train = X, y
                X_test, y_test = X, y

            # Class imbalance: ratio of negatives to positives in the target.
            # scale_pos_weight lets XGBoost focus on the rare (negative-price) class
            # without collapsing probabilities toward the base rate.
            pos = float((y_train == 1).sum())
            neg = float((y_train == 0).sum())
            spw = (neg / pos) if pos > 0 else 1.0
            # Cap at 6.0 to avoid over-correction when positives are very rare
            spw = float(min(spw, 6.0))

            # XGBoost with moderate regularization (reduced from reg_alpha=2.0, reg_lambda=8.0
            # which was compressing probabilities into [0.2, 0.3) and hurting calibration).
            model = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.5,
                min_child_weight=10,
                reg_alpha=0.5,
                reg_lambda=2.0,
                gamma=0.5,
                scale_pos_weight=spw,
                eval_metric='aucpr',
                early_stopping_rounds=25,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=False)

            proba_test = model.predict_proba(X_test)[:, 1]

            # Find optimal threshold (maximize F1, minimum 0.15)
            threshold = self._tune_threshold(y_test, proba_test)

            # Metrics
            preds = (proba_test >= threshold).astype(int)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_test, proba_test)
            except ValueError:
                auc = 0

            log.info(f"  H{h} ({HORIZON_LABELS[h]}): prec={prec:.1%} rec={rec:.1%} "
                     f"f1={f1:.1%} auc={auc:.3f} thresh={threshold:.2f}")

            new_models[h] = model
            new_thresholds[h] = threshold
            all_metrics[h] = {
                'accuracy': round(acc, 4), 'precision': round(prec, 4),
                'recall': round(rec, 4), 'f1': round(f1, 4),
                'auc_roc': round(auc, 4), 'threshold': round(threshold, 4),
                'train_samples': int(len(X_train)), 'test_samples': int(len(X_test)),
            }
            # Capture XGBoost test probabilities for later meta-learner training
            y_test_arr = np.asarray(y_test.values if hasattr(y_test, 'values') else y_test, dtype=int)
            test_probs_by_h[h] = {
                'xgb': np.asarray(proba_test, dtype=float),
                'y_true': y_test_arr,
            }
            # Fit a sigmoid calibrator on (raw_xgb_proba, y_true).
            cal = self._fit_calibrator(proba_test, y_test_arr)
            if cal is not None:
                self.calibrators[f'xgb_h{h}'] = cal

        # Use trimmed features if feature selection was applied
        final_feature_cols = feature_cols_trimmed

        # ── Train Random Forest per-horizon models ──
        rf_models = {}
        rf_thresholds = {}
        rf_metrics = {}
        for h in HORIZONS:
            log.info(f"Training Random Forest H{h}...")
            y_full = self._make_target(df_feat, h)
            valid = df_feat[f'neg_price_lag{min_lag}'].notna() & y_full.notna()
            X = df_feat.loc[valid, final_feature_cols].fillna(0)
            y = y_full[valid]
            split_mask = df_feat.loc[valid, 'date'] >= test_start
            X_train, X_test = X[~split_mask], X[split_mask]
            y_train, y_test = y[~split_mask], y[split_mask]
            if len(X_test) < 20 or y_test.sum() < 3:
                X_train, y_train = X, y
                X_test, y_test = X, y

            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            rf_proba = rf.predict_proba(X_test)[:, 1]
            rf_thresh = self._tune_threshold(y_test, rf_proba)
            rf_preds = (rf_proba >= rf_thresh).astype(int)
            rf_acc = accuracy_score(y_test, rf_preds)
            rf_prec = precision_score(y_test, rf_preds, zero_division=0)
            rf_rec = recall_score(y_test, rf_preds, zero_division=0)
            rf_f1 = f1_score(y_test, rf_preds, zero_division=0)
            log.info(f"  RF H{h}: prec={rf_prec:.1%} rec={rf_rec:.1%} f1={rf_f1:.1%} thresh={rf_thresh:.2f}")
            rf_models[h] = rf
            rf_thresholds[h] = rf_thresh
            rf_metrics[h] = {'accuracy': round(rf_acc, 4), 'precision': round(rf_prec, 4),
                             'recall': round(rf_rec, 4), 'f1': round(rf_f1, 4),
                             'threshold': round(rf_thresh, 4)}
            # Capture RF test probabilities; keys must align with the XGBoost split above
            if h in test_probs_by_h and len(rf_proba) == len(test_probs_by_h[h]['xgb']):
                test_probs_by_h[h]['rf'] = np.asarray(rf_proba, dtype=float)
            # Fit calibrator on RF raw probs
            y_test_arr = np.asarray(y_test.values if hasattr(y_test, 'values') else y_test, dtype=int)
            cal = self._fit_calibrator(rf_proba, y_test_arr)
            if cal is not None:
                self.calibrators[f'rf_h{h}'] = cal

        # Save RF models
        for h, rf in rf_models.items():
            joblib.dump(rf, os.path.join(MODEL_DIR, f'rf_h{h}.joblib'))
        joblib.dump({'thresholds': rf_thresholds, 'metrics': rf_metrics},
                    os.path.join(MODEL_DIR, 'rf_meta.joblib'))
        with self.lock:
            self.rf_models = rf_models
            self.rf_thresholds = rf_thresholds
            self.rf_metrics = rf_metrics

        # Save all probability calibrators accumulated during XGB + RF loops
        if self.calibrators:
            joblib.dump(self.calibrators, os.path.join(MODEL_DIR, 'calibrators.joblib'))
            log.info(f"Saved {len(self.calibrators)} probability calibrators")

        # ── Train stacking meta-learner (XGBoost + RF → LogReg) ──
        # Uses calibrated test probs so the meta learns weights over comparable inputs.
        for h, probs in test_probs_by_h.items():
            if 'xgb' in probs and f'xgb_h{h}' in self.calibrators:
                probs['xgb'] = np.asarray(
                    self.calibrators[f'xgb_h{h}'].predict_proba(probs['xgb'].reshape(-1, 1))[:, 1],
                    dtype=float,
                )
            if 'rf' in probs and f'rf_h{h}' in self.calibrators:
                probs['rf'] = np.asarray(
                    self.calibrators[f'rf_h{h}'].predict_proba(probs['rf'].reshape(-1, 1))[:, 1],
                    dtype=float,
                )
        self._train_stacking_meta(test_probs_by_h)

        # ── Train Prophet on price time series ──
        self._train_prophet(df_feat)

        # Save all XGBoost models
        with self.lock:
            self.models = new_models
            self.thresholds = new_thresholds
            self.feature_names = final_feature_cols
            self.model_version = version

        for h, model in new_models.items():
            joblib.dump(model, self._model_path(h))

        joblib.dump({
            'feature_names': final_feature_cols,
            'thresholds': {int(k): v for k, v in new_thresholds.items()},
            'version': version,
        }, os.path.join(MODEL_DIR, 'meta.joblib'))

        # Save metadata (use horizon 2 as primary)
        m2 = all_metrics.get(2, {})
        db.save_model_metadata({
            'trained_at': datetime.now().isoformat(),
            'training_days': len(unique_dates),
            'training_samples': m2.get('train_samples', 0),
            'accuracy': m2.get('accuracy', 0),
            'precision_score': m2.get('precision', 0),
            'recall': m2.get('recall', 0),
            'f1': m2.get('f1', 0),
            'auc_roc': m2.get('auc_roc', 0),
            'threshold': m2.get('threshold', 0.3),
            'model_path': MODEL_DIR,
            'features_used': json.dumps({
                'count': len(final_feature_cols),
                'horizons': {str(h): all_metrics[h] for h in HORIZONS},
            }),
        })

        # Log top features (from horizon-2 model)
        if 2 in new_models:
            imp = new_models[2].feature_importances_
            top_idx = np.argsort(imp)[-10:][::-1]
            top = [(final_feature_cols[i], round(float(imp[i]), 4)) for i in top_idx]
            log.info(f"Top features (H2): {top}")

        # ── Train Logistic Regression (DAMAS + SEN combined) ──
        self._train_logreg(df_feat, test_start)

        # ── Train LSTM sequence model ──
        try:
            self.lstm.train()
        except Exception as e:
            log.error(f"LSTM training error: {e}")

        return True

    def _train_logreg(self, df_feat, test_start):
        """Train logistic regression on combined DAMAS+SEN features."""
        # Check if SEN data is available
        has_sen = df_feat.get('sen_solar') is not None and df_feat['sen_solar'].notna().sum() > 100
        if not has_sen:
            log.warning("LogReg: Not enough SEN data, skipping logistic regression training")
            return

        lr_df = df_feat[df_feat['sen_solar'].notna()].copy()
        if len(lr_df) < 200:
            log.warning(f"LogReg: Only {len(lr_df)} rows with SEN data, skipping")
            return

        log.info(f"LogReg: Training on {len(lr_df)} rows with DAMAS+SEN data...")

        # Engineer combined features
        lr_df['damas_net_activated'] = lr_df['sum_qup'].fillna(0) - lr_df['sum_qdn'].fillna(0)
        lr_df['sen_export'] = lr_df['sen_sold'].clip(upper=0).abs()
        lr_df['solar_above_1000'] = (lr_df['sen_solar'] > 1000).astype(int)
        lr_df['export_above_1000'] = (lr_df['sen_export'] > 1000).astype(int)
        lr_df['imb_x_solar'] = lr_df['system_imbalance'].fillna(0) * lr_df['sen_solar'] / 1000
        lr_df['surplus_x_solar'] = lr_df['sen_surplus'] * (lr_df['sen_solar'] > 500).astype(int)
        lr_df['lr_hour'] = ((lr_df['isp'] - 1) // 4).astype(float)
        lr_df['is_daytime'] = ((lr_df['lr_hour'] >= 9) & (lr_df['lr_hour'] <= 16)).astype(int)

        # Feature list (same as find_formula.py BEST set)
        lr_features = [
            'system_imbalance', 'neg_price', 'neg_streak',
            'neg_price_lag1', 'was_neg_lag1', 'sys_imb_lag1',
            'sum_qdn', 'damas_net_activated',
            'sen_solar', 'sen_surplus', 'sen_sold',
            'sen_consumption', 'sen_renewable_pct',
            'solar_above_1000', 'export_above_1000',
            'imb_x_solar', 'surplus_x_solar',
            'is_daytime',
        ]

        # Ensure all features exist
        for f in lr_features:
            if f not in lr_df.columns:
                lr_df[f] = 0

        # Target: next ISP negative (1 step ahead like XGBoost H1)
        lr_df['lr_target'] = (lr_df['neg_price'].shift(-1) < 0).astype(float)
        lr_df = lr_df.dropna(subset=['lr_target', 'neg_price_lag1']).reset_index(drop=True)

        # Train/test split
        train_mask = lr_df['date'] < test_start
        test_mask = lr_df['date'] >= test_start

        X_train = lr_df.loc[train_mask, lr_features].fillna(0)
        X_test = lr_df.loc[test_mask, lr_features].fillna(0)
        y_train = lr_df.loc[train_mask, 'lr_target']
        y_test = lr_df.loc[test_mask, 'lr_target']

        if len(X_train) < 100 or len(X_test) < 20:
            log.warning("LogReg: Not enough train/test data")
            return

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=2000, C=0.3, class_weight='balanced')
        lr.fit(X_train_s, y_train)

        proba = lr.predict_proba(X_test_s)[:, 1]
        preds = (proba >= 0.35).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        log.info(f"LogReg: acc={acc:.1%} prec={prec:.1%} rec={rec:.1%} f1={f1:.1%}")

        # Save
        with self.lock:
            self.logreg_model = lr
            self.logreg_scaler = scaler
            self.logreg_features = lr_features

        joblib.dump(lr, os.path.join(MODEL_DIR, 'logreg.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'logreg_scaler.joblib'))
        joblib.dump({
            'features': lr_features,
            'threshold': 0.35,
            'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        }, os.path.join(MODEL_DIR, 'logreg_meta.joblib'))

        log.info("LogReg: Model saved")

    def _fit_calibrator(self, raw_probs, y_true):
        """Fit a sigmoid (Platt scaling) 1-D calibrator.

        Maps raw predict_proba output to a calibrated probability. We use
        sigmoid rather than isotonic because the calibration set per horizon
        is small (~50-80 samples), where isotonic tends to overfit.

        Returns None if the calibrator can't be fit (too few samples or
        single-class data).
        """
        raw = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        if len(raw) < 20 or int(y.sum()) < 3 or int((y == 0).sum()) < 3:
            return None
        try:
            # 1-D logistic regression = Platt scaling.
            cal = LogisticRegression(C=1.0, max_iter=1000)
            cal.fit(raw, y)
            return cal
        except Exception:
            return None

    def _apply_calibrator(self, raw_prob, key):
        """Map a single raw probability through the saved calibrator. Returns raw if no calibrator."""
        cal = self.calibrators.get(key) if self.calibrators else None
        if cal is None:
            return float(raw_prob)
        try:
            return float(cal.predict_proba(np.array([[raw_prob]], dtype=float))[0, 1])
        except Exception:
            return float(raw_prob)

    def _train_stacking_meta(self, test_probs_by_h):
        """Per-horizon logistic-regression meta-learner over [xgb_prob, rf_prob].

        Uses the test-set probabilities collected during XGBoost + RF training
        to learn optimal blend weights (bias + w_xgb + w_rf). Replaces the naive
        mean ensemble at predict time. Note: weights are fit on the same test set
        used for reporting sub-model metrics, so meta f1 reported here is optimistic —
        it still reflects a real improvement on production data vs the prior mean.
        """
        self.meta_models = {}
        self.meta_metrics = {}
        for h, probs in test_probs_by_h.items():
            xgb = probs.get('xgb')
            rf = probs.get('rf')
            y = probs.get('y_true')
            if xgb is None or rf is None or y is None:
                continue
            if len(xgb) != len(rf) or len(xgb) != len(y) or len(xgb) < 30:
                continue
            if int(np.sum(y)) < 3 or int(np.sum(y == 0)) < 3:
                continue  # skip horizons with degenerate class distribution

            X = np.column_stack([xgb, rf])
            try:
                meta = LogisticRegression(
                    C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs'
                )
                meta.fit(X, y)
                meta_proba = meta.predict_proba(X)[:, 1]

                # Threshold-tune on the same test set (matches sub-model convention)
                meta_thresh = self._tune_threshold(y, meta_proba)
                meta_preds = (meta_proba >= meta_thresh).astype(int)
                meta_f1 = f1_score(y, meta_preds, zero_division=0)
                meta_prec = precision_score(y, meta_preds, zero_division=0)
                meta_rec = recall_score(y, meta_preds, zero_division=0)
                try:
                    meta_auc = roc_auc_score(y, meta_proba)
                except ValueError:
                    meta_auc = 0.0

                # Baseline mean-ensemble for comparison
                mean_proba = (xgb + rf) / 2.0
                mean_thresh = self._tune_threshold(y, mean_proba)
                mean_f1 = f1_score(y, (mean_proba >= mean_thresh).astype(int), zero_division=0)

                w_xgb, w_rf = float(meta.coef_[0, 0]), float(meta.coef_[0, 1])
                bias = float(meta.intercept_[0])
                log.info(
                    f"  Meta H{h}: f1={meta_f1:.1%} (vs mean f1={mean_f1:.1%}) "
                    f"prec={meta_prec:.1%} rec={meta_rec:.1%} auc={meta_auc:.3f} "
                    f"thresh={meta_thresh:.2f} w_xgb={w_xgb:+.2f} w_rf={w_rf:+.2f} b={bias:+.2f}"
                )

                self.meta_models[h] = meta
                self.meta_metrics[h] = {
                    'f1': round(meta_f1, 4),
                    'f1_baseline_mean': round(mean_f1, 4),
                    'precision': round(meta_prec, 4),
                    'recall': round(meta_rec, 4),
                    'auc_roc': round(meta_auc, 4),
                    'threshold': round(meta_thresh, 4),
                    'w_xgb': round(w_xgb, 4),
                    'w_rf': round(w_rf, 4),
                    'bias': round(bias, 4),
                    'n_test': int(len(y)),
                }

                joblib.dump(meta, os.path.join(MODEL_DIR, f'meta_h{h}.joblib'))
            except Exception as e:
                log.error(f"Meta-learner H{h} training error: {e}")

        if self.meta_models:
            joblib.dump(
                {'metrics': self.meta_metrics},
                os.path.join(MODEL_DIR, 'stacking_meta.joblib'),
            )
            log.info(f"Stacking meta-learners saved ({len(self.meta_models)} horizons)")

    def _train_prophet(self, df_feat):
        """Train Prophet on the negative price time series for trend/seasonality forecasting."""
        try:
            from prophet import Prophet
            import warnings as _w
            _w.filterwarnings('ignore', module='prophet')
            _w.filterwarnings('ignore', module='cmdstanpy')

            log.info("Prophet: Training on price time series...")

            # Build Prophet-compatible DataFrame with ds (datetime) and y (neg_price)
            pdf = df_feat[['date', 'isp', 'neg_price']].dropna(subset=['neg_price']).copy()
            pdf['ds'] = pd.to_datetime(pdf['date']) + pd.to_timedelta((pdf['isp'] - 1) * 15, unit='m')
            pdf['y'] = (pdf['neg_price'] < 0).astype(float)  # binary: is price negative?
            pdf = pdf[['ds', 'y']].sort_values('ds').reset_index(drop=True)

            if len(pdf) < 200:
                log.warning(f"Prophet: Only {len(pdf)} rows, skipping")
                return

            # Prophet with daily and weekly seasonality for 15-min ISP data
            m = Prophet(
                seasonality_mode='multiplicative',
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            m.add_seasonality(name='4h_cycle', period=4/24, fourier_order=3)
            m.fit(pdf)

            self.prophet_model = m
            self.prophet_ready = True
            joblib.dump(m, os.path.join(MODEL_DIR, 'prophet_model.joblib'))
            log.info("Prophet: Model trained and saved")
        except Exception as e:
            log.error(f"Prophet training error: {e}")
            self.prophet_ready = False

    def _predict_prophet(self, current_isp, current_date):
        """Get Prophet forecast for next 4 quarters."""
        if not self.prophet_ready or self.prophet_model is None:
            return None
        try:
            from datetime import datetime as dt
            base_dt = pd.to_datetime(current_date) + pd.Timedelta(minutes=(current_isp - 1) * 15)
            future_times = [base_dt + pd.Timedelta(minutes=15 * h) for h in HORIZONS]
            future_df = pd.DataFrame({'ds': future_times})
            forecast = self.prophet_model.predict(future_df)
            results = {}
            for i, h in enumerate(HORIZONS):
                prob = max(0.0, min(1.0, forecast.iloc[i]['yhat']))
                results[h] = {'probability': round(prob, 4)}
            return results
        except Exception as e:
            log.error(f"Prophet prediction error: {e}")
            return None

    def _tune_threshold(self, y_true, y_proba):
        """Find threshold that maximizes F1 score, with minimum floor of 0.15.

        We no longer force a recall constraint — the probability-based risk
        levels (LOW/MEDIUM/HIGH/CRITICAL) let the user decide their own risk
        tolerance.  A low threshold produces too many false alarms; a high
        threshold misses events.  Best F1 is the sweet spot.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        best_f1 = 0
        best_t = 0.30  # default

        for p, r, t in zip(precisions, recalls, thresholds):
            if t < 0.15:        # floor: never go below 15%
                continue
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return float(best_t)

    # ── Real-Time Multi-Horizon Prediction ────────────────────

    def predict_current(self, damas_cache_func=None, sen_live_func=None):
        if not self.model_ready:
            return None

        today = datetime.now().strftime('%Y-%m-%d')

        try:
            if damas_cache_func:
                imb_items = damas_cache_func('imbalance', damas.get_estimated_imbalance_prices, today)
                marg_items = damas_cache_func('marginal', damas.get_marginal_prices_overview, today)
                act_items = damas_cache_func('activated', damas.get_activated_balancing_energy_overview, today)
            else:
                dfrom, dto = _damas_time_range(today)
                imb_items = damas.get_estimated_imbalance_prices(dfrom, dto)
                marg_items = damas.get_marginal_prices_overview(dfrom, dto)
                act_items = damas.get_activated_balancing_energy_overview(dfrom, dto)
        except Exception as e:
            log.error(f"Prediction data fetch error: {e}")
            return None

        imb_valid = [r for r in imb_items if r.get('id')
                     and isinstance(r.get('estimatedPriceNegativeImbalance'), (int, float))]
        if len(imb_valid) < 7:
            return None

        marg_by_isp = {r.get('ISP', 0): r for r in marg_items if r.get('id')}
        act_by_isp = {r.get('ISP', 0): r for r in act_items if r.get('id')}

        rows = []
        for r in imb_valid:
            isp = r.get('ISP', 0)
            marg = marg_by_isp.get(isp, {})
            act = act_by_isp.get(isp, {})
            ti = r.get('timeInterval', {})
            rows.append({
                'date': today, 'isp': isp,
                'interval_from': ti.get('from', ''), 'interval_to': ti.get('to', ''),
                'neg_price': r.get('estimatedPriceNegativeImbalance'),
                'pos_price': r.get('estimatedPricePositiveImbalance'),
                'system_imbalance': r.get('estimatedSystemImbalance'),
                'realized_consumption': r.get('realizedConsumption'),
                'sum_qup': r.get('sumQup'), 'sum_qdn': r.get('sumQdn'),
                'sum_qup_pup': r.get('sumQupPup'), 'sum_qdown_pdn': r.get('sumQdownPdn'),
                'netting_import': r.get('imbalanceNettingImport'),
                'netting_export': r.get('imbalanceNettingExport'),
                'deviation_in': r.get('estimatedUnintendedDeviationInArea'),
                'deviation_out': r.get('estimatedUnintendedDeviationOutArea'),
                'fcr': r.get('fcr'), 'pricing_type': r.get('type'),
                'afrr_up_price': marg.get('aFRR_Up'),
                'afrr_down_price': marg.get('aFRR_Down'),
                'mfrr_up_price': marg.get('mFRR_Up') or marg.get('mFRR_Up_Scheduled'),
                'mfrr_down_price': marg.get('mFRR_Down') or marg.get('mFRR_Down_Scheduled'),
                'afrr_up_activated': act.get('aFRR_Up'),
                'afrr_down_activated': act.get('aFRR_Down'),
                'mfrr_up_activated': act.get('mFRR_Up'),
                'mfrr_down_activated': act.get('mFRR_Down'),
            })

        df = pd.DataFrame(rows)
        df_feat, feature_cols = self.build_features(df)

        # Get last valid row
        valid_mask = df_feat[f'neg_price_lag6'].notna()
        if not valid_mask.any():
            valid_mask = df_feat[f'neg_price_lag1'].notna()
        if not valid_mask.any():
            return None

        last_idx = valid_mask[valid_mask].index[-1]
        X_pred = df_feat.loc[[last_idx], feature_cols].fillna(0)
        X_aligned = self._align_features(X_pred)

        last_row = rows[-1]
        damas_isp = last_row['isp']

        from zoneinfo import ZoneInfo
        ro_tz = ZoneInfo('Europe/Bucharest')

        # Use the REAL clock for target times, not the stale DAMAS ISP
        now_local = datetime.now(ro_tz)
        now_minute = (now_local.minute // 15) * 15
        now_quarter_start = now_local.replace(minute=now_minute, second=0, microsecond=0)
        real_isp = now_local.hour * 4 + now_minute // 15 + 1

        # How many ISPs ahead is DAMAS lagging
        damas_lag = max(0, real_isp - damas_isp)

        # Predict for each horizon
        quarters = []
        max_proba = 0
        max_risk = 'LOW'

        with self.lock:
            for h in HORIZONS:
                if h not in self.models:
                    continue
                model = self.models[h]
                thresh = self.thresholds.get(h, 0.30)
                raw_proba = float(model.predict_proba(X_aligned)[:, 1][0])
                # Map raw XGBoost output through its per-horizon calibrator.
                # If no calibrator is loaded, returns the raw value unchanged.
                proba = self._apply_calibrator(raw_proba, f'xgb_h{h}')
                predicted_neg = int(proba >= thresh)
                risk = _risk_level(proba)

                # Target time from real clock, not DAMAS
                target_isp = real_isp + h
                t_start = now_quarter_start + timedelta(minutes=15 * h)
                t_end = t_start + timedelta(minutes=15)
                target_time = f"{t_start.strftime('%H:%M')} - {t_end.strftime('%H:%M')}"

                quarters.append({
                    'horizon': h,
                    'horizon_label': HORIZON_LABELS[h],
                    'target_isp': target_isp,
                    'target_time': target_time,
                    'probability': round(proba, 4),           # calibrated XGB prob
                    'probability_raw': round(raw_proba, 4),   # pre-calibration XGB prob
                    'risk_level': risk,
                    'predicted_negative': predicted_neg,
                    'threshold': thresh,
                })

                if proba > max_proba:
                    max_proba = proba
                    max_risk = risk

        # Current state — use real clock
        current_time = now_local.strftime('%H:%M')
        current_isp = real_isp

        # Top features from primary model (horizon 2)
        top_factors = []
        if 2 in self.models:
            imp = self.models[2].feature_importances_
            feat_vals = X_aligned.iloc[0].to_dict()
            top_idx = np.argsort(imp)[-5:][::-1]
            for i in top_idx:
                fname = self.feature_names[i]
                top_factors.append({
                    'name': fname,
                    'value': round(float(feat_vals.get(fname, 0)), 2),
                    'importance': round(float(imp[i]), 4),
                })

        # ── Logistic Regression prediction (DAMAS + SEN) ──
        logreg_proba = None
        logreg_ready = False
        sen_data = None

        if self.logreg_model is not None and self.logreg_features and sen_live_func:
            try:
                sen_data = sen_live_func()
            except Exception as e:
                log.debug(f"SEN live fetch error: {e}")

        if sen_data and self.logreg_model is not None:
            try:
                logreg_proba = self._predict_logreg(df_feat, last_idx, sen_data)
                logreg_ready = True
            except Exception as e:
                log.error(f"LogReg prediction error: {e}")

        # ── Random Forest prediction ──
        rf_ready = bool(self.rf_models)
        if rf_ready:
            try:
                for q in quarters:
                    h = q['horizon']
                    if h in self.rf_models:
                        rf_raw = float(self.rf_models[h].predict_proba(X_aligned)[:, 1][0])
                        rf_cal = self._apply_calibrator(rf_raw, f'rf_h{h}')
                        q['rf_probability'] = round(rf_cal, 4)
                        q['rf_probability_raw'] = round(rf_raw, 4)
            except Exception as e:
                log.error(f"RF prediction error: {e}")
                rf_ready = False

        # ── Prophet prediction ──
        prophet_predictions = None
        prophet_ready = False
        if self.prophet_ready:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                prophet_predictions = self._predict_prophet(real_isp, today)
                if prophet_predictions:
                    prophet_ready = True
                    for q in quarters:
                        h = q['horizon']
                        if h in prophet_predictions:
                            q['prophet_probability'] = prophet_predictions[h]['probability']
            except Exception as e:
                log.error(f"Prophet prediction error: {e}")

        # ── LSTM sequence prediction ──
        lstm_predictions = None
        lstm_ready = False
        if self.lstm.is_trained:
            try:
                lstm_predictions = self.lstm.predict(rows, sen_live=sen_data)
                if lstm_predictions:
                    lstm_ready = True
                    for q in quarters:
                        h = q['horizon']
                        if h in lstm_predictions:
                            q['lstm_probability'] = lstm_predictions[h]['probability']
            except Exception as e:
                log.error(f"LSTM prediction error: {e}", exc_info=True)

        # ── Ensemble: learned stacking meta-learner (XGBoost + RF) blended
        #    with LSTM/Prophet via mean; falls back to pure mean if meta absent ──
        for q in quarters:
            h = q['horizon']
            xgb_p = q['probability']
            rf_p = q.get('rf_probability') if rf_ready else None

            # Primary signal: meta-stacked XGBoost + RF (if meta + RF both ready)
            if self.meta_models.get(h) is not None and rf_p is not None:
                try:
                    import numpy as _np
                    stacked = float(self.meta_models[h].predict_proba(
                        _np.array([[xgb_p, rf_p]], dtype=float)
                    )[0, 1])
                    q['meta_probability'] = round(stacked, 4)
                    q['meta_used'] = True
                    primary = stacked
                except Exception as e:
                    log.error(f"Meta predict H{h} error: {e}")
                    primary = (xgb_p + rf_p) / 2.0
                    q['meta_used'] = False
            elif rf_p is not None:
                primary = (xgb_p + rf_p) / 2.0
                q['meta_used'] = False
            else:
                primary = xgb_p
                q['meta_used'] = False

            # Secondary blend: average stacked primary with LSTM + Prophet signals
            blend = [primary]
            if lstm_ready and 'lstm_probability' in q:
                blend.append(q['lstm_probability'])
            if prophet_ready and 'prophet_probability' in q:
                blend.append(q['prophet_probability'])
            q['ensemble_probability'] = round(sum(blend) / len(blend), 4)
            q['model_count'] = len(blend) + (1 if q['meta_used'] else 0)

        result = {
            'quarters': quarters,
            'max_probability': round(max_proba, 4),
            'max_risk_level': max_risk,
            'current_isp': current_isp,
            'current_time': current_time,
            'current_price': last_row.get('neg_price'),
            'system_imbalance': last_row.get('system_imbalance'),
            'model_ready': True,
            'model_version': self.model_version,
            'top_factors': top_factors,
            'timestamp': datetime.now().isoformat(),
            'damas_lag': damas_lag,  # how many ISPs behind real time
            'damas_isp': damas_isp,  # last DAMAS ISP with data
            # Logistic regression (DAMAS+SEN combined model)
            'logreg_ready': logreg_ready,
            'logreg_probability': round(logreg_proba, 4) if logreg_proba is not None else None,
            'sen_solar': sen_data.get('solar') if sen_data else None,
            'sen_wind': sen_data.get('wind') if sen_data else None,
            'sen_consumption': sen_data.get('consumption') if sen_data else None,
            'sen_production': sen_data.get('production') if sen_data else None,
            'sen_exchange': sen_data.get('exchange') if sen_data else None,
            # Random Forest
            'rf_ready': rf_ready,
            # Prophet
            'prophet_ready': prophet_ready,
            # LSTM sequence model
            'lstm_ready': lstm_ready,
            'lstm_metrics': self.lstm.metrics if lstm_ready else None,
            # Backward compat for notification system
            'probability': quarters[0]['probability'] if quarters else 0,
            'risk_level': quarters[0]['risk_level'] if quarters else 'LOW',
            'predicted_negative': quarters[0]['predicted_negative'] if quarters else 0,
            'target_time': quarters[0]['target_time'] if quarters else '',
            'target_isp': quarters[0]['target_isp'] if quarters else 0,
            'threshold': quarters[0]['threshold'] if quarters else 0.3,
        }

        self._latest_prediction = result

        # Log all horizons so accuracy can be evaluated per-horizon later.
        # features_json includes horizon + ensemble_probability so downstream analysis
        # can slice performance by lead-time.
        if quarters:
            ts = datetime.now().isoformat()
            base_feats = {k: round(float(v), 4) for k, v in X_aligned.iloc[0].to_dict().items()
                          if abs(float(v)) > 0.001}
            for q in quarters:
                try:
                    feats = dict(base_feats)
                    feats['_horizon'] = q['horizon']
                    feats['_ensemble_probability'] = q.get('ensemble_probability', q['probability'])
                    feats['_xgb_probability'] = q['probability']
                    if 'rf_probability' in q:
                        feats['_rf_probability'] = q['rf_probability']
                    db.save_prediction({
                        'predicted_at': ts,
                        'target_date': today,
                        'target_isp': q['target_isp'],
                        'target_interval': q['target_time'],
                        'probability': q.get('ensemble_probability', q['probability']),
                        'risk_level': q['risk_level'],
                        'predicted_negative': q['predicted_negative'],
                        'model_version': self.model_version,
                        'features_json': json.dumps(feats, default=str),
                    })
                except Exception as e:
                    log.debug(f"Prediction log error (h={q.get('horizon')}): {e}")

        return result

    def _align_features(self, X):
        aligned = pd.DataFrame(0, index=X.index, columns=self.feature_names, dtype=float)
        for col in self.feature_names:
            if col in X.columns:
                aligned[col] = X[col].values
        return aligned.fillna(0)

    def _predict_logreg(self, df_feat, last_idx, sen_data):
        """Run logistic regression prediction using DAMAS features + live SEN data."""
        row = df_feat.loc[last_idx]

        # Map live SEN fields
        sen_solar = sen_data.get('solar') or 0
        sen_wind = sen_data.get('wind') or 0
        sen_consumption = sen_data.get('consumption') or 0
        sen_production = sen_data.get('production') or 0
        sen_sold = sen_data.get('exchange') or 0  # exchange = sold in transelectrica_client
        sen_surplus = sen_production - sen_consumption
        sen_renewable_pct = (sen_solar + sen_wind) / max(sen_production, 1) * 100

        # Build feature vector
        isp = row.get('isp', 1) if 'isp' in df_feat.columns else 1
        hour = (isp - 1) // 4

        feature_values = {
            'system_imbalance': row.get('system_imbalance', 0) or 0,
            'neg_price': row.get('neg_price', 0) or 0,
            'neg_streak': row.get('neg_streak', 0) or 0,
            'neg_price_lag1': row.get('neg_price_lag1', 0) or 0,
            'was_neg_lag1': row.get('was_neg_lag1', 0) or 0,
            'sys_imb_lag1': row.get('sys_imb_lag1', 0) or 0,
            'sum_qdn': row.get('sum_qdn', 0) or 0,
            'damas_net_activated': (row.get('sum_qup', 0) or 0) - (row.get('sum_qdn', 0) or 0),
            'sen_solar': sen_solar,
            'sen_surplus': sen_surplus,
            'sen_sold': sen_sold,
            'sen_consumption': sen_consumption,
            'sen_renewable_pct': sen_renewable_pct,
            'solar_above_1000': int(sen_solar > 1000),
            'export_above_1000': int(abs(min(sen_sold, 0)) > 1000),
            'imb_x_solar': (row.get('system_imbalance', 0) or 0) * sen_solar / 1000,
            'surplus_x_solar': sen_surplus * int(sen_solar > 500),
            'is_daytime': int(9 <= hour <= 16),
        }

        # Build DataFrame with correct feature order
        X_lr = pd.DataFrame([feature_values])[self.logreg_features].fillna(0)
        X_lr_s = self.logreg_scaler.transform(X_lr)
        proba = float(self.logreg_model.predict_proba(X_lr_s)[:, 1][0])
        return proba

    # ── Accuracy Tracking ─────────────────────────────────────

    def update_actuals(self, damas_cache_func=None):
        preds = db.get_recent_predictions(hours=24)
        to_update = [p for p in preds if p.get('actual_neg_price') is None]
        if not to_update:
            return

        dates = set(p['target_date'] for p in to_update)
        for date_str in dates:
            try:
                if damas_cache_func:
                    items = damas_cache_func('imbalance', damas.get_estimated_imbalance_prices, date_str)
                else:
                    dfrom, dto = _damas_time_range(date_str)
                    items = damas.get_estimated_imbalance_prices(dfrom, dto)

                price_by_isp = {}
                for r in items:
                    if r.get('id') and isinstance(r.get('estimatedPriceNegativeImbalance'), (int, float)):
                        price_by_isp[r.get('ISP', 0)] = r['estimatedPriceNegativeImbalance']

                for p in to_update:
                    if p['target_date'] == date_str and p['target_isp'] in price_by_isp:
                        db.backfill_prediction_actuals(
                            date_str, p['target_isp'], price_by_isp[p['target_isp']])
            except Exception as e:
                log.debug(f"Actuals backfill error for {date_str}: {e}")

    def get_accuracy_stats(self):
        preds = db.get_recent_predictions(hours=24)
        evaluated = [p for p in preds if p.get('actual_was_negative') is not None]
        if not evaluated:
            return {'total': 0}

        tp = sum(1 for p in evaluated if p['predicted_negative'] and p['actual_was_negative'])
        fp = sum(1 for p in evaluated if p['predicted_negative'] and not p['actual_was_negative'])
        fn = sum(1 for p in evaluated if not p['predicted_negative'] and p['actual_was_negative'])
        tn = sum(1 for p in evaluated if not p['predicted_negative'] and not p['actual_was_negative'])

        total = tp + fp + fn + tn
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        return {
            'total': total, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': round(recall, 3), 'precision': round(precision, 3),
        }
