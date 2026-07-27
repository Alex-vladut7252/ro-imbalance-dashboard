"""
LSTM Negative Price Predictor

A recurrent neural network that learns temporal patterns in energy market data.
Unlike XGBoost (which sees each row independently) or LogReg (which uses a fixed formula),
the LSTM reads a SEQUENCE of past quarters and learns what patterns lead to negative prices.

Architecture:
    Input:  last 16 quarters (4 hours) of DAMAS + SEN features
    Model:  2-layer LSTM → dropout → fully connected → sigmoid
    Output: probability of negative price for next 1/2/3/4 quarters

The LSTM's hidden state acts as "memory" — it learns things like:
    - "prices have been dropping for 6 quarters AND solar is rising" → high negative risk
    - "system was in surplus but consumption is now climbing" → risk decreasing
    - "every day around 13:00 when solar peaks, prices dip" → time-of-day patterns
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

import database as db

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DEVICE = torch.device('cpu')

# How many past quarters the LSTM looks at
SEQ_LEN = 16  # 4 hours of context

# Prediction horizons (same as main predictor)
HORIZONS = [1, 2, 3, 4]

# Features the LSTM uses (raw values, no manual lag engineering needed)
LSTM_FEATURES = [
    # DAMAS core
    'neg_price', 'pos_price', 'system_imbalance',
    'sum_qup', 'sum_qdn', 'sum_qup_pup', 'sum_qdown_pdn',
    'netting_import', 'netting_export',
    # DAMAS marginal prices
    'afrr_up_price', 'afrr_down_price',
    'mfrr_up_price', 'mfrr_down_price',
    # DAMAS activated energy
    'afrr_up_activated', 'afrr_down_activated',
    'mfrr_up_activated', 'mfrr_down_activated',
    # SEN grid data
    'sen_solar', 'sen_wind', 'sen_consumption', 'sen_sold', 'sen_surplus',
    # Time encoding (cyclical)
    'hour_sin', 'hour_cos', 'is_weekend',
]


class PriceSequenceDataset(Dataset):
    """Converts tabular data into sequences for the LSTM."""

    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Sequence of past quarters
        x_seq = self.X[idx:idx + self.seq_len]
        # Target: will price be negative at each horizon?
        y_target = self.y[idx + self.seq_len]
        return x_seq, y_target


class NegPriceLSTM(nn.Module):
    """2-layer LSTM for multi-horizon negative price prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3,
                 num_horizons=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_horizons)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        out = self.fc(out)  # (batch, num_horizons)
        out = self.sigmoid(out)
        return out


class LSTMPredictor:
    """Manages LSTM training, saving, loading, and live prediction."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = LSTM_FEATURES
        self.thresholds = {}  # {horizon_idx: threshold}
        self.metrics = {}
        self.is_trained = False
        self._try_load()

    def _try_load(self):
        """Load saved LSTM model if available."""
        model_path = os.path.join(MODEL_DIR, 'lstm_model.pt')
        scaler_path = os.path.join(MODEL_DIR, 'lstm_scaler.joblib')
        meta_path = os.path.join(MODEL_DIR, 'lstm_meta.joblib')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, meta_path]):
            return

        try:
            meta = joblib.load(meta_path)
            self.scaler = joblib.load(scaler_path)
            self.thresholds = meta.get('thresholds', {})
            self.metrics = meta.get('metrics', {})
            self.feature_names = meta.get('features', LSTM_FEATURES)

            input_size = len(self.feature_names)
            self.model = NegPriceLSTM(input_size=input_size)
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                                   weights_only=True))
            self.model.eval()
            self.is_trained = True
            log.info(f"LSTM: Loaded model ({input_size} features, "
                     f"metrics: {self.metrics})")
        except Exception as e:
            log.error(f"LSTM: Failed to load model: {e}")

    def train(self):
        """Train the LSTM on historical DAMAS+SEN data."""
        log.info("LSTM: Starting training...")

        # Get data
        hist = db.get_damas_history()
        if len(hist) < 500:
            log.warning(f"LSTM: Not enough data ({len(hist)} rows, need 500+)")
            return False

        df = pd.DataFrame(hist)

        # Check SEN data availability
        has_sen = 'sen_solar' in df.columns and df['sen_solar'].notna().sum() > 200
        features_to_use = [f for f in self.feature_names
                           if f not in ('hour_sin', 'hour_cos', 'is_weekend',
                                        'sen_solar', 'sen_wind', 'sen_consumption',
                                        'sen_sold', 'sen_surplus')
                           or (f.startswith('sen_') and has_sen)
                           or not f.startswith('sen_')]

        # Engineer time features
        df['hour'] = ((df['isp'] - 1) // 4).astype(float)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_weekend'] = (pd.to_datetime(df['date']).dt.dayofweek >= 5).astype(float)

        # Ensure all feature columns exist
        for f in features_to_use:
            if f not in df.columns:
                df[f] = 0.0

        # Sort chronologically
        df = df.sort_values(['date', 'isp']).reset_index(drop=True)

        # Build targets: will neg_price be < 0 at horizon h?
        targets = np.zeros((len(df), len(HORIZONS)))
        for i, h in enumerate(HORIZONS):
            targets[:, i] = (df['neg_price'].shift(-h) < 0).fillna(0).astype(float).values

        # Get feature matrix
        X = df[features_to_use].fillna(0).values

        # Remove rows where we can't build full sequences or targets
        valid_end = len(X) - max(HORIZONS)
        X = X[:valid_end]
        targets = targets[:valid_end]

        if len(X) < SEQ_LEN + 100:
            log.warning(f"LSTM: Not enough valid rows ({len(X)})")
            return False

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Time-based split: last 7 days as test
        dates = df['date'].values[:valid_end]
        unique_dates = sorted(set(dates))
        test_start = unique_dates[-7] if len(unique_dates) >= 10 else unique_dates[-2]
        test_mask = dates >= test_start
        split_idx = np.where(test_mask)[0][0] if test_mask.any() else int(len(X) * 0.85)

        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]

        log.info(f"LSTM: Train {len(X_train)} rows, Test {len(X_test)} rows, "
                 f"{len(features_to_use)} features")

        # Create datasets
        train_ds = PriceSequenceDataset(X_train, y_train)
        test_ds = PriceSequenceDataset(X_test, y_test)

        if len(train_ds) < 50 or len(test_ds) < 20:
            log.warning("LSTM: Not enough sequences after windowing")
            return False

        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

        # Build model
        input_size = len(features_to_use)
        model = NegPriceLSTM(input_size=input_size).to(DEVICE)

        # Class weights for imbalanced data
        pos_count = y_train[SEQ_LEN:, 0].sum()
        neg_count = len(y_train) - SEQ_LEN - pos_count
        pos_weight = torch.FloatTensor([neg_count / max(pos_count, 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.expand(len(HORIZONS)))

        # We need raw logits for BCEWithLogitsLoss, so temporarily remove sigmoid
        model_for_training = NegPriceLSTM(input_size=input_size).to(DEVICE)
        model_for_training.load_state_dict(model.state_dict())
        # Replace the forward to output raw logits
        original_sigmoid = model_for_training.sigmoid
        model_for_training.sigmoid = nn.Identity()

        optimizer = torch.optim.Adam(model_for_training.parameters(), lr=0.001,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                                 factor=0.5)

        # Training loop
        best_loss = float('inf')
        patience = 0
        max_patience = 15

        for epoch in range(100):
            model_for_training.train()
            train_loss = 0
            for x_batch, y_batch in train_dl:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model_for_training(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_for_training.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model_for_training.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in test_dl:
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                    logits = model_for_training(x_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()

            avg_train = train_loss / len(train_dl)
            avg_val = val_loss / len(test_dl)
            scheduler.step(avg_val)

            if avg_val < best_loss:
                best_loss = avg_val
                patience = 0
                # Save best weights
                best_state = {k: v.clone() for k, v in
                              model_for_training.state_dict().items()}
            else:
                patience += 1

            if epoch % 10 == 0:
                log.info(f"  Epoch {epoch}: train_loss={avg_train:.4f} "
                         f"val_loss={avg_val:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

            if patience >= max_patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

        # Load best weights into the inference model (with sigmoid)
        model.load_state_dict(best_state)
        model.eval()

        # Evaluate on test set
        all_proba = []
        all_true = []
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                x_batch = x_batch.to(DEVICE)
                proba = model(x_batch)  # sigmoid applied
                all_proba.append(proba.numpy())
                all_true.append(y_batch.numpy())

        all_proba = np.concatenate(all_proba)
        all_true = np.concatenate(all_true)

        # Find optimal thresholds per horizon
        thresholds = {}
        metrics = {}
        for i, h in enumerate(HORIZONS):
            best_f1 = 0
            best_t = 0.35
            for t in np.arange(0.15, 0.75, 0.02):
                preds = (all_proba[:, i] >= t).astype(int)
                f1 = f1_score(all_true[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            preds = (all_proba[:, i] >= best_t).astype(int)
            acc = accuracy_score(all_true[:, i], preds)
            prec = precision_score(all_true[:, i], preds, zero_division=0)
            rec = recall_score(all_true[:, i], preds, zero_division=0)
            f1 = f1_score(all_true[:, i], preds, zero_division=0)

            thresholds[i] = float(best_t)
            metrics[h] = {
                'accuracy': round(acc, 4), 'precision': round(prec, 4),
                'recall': round(rec, 4), 'f1': round(f1, 4),
                'threshold': round(best_t, 4),
            }
            log.info(f"  LSTM H{h}: acc={acc:.1%} prec={prec:.1%} "
                     f"rec={rec:.1%} f1={f1:.1%} thresh={best_t:.2f}")

        # Save everything
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'lstm_model.pt'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'lstm_scaler.joblib'))
        joblib.dump({
            'features': features_to_use,
            'thresholds': thresholds,
            'metrics': metrics,
            'seq_len': SEQ_LEN,
            'horizons': HORIZONS,
        }, os.path.join(MODEL_DIR, 'lstm_meta.joblib'))

        self.model = model
        self.scaler = scaler
        self.thresholds = thresholds
        self.metrics = metrics
        self.feature_names = features_to_use
        self.is_trained = True

        log.info(f"LSTM: Training complete. Saved to {MODEL_DIR}")
        return True

    def predict(self, recent_rows, sen_live=None):
        """
        Predict negative price probability for next 4 quarters.

        Args:
            recent_rows: list of dicts from DAMAS API (at least SEQ_LEN rows)
            sen_live: dict from te_tracker.get_live() with current SEN data

        Returns:
            dict with probabilities per horizon, or None if not ready
        """
        if not self.is_trained or self.model is None:
            return None

        if len(recent_rows) < SEQ_LEN:
            return None

        # Take last SEQ_LEN rows
        df = pd.DataFrame(recent_rows[-SEQ_LEN - 4:])
        if len(df) < SEQ_LEN:
            return None

        # Engineer time features
        df['hour'] = ((df['isp'] - 1) // 4).astype(float)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_weekend'] = (pd.to_datetime(df['date']).dt.dayofweek >= 5).astype(float)

        # Ensure all features exist first (some rows may lack SEN columns)
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0.0

        # Fill SEN columns from live data if available
        if sen_live:
            sen_prod = sen_live.get('production') or 0
            sen_cons = sen_live.get('consumption') or 0
            df['sen_solar'] = df['sen_solar'].fillna(sen_live.get('solar') or 0)
            df['sen_wind'] = df['sen_wind'].fillna(sen_live.get('wind') or 0)
            df['sen_consumption'] = df['sen_consumption'].fillna(sen_cons)
            df['sen_sold'] = df['sen_sold'].fillna(sen_live.get('exchange') or 0)
            df['sen_surplus'] = df['sen_surplus'].fillna(sen_prod - sen_cons)

        # Get feature matrix and scale
        X = df[self.feature_names].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # Take the last SEQ_LEN rows as input sequence
        seq = X_scaled[-SEQ_LEN:]
        x_tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)  # (1, seq_len, features)

        # Predict
        self.model.eval()
        with torch.no_grad():
            proba = self.model(x_tensor).squeeze(0).numpy()  # (num_horizons,)

        results = {}
        for i, h in enumerate(HORIZONS):
            thresh = self.thresholds.get(i, 0.35)
            prob = float(proba[i])
            results[h] = {
                'probability': round(prob, 4),
                'predicted_negative': int(prob >= thresh),
                'threshold': thresh,
            }

        return results
