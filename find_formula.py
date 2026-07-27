"""Find the best formula combining SEN + DAMAS data to predict negative prices."""
import requests
import pandas as pd
import numpy as np
import database as db
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
import time
warnings.filterwarnings('ignore')

# ── Step 1: Fetch SEN data ──
print("Fetching SEN historical data...")
url = 'https://www.transelectrica.ro/widget/web/tel/sen-grafic'
all_records = []
damas_dates = sorted(db.get_damas_dates())
start = datetime.strptime(damas_dates[0], '%Y-%m-%d')
end = datetime.strptime(damas_dates[-1], '%Y-%m-%d')

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
        print(f"  Error {d}: {e}")
    d = chunk_end + timedelta(days=1)
    time.sleep(0.3)

print(f"SEN records: {len(all_records)}")

# ── Step 2: Parse SEN ──
rows = []
for rec in all_records:
    fields = rec.split(';')
    if len(fields) < 12:
        continue
    try:
        ts = datetime.strptime(fields[0].strip(), '%d-%m-%Y %H:%M:%S')
        rows.append({
            'timestamp': ts,
            'date': ts.strftime('%Y-%m-%d'),
            'sen_consumption': float(fields[1]),
            'sen_production': float(fields[3]),
            'sen_sold': float(fields[4]),
            'sen_coal': float(fields[5]),
            'sen_gas': float(fields[6]),
            'sen_hydro': float(fields[7]),
            'sen_nuclear': float(fields[8]),
            'sen_wind': float(fields[9]),
            'sen_solar': float(fields[10]),
            'sen_biomass': float(fields[11]),
        })
    except:
        pass

sen_df = pd.DataFrame(rows)
sen_df['sen_surplus'] = sen_df['sen_production'] - sen_df['sen_consumption']
sen_df['sen_renewable'] = sen_df['sen_wind'] + sen_df['sen_solar']
sen_df['sen_renewable_pct'] = sen_df['sen_renewable'] / sen_df['sen_production'].clip(lower=1) * 100
sen_df['sen_export'] = sen_df['sen_sold'].clip(upper=0).abs()
sen_df['sen_import'] = sen_df['sen_sold'].clip(lower=0)
sen_df['isp_time'] = sen_df['timestamp'].dt.floor('15min')
sen_df['isp'] = sen_df['isp_time'].dt.hour * 4 + sen_df['isp_time'].dt.minute // 15 + 1

sen_isp = sen_df.groupby(['date', 'isp']).agg({
    'sen_consumption': 'mean', 'sen_production': 'mean', 'sen_sold': 'mean',
    'sen_coal': 'mean', 'sen_gas': 'mean', 'sen_hydro': 'mean',
    'sen_nuclear': 'mean', 'sen_wind': 'mean', 'sen_solar': 'mean',
    'sen_biomass': 'mean', 'sen_surplus': 'mean', 'sen_renewable': 'mean',
    'sen_renewable_pct': 'mean', 'sen_export': 'mean', 'sen_import': 'mean',
}).reset_index()

# ── Step 3: Get DAMAS & merge ──
damas = pd.DataFrame(db.get_damas_history())
merged = pd.merge(damas, sen_isp, on=['date', 'isp'], how='inner')
merged['is_negative'] = (merged['neg_price'] < 0).astype(int)
merged = merged.sort_values(['date', 'isp']).reset_index(drop=True)
print(f"Merged: {len(merged)} rows")

# ── Step 4: Engineer combined features ──
# DAMAS features
merged['damas_net_activated'] = merged['sum_qup'].fillna(0) - merged['sum_qdn'].fillna(0)
merged['damas_price_spread'] = merged['neg_price'].fillna(0) - merged['pos_price'].fillna(0)

# Combined features (SEN + DAMAS interaction)
merged['surplus_x_solar'] = merged['sen_surplus'] * (merged['sen_solar'] > 500).astype(int)
merged['export_x_surplus'] = merged['sen_export'] * (merged['sen_surplus'] > 0).astype(int)
merged['solar_above_1000'] = (merged['sen_solar'] > 1000).astype(int)
merged['solar_above_1500'] = (merged['sen_solar'] > 1500).astype(int)
merged['export_above_1000'] = (merged['sen_export'] > 1000).astype(int)
merged['high_renewable_pct'] = (merged['sen_renewable_pct'] > 30).astype(int)
merged['surplus_positive'] = (merged['sen_surplus'] > 0).astype(int)
merged['low_consumption'] = (merged['sen_consumption'] < 5500).astype(int)
merged['imb_x_solar'] = merged['system_imbalance'].fillna(0) * merged['sen_solar'] / 1000
merged['imb_x_export'] = merged['system_imbalance'].fillna(0) * merged['sen_export'] / 1000

# Time features
merged['hour'] = ((merged['isp'] - 1) // 4)
merged['is_daytime'] = ((merged['hour'] >= 9) & (merged['hour'] <= 16)).astype(int)
merged['is_weekend'] = (pd.to_datetime(merged['date']).dt.dayofweek >= 5).astype(int)

# Lags from DAMAS (previous quarters)
for lag in [1, 2, 4]:
    merged[f'neg_price_lag{lag}'] = merged['neg_price'].shift(lag)
    merged[f'sys_imb_lag{lag}'] = merged['system_imbalance'].shift(lag)
    merged[f'was_neg_lag{lag}'] = merged['is_negative'].shift(lag)

# Streak
streak = []
s = 0
for v in merged['is_negative']:
    if v == 1:
        s += 1
    else:
        s = 0
    streak.append(s)
merged['neg_streak'] = streak

# Drop NaN rows from lags
merged = merged.dropna(subset=['neg_price_lag4']).reset_index(drop=True)

# ── Step 5: Time-based train/test split ──
unique_dates = sorted(merged['date'].unique())
test_start = unique_dates[-7]
train = merged[merged['date'] < test_start]
test = merged[merged['date'] >= test_start]
print(f"Train: {len(train)} rows ({len(train[train['date'].isin(unique_dates[:-7])].date.unique())} days)")
print(f"Test:  {len(test)} rows ({len(test.date.unique())} days, {test_start} onwards)")

# ── Step 6: Try multiple feature sets ──
feature_sets = {
    'DAMAS only': [
        'system_imbalance', 'neg_price', 'pos_price', 'sum_qup', 'sum_qdn',
        'damas_net_activated', 'neg_streak', 'neg_price_lag1', 'neg_price_lag2',
        'was_neg_lag1', 'sys_imb_lag1', 'hour', 'is_daytime',
    ],
    'SEN only': [
        'sen_solar', 'sen_wind', 'sen_consumption', 'sen_production',
        'sen_surplus', 'sen_sold', 'sen_export', 'sen_renewable_pct',
        'sen_hydro', 'sen_nuclear', 'sen_coal', 'sen_gas',
        'solar_above_1000', 'export_above_1000', 'hour', 'is_daytime', 'is_weekend',
    ],
    'DAMAS + SEN combined': [
        'system_imbalance', 'neg_price', 'sum_qup', 'sum_qdn',
        'damas_net_activated', 'neg_streak',
        'neg_price_lag1', 'was_neg_lag1', 'sys_imb_lag1',
        'sen_solar', 'sen_wind', 'sen_surplus', 'sen_sold',
        'sen_consumption', 'sen_renewable_pct',
        'solar_above_1000', 'export_above_1000',
        'imb_x_solar', 'surplus_x_solar',
        'hour', 'is_daytime',
    ],
    'BEST (tuned)': [
        'system_imbalance', 'neg_price', 'neg_streak',
        'neg_price_lag1', 'was_neg_lag1', 'sys_imb_lag1',
        'sum_qdn', 'damas_net_activated',
        'sen_solar', 'sen_surplus', 'sen_sold',
        'sen_consumption', 'sen_renewable_pct',
        'solar_above_1000', 'solar_above_1500',
        'export_above_1000', 'imb_x_solar',
        'surplus_x_solar', 'low_consumption',
        'is_daytime',
    ],
}

print("\n" + "=" * 70)
print("COMPARING FEATURE SETS (test set = last 7 days)")
print("=" * 70)

best_name = None
best_f1 = 0
best_model = None
best_features = None
best_scaler = None

for name, features in feature_sets.items():
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    y_train = train['is_negative']
    y_test = test['is_negative']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, C=0.3)
    lr.fit(X_train_s, y_train)

    proba = lr.predict_proba(X_test_s)[:, 1]
    preds = (proba >= 0.35).astype(int)  # tuned threshold

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    print(f"\n  {name}:")
    print(f"    Accuracy={acc:.1%}  Precision={prec:.1%}  Recall={rec:.1%}  F1={f1:.1%}  AUC={auc:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_model = lr
        best_features = features
        best_scaler = scaler

# ── Step 7: Print the winning formula ──
print("\n" + "=" * 70)
print(f"WINNER: {best_name}")
print("=" * 70)

lr = best_model
features = best_features
scaler = best_scaler

print(f"\nIntercept: {lr.intercept_[0]:+.4f}")
print(f"\nWeights (by importance):")
weights = sorted(zip(features, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
for name, coef in weights:
    arrow = "^" if coef > 0 else "v"
    meaning = "-> MORE likely negative" if coef > 0 else "-> LESS likely negative"
    print(f"  {name:25s}: {coef:+.4f}  {meaning}")

# Print the raw formula (unscaled, usable with real MW values)
print("\n" + "=" * 70)
print("THE FORMULA (plug in real values)")
print("=" * 70)
print("\nP(negative) = sigmoid(")
intercept = lr.intercept_[0]
# Adjust intercept for unscaling
raw_intercept = intercept
for i, (fname, coef) in enumerate(zip(features, lr.coef_[0])):
    raw_intercept -= coef * scaler.mean_[i] / scaler.scale_[i]

print(f"    {raw_intercept:+.6f}  (base)")
for i, (fname, coef) in enumerate(zip(features, lr.coef_[0])):
    raw_coef = coef / scaler.scale_[i]
    if abs(raw_coef) > 0.000001:
        print(f"  + {raw_coef:+.6f} * {fname}")
print(")")
print("\nsigmoid(x) = 1 / (1 + e^(-x))")
print("Result: 0.0 = definitely positive, 1.0 = definitely negative")

# ── Step 8: Practical examples from test set ──
print("\n" + "=" * 70)
print("REAL EXAMPLES FROM TEST SET")
print("=" * 70)

X_test = test[features].fillna(0)
X_test_s = scaler.transform(X_test)
proba = lr.predict_proba(X_test_s)[:, 1]
test_with_proba = test.copy()
test_with_proba['predicted_prob'] = proba

# Show some correct predictions
print("\nCorrectly predicted NEGATIVE (true negatives caught):")
tp = test_with_proba[(test_with_proba['is_negative'] == 1) & (test_with_proba['predicted_prob'] >= 0.35)]
for _, row in tp.head(5).iterrows():
    print(f"  {row['date']} ISP {int(row['isp']):2d} | "
          f"prob={row['predicted_prob']:.0%} actual={row['neg_price']:.1f} EUR | "
          f"solar={row['sen_solar']:.0f} surplus={row['sen_surplus']:.0f} "
          f"imbal={row['system_imbalance']:.0f} sold={row['sen_sold']:.0f}")

print("\nCorrectly predicted POSITIVE (safe intervals):")
tn = test_with_proba[(test_with_proba['is_negative'] == 0) & (test_with_proba['predicted_prob'] < 0.35)]
for _, row in tn.head(5).iterrows():
    print(f"  {row['date']} ISP {int(row['isp']):2d} | "
          f"prob={row['predicted_prob']:.0%} actual={row['neg_price']:.1f} EUR | "
          f"solar={row['sen_solar']:.0f} surplus={row['sen_surplus']:.0f} "
          f"imbal={row['system_imbalance']:.0f} sold={row['sen_sold']:.0f}")

print("\nFALSE ALARMS (predicted negative but was positive):")
fp = test_with_proba[(test_with_proba['is_negative'] == 0) & (test_with_proba['predicted_prob'] >= 0.35)]
for _, row in fp.head(5).iterrows():
    print(f"  {row['date']} ISP {int(row['isp']):2d} | "
          f"prob={row['predicted_prob']:.0%} actual={row['neg_price']:.1f} EUR | "
          f"solar={row['sen_solar']:.0f} surplus={row['sen_surplus']:.0f} "
          f"imbal={row['system_imbalance']:.0f} sold={row['sen_sold']:.0f}")

print("\nMISSED (was negative but predicted positive):")
fn = test_with_proba[(test_with_proba['is_negative'] == 1) & (test_with_proba['predicted_prob'] < 0.35)]
for _, row in fn.head(5).iterrows():
    print(f"  {row['date']} ISP {int(row['isp']):2d} | "
          f"prob={row['predicted_prob']:.0%} actual={row['neg_price']:.1f} EUR | "
          f"solar={row['sen_solar']:.0f} surplus={row['sen_surplus']:.0f} "
          f"imbal={row['system_imbalance']:.0f} sold={row['sen_sold']:.0f}")
