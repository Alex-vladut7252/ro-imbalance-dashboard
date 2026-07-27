"""Fetch SEN historical data and correlate with DAMAS imbalance prices."""
import requests
import pandas as pd
import numpy as np
import database as db
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Step 1: Fetch SEN data for all 61 days in chunks
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
        print(f"  {d.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}: {len(records)} records")
    except Exception as e:
        print(f"  Error {d}: {e}")
    d = chunk_end + timedelta(days=1)
    time.sleep(0.5)

print(f"\nTotal SEN records: {len(all_records)}")

# Step 2: Parse into DataFrame
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
            'hour': ts.hour,
            'minute': ts.minute,
            'consumption': float(fields[1]),
            'hourly_avg_cons': float(fields[2]),
            'production': float(fields[3]),
            'sold': float(fields[4]),  # negative = export
            'coal': float(fields[5]),
            'gas': float(fields[6]),
            'hydro': float(fields[7]),
            'nuclear': float(fields[8]),
            'wind': float(fields[9]),
            'solar': float(fields[10]),
            'biomass': float(fields[11]),
        })
    except:
        pass

sen_df = pd.DataFrame(rows)
print(f"Parsed SEN rows: {len(sen_df)}")

# Derive key features
sen_df['surplus'] = sen_df['production'] - sen_df['consumption']
sen_df['renewable'] = sen_df['wind'] + sen_df['solar']
sen_df['renewable_pct'] = sen_df['renewable'] / sen_df['production'] * 100
sen_df['is_exporting'] = (sen_df['sold'] < 0).astype(int)
sen_df['export_mw'] = sen_df['sold'].clip(upper=0).abs()
sen_df['import_mw'] = sen_df['sold'].clip(lower=0)

# Step 3: Map to 15-min ISP and merge with DAMAS
sen_df['isp_time'] = sen_df['timestamp'].dt.floor('15min')
sen_df['isp'] = sen_df['isp_time'].dt.hour * 4 + sen_df['isp_time'].dt.minute // 15 + 1

sen_isp = sen_df.groupby(['date', 'isp']).agg({
    'consumption': 'mean', 'production': 'mean', 'sold': 'mean',
    'coal': 'mean', 'gas': 'mean', 'hydro': 'mean', 'nuclear': 'mean',
    'wind': 'mean', 'solar': 'mean', 'biomass': 'mean',
    'surplus': 'mean', 'renewable': 'mean', 'renewable_pct': 'mean',
    'is_exporting': 'max', 'export_mw': 'mean', 'import_mw': 'mean',
}).reset_index()

# Get DAMAS data
damas = pd.DataFrame(db.get_damas_history())
print(f"DAMAS rows: {len(damas)}")

# Merge
merged = pd.merge(damas, sen_isp, on=['date', 'isp'], how='inner', suffixes=('_damas', '_sen'))
merged['is_negative'] = (merged['neg_price'] < 0).astype(int)
print(f"Merged rows: {len(merged)}")

# Step 4: Correlation analysis
print("\n" + "=" * 60)
print("CORRELATION WITH NEGATIVE PRICE")
print("=" * 60)

corr_cols = ['consumption', 'production', 'sold', 'surplus', 'coal', 'gas',
             'hydro', 'nuclear', 'wind', 'solar', 'biomass', 'renewable',
             'renewable_pct', 'export_mw', 'import_mw', 'system_imbalance']

for col in corr_cols:
    if col in merged.columns:
        corr = merged[col].corr(merged['is_negative'])
        corr_price = merged[col].corr(merged['neg_price'])
        print(f"  {col:20s}: corr_with_negative={corr:+.3f}  corr_with_price={corr_price:+.3f}")

# Step 5: When exporting vs importing
print("\n" + "=" * 60)
print("IMPORT vs EXPORT vs NEGATIVE PRICES")
print("=" * 60)

exporting = merged[merged['sold'] < 0]
importing = merged[merged['sold'] > 0]

print(f"When EXPORTING (Romania produces MORE than it needs):")
print(f"  {len(exporting)} intervals, {exporting['is_negative'].mean():.1%} have negative prices")
print(f"  Avg price: {exporting['neg_price'].mean():.1f} EUR, Avg export: {exporting['sold'].mean():.0f} MW")

print(f"\nWhen IMPORTING (Romania needs MORE power):")
print(f"  {len(importing)} intervals, {importing['is_negative'].mean():.1%} have negative prices")
print(f"  Avg price: {importing['neg_price'].mean():.1f} EUR, Avg import: {importing['sold'].mean():.0f} MW")

# Step 6: Bucket analysis
print("\n" + "=" * 60)
print("NEGATIVE PRICE RATE BY EXPORT/IMPORT LEVEL")
print("=" * 60)
bins = [-5000, -2000, -1000, -500, 0, 500, 1000, 5000]
labels = ['Export>2000', 'Export 1-2k', 'Export 0.5-1k', 'Export 0-500', 'Import 0-500', 'Import 0.5-1k', 'Import>1000']
merged['sold_bucket'] = pd.cut(merged['sold'], bins=bins, labels=labels)
bucket_stats = merged.groupby('sold_bucket', observed=True).agg(
    count=('is_negative', 'count'),
    neg_rate=('is_negative', 'mean'),
    avg_price=('neg_price', 'mean'),
).reset_index()
for _, row in bucket_stats.iterrows():
    bar = '#' * int(row['neg_rate'] * 50)
    print(f"  {row['sold_bucket']:18s}: {row['neg_rate']:5.1%} negative  avg={row['avg_price']:7.1f} EUR  n={int(row['count']):4d}  {bar}")

# Step 7: Generation mix when prices go negative
print("\n" + "=" * 60)
print("GENERATION MIX: NEGATIVE vs POSITIVE PRICES")
print("=" * 60)
neg = merged[merged['is_negative'] == 1]
pos = merged[merged['is_negative'] == 0]
gen_cols = ['coal', 'gas', 'hydro', 'nuclear', 'wind', 'solar', 'biomass']
print(f"{'Source':12s} {'When Negative':>14s} {'When Positive':>14s} {'Difference':>12s}")
for col in gen_cols:
    neg_avg = neg[col].mean()
    pos_avg = pos[col].mean()
    diff = neg_avg - pos_avg
    print(f"  {col:10s}  {neg_avg:10.0f} MW   {pos_avg:10.0f} MW   {diff:+8.0f} MW")

total_neg = neg['production'].mean()
total_pos = pos['production'].mean()
cons_neg = neg['consumption'].mean()
cons_pos = pos['consumption'].mean()
print(f"\n  {'PRODUCTION':10s}  {total_neg:10.0f} MW   {total_pos:10.0f} MW   {total_neg - total_pos:+8.0f} MW")
print(f"  {'CONSUMPTION':10s}  {cons_neg:10.0f} MW   {cons_pos:10.0f} MW   {cons_neg - cons_pos:+8.0f} MW")
print(f"  {'SURPLUS':10s}  {neg['surplus'].mean():10.0f} MW   {pos['surplus'].mean():10.0f} MW   {neg['surplus'].mean() - pos['surplus'].mean():+8.0f} MW")

# Step 8: Solar vs negative prices
print("\n" + "=" * 60)
print("SOLAR GENERATION vs NEGATIVE PRICES")
print("=" * 60)
solar_bins = [-100, 0, 500, 1000, 1500, 2000, 5000]
solar_labels = ['No solar', '0-500', '500-1000', '1000-1500', '1500-2000', '2000+']
merged['solar_bucket'] = pd.cut(merged['solar'], bins=solar_bins, labels=solar_labels)
solar_stats = merged.groupby('solar_bucket', observed=True).agg(
    count=('is_negative', 'count'),
    neg_rate=('is_negative', 'mean'),
    avg_surplus=('surplus', 'mean'),
).reset_index()
for _, row in solar_stats.iterrows():
    bar = '#' * int(row['neg_rate'] * 50)
    print(f"  Solar {row['solar_bucket']:10s}: {row['neg_rate']:5.1%} negative  surplus={row['avg_surplus']:+7.0f} MW  n={int(row['count']):4d}  {bar}")

# Step 9: Wind vs negative prices
print("\n" + "=" * 60)
print("WIND GENERATION vs NEGATIVE PRICES")
print("=" * 60)
wind_bins = [-100, 200, 500, 1000, 1500, 2000, 5000]
wind_labels = ['<200', '200-500', '500-1000', '1000-1500', '1500-2000', '2000+']
merged['wind_bucket'] = pd.cut(merged['wind'], bins=wind_bins, labels=wind_labels)
wind_stats = merged.groupby('wind_bucket', observed=True).agg(
    count=('is_negative', 'count'),
    neg_rate=('is_negative', 'mean'),
).reset_index()
for _, row in wind_stats.iterrows():
    bar = '#' * int(row['neg_rate'] * 50)
    print(f"  Wind {row['wind_bucket']:10s}: {row['neg_rate']:5.1%} negative  n={int(row['count']):4d}  {bar}")

# Step 10: Logistic regression — actual formula
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION — THE FORMULA")
print("=" * 60)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

formula_features = ['sold', 'surplus', 'solar', 'wind', 'consumption', 'production',
                    'renewable_pct', 'system_imbalance', 'nuclear', 'hydro']
# Only rows with all data
valid = merged.dropna(subset=formula_features + ['is_negative'])
X = valid[formula_features].fillna(0)
y = valid['is_negative']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, C=0.5)
lr.fit(X_scaled, y)

preds = lr.predict(X_scaled)
print(f"Accuracy: {accuracy_score(y, preds):.1%}")
print(f"Precision: {precision_score(y, preds):.1%}")
print(f"Recall: {recall_score(y, preds):.1%}")
print(f"F1: {f1_score(y, preds):.1%}")

print(f"\nIntercept: {lr.intercept_[0]:+.4f}")
print(f"\nFeature weights (standardized):")
for name, coef in sorted(zip(formula_features, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    direction = "more negative prices" if coef > 0 else "less negative prices"
    print(f"  {name:20s}: {coef:+.4f}  ({direction})")

print("\n\nRAW FORMULA (use actual MW values):")
print("P(negative) = sigmoid(")
print(f"    {lr.intercept_[0]:+.4f}")
for name, coef, mean, std in zip(formula_features, lr.coef_[0], scaler.mean_, scaler.scale_):
    raw_coef = coef / std
    raw_intercept_adj = -coef * mean / std
    print(f"    {raw_coef:+.6f} * {name}")
print(")")
print("\nwhere sigmoid(x) = 1 / (1 + e^(-x))")
