"""How important is import/export for predicting negative prices?"""
import pandas as pd
import numpy as np
import database as db
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings, time
warnings.filterwarnings('ignore')

# Fetch SEN
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
    except:
        pass
    d = chunk_end + timedelta(days=1)
    time.sleep(0.3)

rows = []
for rec in all_records:
    fields = rec.split(';')
    if len(fields) < 12:
        continue
    try:
        ts = datetime.strptime(fields[0].strip(), '%d-%m-%Y %H:%M:%S')
        rows.append({
            'timestamp': ts, 'date': ts.strftime('%Y-%m-%d'),
            'sen_sold': float(fields[4]), 'sen_solar': float(fields[10]),
            'sen_consumption': float(fields[1]), 'sen_production': float(fields[3]),
            'sen_wind': float(fields[9]), 'sen_hydro': float(fields[7]),
            'sen_nuclear': float(fields[8]), 'sen_coal': float(fields[5]),
            'sen_gas': float(fields[6]),
        })
    except:
        pass

sen_df = pd.DataFrame(rows)
sen_df['sen_surplus'] = sen_df['sen_production'] - sen_df['sen_consumption']
sen_df['sen_export'] = sen_df['sen_sold'].clip(upper=0).abs()
sen_df['sen_renewable_pct'] = (sen_df['sen_wind'] + sen_df['sen_solar']) / sen_df['sen_production'].clip(lower=1) * 100
sen_df['isp'] = sen_df['timestamp'].dt.floor('15min').dt.hour * 4 + sen_df['timestamp'].dt.floor('15min').dt.minute // 15 + 1
sen_isp = sen_df.groupby(['date', 'isp']).mean(numeric_only=True).reset_index()

damas = pd.DataFrame(db.get_damas_history())
m = pd.merge(damas, sen_isp, on=['date', 'isp'], how='inner').sort_values(['date', 'isp']).reset_index(drop=True)
m['is_negative'] = (m['neg_price'] < 0).astype(int)
m['net_activated'] = m['sum_qup'].fillna(0) - m['sum_qdn'].fillna(0)
m['solar_above_1000'] = (m['sen_solar'] > 1000).astype(int)
m['export_above_1000'] = (m['sen_export'] > 1000).astype(int)
m['imb_x_solar'] = m['system_imbalance'].fillna(0) * m['sen_solar'] / 1000
m['surplus_x_solar'] = m['sen_surplus'] * (m['sen_solar'] > 500).astype(int)
m['hour'] = (m['isp'] - 1) // 4
m['is_daytime'] = ((m['hour'] >= 9) & (m['hour'] <= 16)).astype(int)
streak = []
s = 0
for v in m['is_negative']:
    s = s + 1 if v == 1 else 0
    streak.append(s)
m['neg_streak'] = streak
m['neg_price_lag1'] = m['neg_price'].shift(1)
m['sys_imb_lag1'] = m['system_imbalance'].shift(1)
m['was_neg_lag1'] = m['is_negative'].shift(1)
m = m.dropna(subset=['neg_price_lag1']).reset_index(drop=True)

unique_dates = sorted(m['date'].unique())
test_start = unique_dates[-7]
train = m[m['date'] < test_start]
test = m[m['date'] >= test_start]

# ── Compare models ──
base = ['system_imbalance', 'neg_price', 'sum_qdn', 'net_activated',
        'neg_streak', 'neg_price_lag1', 'was_neg_lag1', 'sys_imb_lag1', 'hour', 'is_daytime']

sen_no_sold = base + ['sen_solar', 'sen_surplus', 'sen_consumption',
                       'sen_renewable_pct', 'solar_above_1000', 'imb_x_solar', 'surplus_x_solar']

sen_with_sold = sen_no_sold + ['sen_sold', 'sen_export', 'export_above_1000']

sets = {
    'DAMAS ONLY (no SEN data)': base,
    'DAMAS + SEN WITHOUT import/export': sen_no_sold,
    'DAMAS + SEN WITH import/export': sen_with_sold,
    'ONLY import/export (nothing else)': ['sen_sold', 'sen_export', 'export_above_1000', 'hour', 'is_daytime'],
}

print('=' * 70)
print('HOW IMPORTANT IS IMPORT/EXPORT?')
print('=' * 70)

for name, features in sets.items():
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[features].fillna(0))
    X_te = scaler.transform(test[features].fillna(0))
    lr = LogisticRegression(max_iter=2000, C=0.3)
    lr.fit(X_tr, train['is_negative'])
    proba = lr.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.35).astype(int)
    y = test['is_negative']
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, proba)
    print(f'\n  {name}:')
    print(f'    Acc={acc:.1%}  Prec={prec:.1%}  Rec={rec:.1%}  F1={f1:.1%}  AUC={auc:.3f}')

# ── Direct stats ──
print('\n' + '=' * 70)
print('IMPORT/EXPORT — RAW NUMBERS')
print('=' * 70)

total = len(m)
neg_total = m['is_negative'].sum()
print(f'Total: {total} intervals, {neg_total} negative ({neg_total/total:.1%})')

for label, cond in [
    ('Heavy EXPORT  (sold < -1500 MW)', m['sen_sold'] < -1500),
    ('Export         (sold -500 to -1500)', (m['sen_sold'] >= -1500) & (m['sen_sold'] < -500)),
    ('Mild export    (sold -500 to 0)', (m['sen_sold'] >= -500) & (m['sen_sold'] < 0)),
    ('Mild import    (sold 0 to +500)', (m['sen_sold'] >= 0) & (m['sen_sold'] < 500)),
    ('Import         (sold +500 to +1000)', (m['sen_sold'] >= 500) & (m['sen_sold'] < 1000)),
    ('Heavy IMPORT   (sold > +1000 MW)', m['sen_sold'] >= 1000),
]:
    subset = m[cond]
    if len(subset) == 0:
        continue
    neg_n = subset['is_negative'].sum()
    neg_pct = subset['is_negative'].mean()
    avg_p = subset['neg_price'].mean()
    bar = '#' * int(neg_pct * 40)
    print(f'  {label:40s}  n={len(subset):4d}  neg={neg_pct:5.1%}  avg_price={avg_p:+8.1f} EUR  {bar}')

# Extreme combo
print('\n' + '=' * 70)
print('DEADLY COMBOS')
print('=' * 70)

combos = [
    ('Export>1000 + Solar>1000', (m['sen_export'] > 1000) & (m['sen_solar'] > 1000)),
    ('Export>1000 + Solar>1500', (m['sen_export'] > 1000) & (m['sen_solar'] > 1500)),
    ('Export>1500 + Surplus>500', (m['sen_export'] > 1500) & (m['sen_surplus'] > 500)),
    ('Export>1000 + DAMAS imbal>50', (m['sen_export'] > 1000) & (m['system_imbalance'] > 50)),
    ('Import>500 + DAMAS imbal<-50', (m['sen_sold'] > 500) & (m['system_imbalance'] < -50)),
    ('Export>500 + low consumption<5000', (m['sen_export'] > 500) & (m['sen_consumption'] < 5000)),
]

for label, cond in combos:
    subset = m[cond]
    if len(subset) == 0:
        continue
    neg_pct = subset['is_negative'].mean()
    bar = '#' * int(neg_pct * 40)
    print(f'  {label:45s}  n={len(subset):4d}  neg={neg_pct:5.1%}  {bar}')

# Correlation ranking
print('\n' + '=' * 70)
print('FULL CORRELATION RANKING')
print('=' * 70)
all_cols = ['system_imbalance', 'sen_solar', 'sen_sold', 'sen_export', 'sen_surplus',
            'sen_consumption', 'sen_wind', 'sen_renewable_pct', 'sen_production',
            'sen_hydro', 'sen_nuclear', 'sen_coal', 'sen_gas',
            'sum_qdn', 'sum_qup', 'neg_streak', 'neg_price']
corrs = []
for col in all_cols:
    if col in m.columns:
        c = m[col].corr(m['is_negative'])
        corrs.append((col, c))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for col, c in corrs:
    bar = '#' * int(abs(c) * 50)
    sign = '+' if c > 0 else '-'
    print(f'  {col:25s}: {sign}{abs(c):.3f}  {bar}')
