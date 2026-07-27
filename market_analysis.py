"""
Market Analysis Engine — Historical pattern analysis + Rule-based risk assessment
for negative imbalance prices in the Romanian balancing market.

Provides SAFE / NORMAL / RISKY suggestions based on 61 days of DAMAS data analysis.
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import database as db

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  RULE-BASED RISK ENGINE  (derived from 61-day statistical analysis)
# ══════════════════════════════════════════════════════════════
#
# KEY FINDINGS FROM ARCHIVE (Feb 14 – Apr 15, 2026):
#
#  1. 98% of days have negative prices; avg 6.3 hours/day
#  2. Peak negative window: 09:00–16:00 (45–56% of intervals)
#  3. System imbalance is THE dominant signal:
#       Surplus >200 MW → 94.9% negative
#       Surplus 50–200  → 65.4% negative
#       Balanced ±50    → 15.4% negative
#       Deficit < -50   → 0% negative (never)
#  4. aFRR Down price < 0 → 90.6% negative
#  5. Combined: SysImb>100 + aFRR Dn<0 + daytime → 100% negative (n=266)
#  6. Net netting export > 10 MW → 42.8% negative
#  7. Net netting import > 10 MW → 7.0% negative
#  8. Transitions: 76% happen from prices in 0–500 EUR range
#  9. Negative streaks: median 2 ISPs (30 min), max 40 ISPs (10 hours)
# 10. Wednesday worst weekday (35.5%), Saturday best (16.7%)
#
# ══════════════════════════════════════════════════════════════

# Risk rules — evaluated in order, first match wins
# Each rule: (name, condition_func, level, confidence, explanation)

def _build_rules():
    """Build ordered rule list for risk assessment."""
    rules = [
        # ─── SAFE rules (green) ───────────────────────────────
        {
            'id': 'deficit_strong',
            'level': 'SAFE',
            'confidence': 99,
            'short': 'System in deficit',
            'detail': 'System imbalance < -50 MW. Historically 0% chance of negative prices when system is short.',
            'check': lambda d: d.get('system_imbalance') is not None and d['system_imbalance'] < -50,
        },
        {
            'id': 'afrr_high_positive',
            'level': 'SAFE',
            'confidence': 95,
            'short': 'aFRR Down price strongly positive',
            'detail': 'aFRR Down marginal > 100 EUR/MWh — only 4.7% negative historically. Market is buying downward regulation at premium.',
            'check': lambda d: d.get('afrr_down_price') is not None and d['afrr_down_price'] > 100,
        },
        {
            'id': 'net_import_strong',
            'level': 'SAFE',
            'confidence': 93,
            'short': 'Strong netting import',
            'detail': 'Net netting import > 10 MW — only 7% negative. System absorbing cross-border imbalance.',
            'check': lambda d: d.get('net_netting') is not None and d['net_netting'] > 10,
        },
        {
            'id': 'night_balanced',
            'level': 'SAFE',
            'confidence': 88,
            'short': 'Night + balanced system',
            'detail': 'Hours 21–05 with system imbalance ±50 MW. Low solar output, stable demand — ~15% risk.',
            'check': lambda d: (d.get('hour') is not None and (d['hour'] >= 21 or d['hour'] < 5)
                                and d.get('system_imbalance') is not None
                                and -50 <= d['system_imbalance'] <= 50),
        },

        # ─── RISKY rules (red) ───────────────────────────────
        {
            'id': 'triple_threat',
            'level': 'RISKY',
            'confidence': 100,
            'short': 'Surplus + aFRR negative + daytime',
            'detail': 'System surplus >100 MW + aFRR Down price < 0 + hours 09–16. 100% negative in 266 historical observations.',
            'check': lambda d: (d.get('system_imbalance') is not None and d['system_imbalance'] > 100
                                and d.get('afrr_down_price') is not None and d['afrr_down_price'] < 0
                                and d.get('hour') is not None and 9 <= d['hour'] <= 16),
        },
        {
            'id': 'surplus_afrr_neg',
            'level': 'RISKY',
            'confidence': 99,
            'short': 'Surplus + aFRR Down negative',
            'detail': 'System surplus >50 MW and aFRR Down < 0 EUR/MWh. 99.4% negative (n=872). Nearly certain loss.',
            'check': lambda d: (d.get('system_imbalance') is not None and d['system_imbalance'] > 50
                                and d.get('afrr_down_price') is not None and d['afrr_down_price'] < 0),
        },
        {
            'id': 'afrr_negative',
            'level': 'RISKY',
            'confidence': 91,
            'short': 'aFRR Down price negative',
            'detail': 'aFRR Down marginal < 0 EUR/MWh. 90.6% negative historically (n=1479). Strong sell signal.',
            'check': lambda d: d.get('afrr_down_price') is not None and d['afrr_down_price'] < 0,
        },
        {
            'id': 'big_surplus',
            'level': 'RISKY',
            'confidence': 95,
            'short': 'Large system surplus',
            'detail': 'System surplus > 200 MW. 94.9% negative. Massive excess generation.',
            'check': lambda d: d.get('system_imbalance') is not None and d['system_imbalance'] > 200,
        },
        {
            'id': 'surplus_daytime',
            'level': 'RISKY',
            'confidence': 78,
            'short': 'Daytime surplus',
            'detail': 'System surplus >50 MW during 09–16h. 77.9% negative. Solar peak + surplus = trouble.',
            'check': lambda d: (d.get('system_imbalance') is not None and d['system_imbalance'] > 50
                                and d.get('hour') is not None and 9 <= d['hour'] <= 16),
        },
        {
            'id': 'surplus_moderate',
            'level': 'RISKY',
            'confidence': 80,
            'short': 'Moderate system surplus',
            'detail': 'System surplus > 100 MW (any hour). 79.9% negative historically (n=507).',
            'check': lambda d: d.get('system_imbalance') is not None and d['system_imbalance'] > 100,
        },
        {
            'id': 'strong_export_daytime',
            'level': 'RISKY',
            'confidence': 65,
            'short': 'Strong export + daytime',
            'detail': 'Net netting export > 10 MW during 09–16h. Export flow + solar peak.',
            'check': lambda d: (d.get('net_netting') is not None and d['net_netting'] < -10
                                and d.get('hour') is not None and 9 <= d['hour'] <= 16),
        },

        # ─── NORMAL (yellow — default) ────────────────────────
        {
            'id': 'mild_surplus',
            'level': 'NORMAL',
            'confidence': 65,
            'short': 'Mild system surplus',
            'detail': 'System surplus 50–100 MW. 65% negative chance overall — depends on hour and reserves.',
            'check': lambda d: d.get('system_imbalance') is not None and 50 < d['system_imbalance'] <= 100,
        },
        {
            'id': 'daytime_window',
            'level': 'NORMAL',
            'confidence': 45,
            'short': 'Solar peak window (09–16h)',
            'detail': 'Daytime hours are the peak negative window (45–56% negative). Monitor closely.',
            'check': lambda d: d.get('hour') is not None and 9 <= d['hour'] <= 16,
        },
        {
            'id': 'default_balanced',
            'level': 'NORMAL',
            'confidence': 27,
            'short': 'Balanced market',
            'detail': 'No strong signals either way. Base rate: 26.5% of all intervals are negative.',
            'check': lambda d: True,  # fallback
        },
    ]
    return rules


RULES = _build_rules()


def assess_risk(current_data):
    """
    Evaluate current market conditions against rule engine.
    Returns list of matching rules (first = primary recommendation).
    Each: {level, confidence, short, detail, id}
    """
    # Pre-compute derived fields
    d = dict(current_data)
    if d.get('netting_import') is not None and d.get('netting_export') is not None:
        d['net_netting'] = (d.get('netting_import') or 0) - (d.get('netting_export') or 0)

    matches = []
    primary = None
    for rule in RULES:
        try:
            if rule['check'](d):
                entry = {k: rule[k] for k in ['id', 'level', 'confidence', 'short', 'detail']}
                if primary is None:
                    primary = entry
                matches.append(entry)
        except Exception:
            pass

    return {
        'primary': primary or {'level': 'NORMAL', 'confidence': 27, 'short': 'Unknown', 'detail': 'Insufficient data'},
        'all_matches': matches,
    }


# ══════════════════════════════════════════════════════════════
#  FULL ARCHIVE ANALYSIS  (computed once, cached)
# ══════════════════════════════════════════════════════════════

_analysis_cache = {'data': None, 'time': None}


def get_full_analysis(force_refresh=False):
    """Compute comprehensive analysis from archive. Cached for 1 hour."""
    now = datetime.now()
    if not force_refresh and _analysis_cache['data'] and _analysis_cache['time']:
        if (now - _analysis_cache['time']).total_seconds() < 3600:
            return _analysis_cache['data']

    rows = db.get_damas_history()
    if not rows:
        return {'error': 'No historical data available'}

    df = pd.DataFrame(rows)
    df['hour'] = ((df['isp'] - 1) // 4).astype(int)
    df['is_neg'] = (df['neg_price'] < 0).astype(int)
    df['dow'] = pd.to_datetime(df['date']).dt.dayofweek

    result = {
        'date_range': {'from': df['date'].min(), 'to': df['date'].max()},
        'total_isps': len(df),
        'total_days': int(df['date'].nunique()),
        'negative_isps': int(df['is_neg'].sum()),
        'negative_pct': round(df['is_neg'].mean() * 100, 1),
        'price_stats': {
            'mean': round(df['neg_price'].mean(), 2),
            'median': round(df['neg_price'].median(), 2),
            'min': round(df['neg_price'].min(), 2),
            'max': round(df['neg_price'].max(), 2),
            'std': round(df['neg_price'].std(), 2),
        },
    }

    # Hourly profile
    hourly = []
    for h in range(24):
        sub = df[df['hour'] == h]
        if len(sub) == 0:
            continue
        neg_pct = round((sub['neg_price'] < 0).mean() * 100, 1)
        avg_price = round(sub['neg_price'].mean(), 2)
        min_price = round(sub['neg_price'].min(), 2)
        avg_neg = round(sub.loc[sub['is_neg'] == 1, 'neg_price'].mean(), 2) if sub['is_neg'].sum() > 0 else None
        hourly.append({
            'hour': h, 'neg_pct': neg_pct, 'avg_price': avg_price,
            'min_price': min_price, 'avg_neg_price': avg_neg,
            'count': int(len(sub)), 'neg_count': int(sub['is_neg'].sum()),
        })
    result['hourly_profile'] = hourly

    # Day of week
    days_of_week = []
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for d in range(7):
        sub = df[df['dow'] == d]
        if len(sub) == 0:
            continue
        neg_pct = round(sub['is_neg'].mean() * 100, 1)
        avg_neg = round(sub.loc[sub['is_neg'] == 1, 'neg_price'].mean(), 2) if sub['is_neg'].sum() > 0 else None
        days_of_week.append({
            'dow': d, 'name': day_names[d], 'neg_pct': neg_pct,
            'avg_neg_price': avg_neg, 'neg_count': int(sub['is_neg'].sum()),
        })
    result['day_of_week'] = days_of_week

    # System imbalance buckets
    df_v = df.dropna(subset=['system_imbalance'])
    imb_buckets = []
    for lo, hi, label in [(-999, -200, 'Deficit < -200'), (-200, -50, 'Deficit -200 to -50'),
                           (-50, 0, 'Balanced -50 to 0'), (0, 50, 'Balanced 0 to 50'),
                           (50, 100, 'Surplus 50-100'), (100, 200, 'Surplus 100-200'),
                           (200, 999, 'Surplus > 200')]:
        sub = df_v[(df_v['system_imbalance'] >= lo) & (df_v['system_imbalance'] < hi)]
        if len(sub) == 0:
            continue
        imb_buckets.append({
            'label': label, 'neg_pct': round(sub['is_neg'].mean() * 100, 1),
            'count': int(len(sub)), 'avg_price': round(sub['neg_price'].mean(), 2),
        })
    result['system_imbalance_buckets'] = imb_buckets

    # Severity distribution
    negs = df[df['is_neg'] == 1]['neg_price']
    severity = []
    for lo, hi, label in [(-99999, -5000, 'Extreme (< -5000)'), (-5000, -1000, 'Severe (-5000 to -1000)'),
                           (-1000, -100, 'Moderate (-1000 to -100)'), (-100, 0, 'Mild (-100 to 0)')]:
        sub = negs[(negs >= lo) & (negs < hi)]
        severity.append({'label': label, 'count': int(len(sub)),
                         'pct': round(len(sub) / max(len(negs), 1) * 100, 1)})
    result['severity'] = severity

    # Streak analysis
    streaks = []
    cur = 0
    for v in df.sort_values(['date', 'isp'])['is_neg']:
        if v:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    s = pd.Series(streaks) if streaks else pd.Series([0])
    streak_dist = []
    for lo, hi, label in [(1, 1, '15 min'), (2, 4, '30-60 min'), (5, 8, '1-2 hours'),
                           (9, 16, '2-4 hours'), (17, 999, '4+ hours')]:
        count = int(((s >= lo) & (s <= hi)).sum())
        streak_dist.append({'label': label, 'count': count,
                            'pct': round(count / max(len(s), 1) * 100, 1)})
    result['streaks'] = {
        'total': int(len(s)),
        'mean_isps': round(float(s.mean()), 1),
        'mean_minutes': round(float(s.mean()) * 15, 0),
        'max_isps': int(s.max()),
        'max_hours': round(int(s.max()) * 15 / 60, 1),
        'distribution': streak_dist,
    }

    # Key risk factors
    result['risk_factors'] = [
        {'factor': 'System surplus > 100 MW', 'neg_pct': 79.9, 'n': 507},
        {'factor': 'aFRR Down price < 0', 'neg_pct': 90.6, 'n': 1479},
        {'factor': 'Surplus + aFRR neg + daytime', 'neg_pct': 100.0, 'n': 266},
        {'factor': 'Net netting export > 10 MW', 'neg_pct': 42.8, 'n': 1435},
        {'factor': 'Hours 09-16 + surplus > 50', 'neg_pct': 77.9, 'n': 810},
        {'factor': 'Weekend + surplus > 50', 'neg_pct': 66.7, 'n': 324},
    ]

    # Daily summary
    daily = df.groupby('date').agg(
        neg_isps=('is_neg', 'sum'),
        avg_price=('neg_price', 'mean'),
        min_price=('neg_price', 'min'),
    ).reset_index()
    daily['neg_hours'] = daily['neg_isps'] * 0.25
    result['daily_stats'] = {
        'days_with_negative': int((daily['neg_isps'] > 0).sum()),
        'days_with_4h_plus': int((daily['neg_hours'] > 4).sum()),
        'avg_neg_hours': round(daily['neg_hours'].mean(), 1),
        'max_neg_hours': round(daily['neg_hours'].max(), 1),
    }

    # Worst 10 days
    worst = daily.nlargest(10, 'neg_isps')
    result['worst_days'] = [
        {'date': r['date'], 'neg_hours': round(r['neg_hours'], 1),
         'avg_price': round(r['avg_price'], 2), 'min_price': round(r['min_price'], 2)}
        for _, r in worst.iterrows()
    ]

    # Hourly risk heatmap (hour x system_imbalance_bucket)
    heatmap = []
    for h in range(24):
        for lo, hi, label in [(0, 50, '0-50'), (50, 100, '50-100'), (100, 200, '100-200')]:
            sub = df_v[(df_v['hour'] == h) & (df_v['system_imbalance'] >= lo) & (df_v['system_imbalance'] < hi)]
            if len(sub) >= 5:
                heatmap.append({'hour': h, 'imbalance': label,
                                'neg_pct': round(sub['is_neg'].mean() * 100, 1), 'count': int(len(sub))})
    result['heatmap'] = heatmap

    _analysis_cache['data'] = result
    _analysis_cache['time'] = now
    return result


def get_current_suggestion(imbalance_data, marginal_data):
    """
    Given current DAMAS data, return SAFE/NORMAL/RISKY suggestion
    with explanation for the user.
    """
    if not imbalance_data:
        return None

    # Build data dict for rule engine
    d = {
        'system_imbalance': imbalance_data.get('estimatedSystemImbalance'),
        'neg_price': imbalance_data.get('estimatedPriceNegativeImbalance'),
        'pos_price': imbalance_data.get('estimatedPricePositiveImbalance'),
        'netting_import': imbalance_data.get('imbalanceNettingImport'),
        'netting_export': imbalance_data.get('imbalanceNettingExport'),
        'realized_consumption': imbalance_data.get('realizedConsumption'),
        'fcr': imbalance_data.get('fcr'),
        'sum_qup': imbalance_data.get('sumQup'),
        'sum_qdn': imbalance_data.get('sumQdn'),
        'afrr_down_price': marginal_data.get('aFRR_Down') if marginal_data else None,
        'afrr_up_price': marginal_data.get('aFRR_Up') if marginal_data else None,
    }

    # Get hour from ISP or current time
    isp = imbalance_data.get('ISP')
    if isp:
        d['hour'] = (isp - 1) // 4
    else:
        d['hour'] = datetime.now().hour

    d['dow'] = datetime.now().weekday()

    risk = assess_risk(d)

    # Build lookahead: what about the next 2 hours?
    hour = d.get('hour', 12)
    lookahead = []
    for offset_h in [1, 2]:
        future_h = (hour + offset_h) % 24
        # Use hourly base rates from analysis
        base_rates = {
            0: 20.5, 1: 16.8, 2: 13.5, 3: 15.6, 4: 11.1, 5: 9.4,
            6: 16.0, 7: 22.5, 8: 35.2, 9: 49.6, 10: 55.7, 11: 48.8,
            12: 48.0, 13: 46.3, 14: 44.3, 15: 42.2, 16: 28.9, 17: 25.0,
            18: 17.5, 19: 15.0, 20: 12.5, 21: 10.4, 22: 16.2, 23: 12.7,
        }
        base = base_rates.get(future_h, 26.5)
        # Adjust by current system state
        sys_imb = d.get('system_imbalance', 0) or 0
        if sys_imb > 100:
            base = min(base * 1.8, 100)
        elif sys_imb > 50:
            base = min(base * 1.4, 100)
        elif sys_imb < -50:
            base = base * 0.1

        level = 'SAFE' if base < 20 else ('RISKY' if base > 50 else 'NORMAL')
        lookahead.append({
            'hour': f'{future_h:02d}:00',
            'base_rate': round(base_rates.get(future_h, 26.5), 1),
            'adjusted_rate': round(base, 1),
            'level': level,
        })

    return {
        'risk': risk,
        'current': d,
        'lookahead': lookahead,
        'timestamp': datetime.now().isoformat(),
    }
