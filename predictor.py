"""
Energy prediction engine.
Uses historical patterns from ENTSO-E data to forecast load, prices, and generation.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import database as db

log = logging.getLogger(__name__)


def predict_load(target_date):
    """
    Predict hourly load for target_date using historical patterns.
    Uses similar-day analysis: same weekday from recent weeks + trend.
    """
    target = datetime.strptime(target_date, '%Y-%m-%d')
    weekday = target.weekday()

    # Gather same-weekday data from last 8 weeks
    similar_days = []
    for weeks_back in range(1, 9):
        d = target - timedelta(weeks=weeks_back)
        load = db.get_actual_load(d.strftime('%Y-%m-%d'))
        if load and len(load) >= 20:
            similar_days.append(load)

    if not similar_days:
        # Try any recent data
        end = target - timedelta(days=1)
        start = end - timedelta(days=14)
        all_load = db.get_actual_load_range(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        if all_load:
            by_date = defaultdict(list)
            for r in all_load:
                by_date[r['date']].append(r)
            similar_days = [v for v in by_date.values() if len(v) >= 20]

    if not similar_days:
        return None

    # Build hourly averages with trend weighting (more recent = higher weight)
    hourly = defaultdict(list)
    weights = []
    for i, day_data in enumerate(similar_days):
        w = 1.0 / (i + 1)  # More recent days get higher weight
        weights.append(w)
        for row in day_data:
            hourly[row['hour']].append((row['load_mw'], w))

    prediction = []
    for h in range(24):
        if h in hourly and hourly[h]:
            vals = hourly[h]
            weighted_sum = sum(v * w for v, w in vals)
            weight_total = sum(w for _, w in vals)
            avg = weighted_sum / weight_total
            # Standard deviation for confidence interval
            raw_vals = [v for v, _ in vals]
            std = float(np.std(raw_vals)) if len(raw_vals) > 1 else avg * 0.05
            prediction.append({
                'hour': h,
                'predicted_mw': round(avg, 1),
                'confidence_low': round(avg - 1.96 * std, 1),
                'confidence_high': round(avg + 1.96 * std, 1),
                'std': round(std, 1),
                'samples': len(vals),
            })
        else:
            prediction.append({
                'hour': h,
                'predicted_mw': None,
                'confidence_low': None,
                'confidence_high': None,
                'std': None,
                'samples': 0,
            })

    return prediction


def predict_prices(target_date):
    """
    Predict DAM prices for target_date using historical patterns.
    Combines similar-day analysis with load-price correlation.
    """
    target = datetime.strptime(target_date, '%Y-%m-%d')
    weekday = target.weekday()

    # Gather same-weekday price data from recent weeks
    similar_days = []
    for weeks_back in range(1, 9):
        d = target - timedelta(weeks=weeks_back)
        prices = db.get_dam_prices(d.strftime('%Y-%m-%d'))
        if prices and len(prices) >= 20:
            similar_days.append(prices)

    if not similar_days:
        end = target - timedelta(days=1)
        start = end - timedelta(days=14)
        all_prices = db.get_dam_prices_range(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        if all_prices:
            by_date = defaultdict(list)
            for r in all_prices:
                by_date[r['date']].append(r)
            similar_days = [v for v in by_date.values() if len(v) >= 20]

    if not similar_days:
        return None

    hourly = defaultdict(list)
    for i, day_data in enumerate(similar_days):
        w = 1.0 / (i + 1)
        for row in day_data:
            hourly[row['hour']].append((row['price_eur'], w))

    prediction = []
    for h in range(24):
        if h in hourly and hourly[h]:
            vals = hourly[h]
            weighted_sum = sum(v * w for v, w in vals)
            weight_total = sum(w for _, w in vals)
            avg = weighted_sum / weight_total
            raw_vals = [v for v, _ in vals]
            std = float(np.std(raw_vals)) if len(raw_vals) > 1 else abs(avg) * 0.1
            prediction.append({
                'hour': h,
                'predicted_eur': round(avg, 2),
                'confidence_low': round(avg - 1.96 * std, 2),
                'confidence_high': round(avg + 1.96 * std, 2),
                'std': round(std, 2),
                'samples': len(vals),
            })
        else:
            prediction.append({
                'hour': h,
                'predicted_eur': None,
                'confidence_low': None,
                'confidence_high': None,
                'std': None,
                'samples': 0,
            })

    return prediction


def predict_generation_mix(target_date):
    """Predict generation mix for target_date."""
    target = datetime.strptime(target_date, '%Y-%m-%d')

    similar_days = []
    for weeks_back in range(1, 5):
        d = target - timedelta(weeks=weeks_back)
        gen = db.get_generation(d.strftime('%Y-%m-%d'))
        if gen:
            similar_days.append(gen)

    if not similar_days:
        return None

    # Aggregate by psr_type
    by_type = defaultdict(list)
    for day_data in similar_days:
        daily = defaultdict(float)
        counts = defaultdict(int)
        for row in day_data:
            key = row['psr_name'] or row['psr_type']
            daily[key] += row['value_mw'] or 0
            counts[key] += 1
        for key in daily:
            if counts[key] > 0:
                by_type[key].append(daily[key] / counts[key])

    result = {}
    for psr, vals in by_type.items():
        avg = sum(vals) / len(vals)
        result[psr] = round(avg, 1)

    return result


def get_forecast_accuracy(date_str):
    """Compare forecast vs actual for a given date."""
    forecast = db.get_load_forecast(date_str)
    actual = db.get_actual_load(date_str)

    if not forecast or not actual:
        return None

    forecast_map = {r['hour']: r['forecast_mw'] for r in forecast}
    actual_map = {r['hour']: r['load_mw'] for r in actual}

    comparison = []
    errors = []
    for h in range(24):
        f = forecast_map.get(h)
        a = actual_map.get(h)
        if f is not None and a is not None:
            error = f - a
            pct_error = (error / a * 100) if a != 0 else 0
            errors.append(abs(pct_error))
            comparison.append({
                'hour': h,
                'forecast_mw': round(f, 1),
                'actual_mw': round(a, 1),
                'error_mw': round(error, 1),
                'error_pct': round(pct_error, 2),
            })

    mape = round(sum(errors) / len(errors), 2) if errors else None

    return {
        'date': date_str,
        'comparison': comparison,
        'mape': mape,
        'rmse': round(float(np.sqrt(np.mean([e**2 for e in errors]))), 2) if errors else None,
    }


def get_price_statistics(days=30):
    """Get price statistics for the last N days."""
    end = datetime.now()
    start = end - timedelta(days=days)
    prices = db.get_dam_prices_range(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

    if not prices:
        return None

    by_hour = defaultdict(list)
    daily_avgs = defaultdict(list)

    for p in prices:
        if p['price_eur'] is not None:
            by_hour[p['hour']].append(p['price_eur'])
            daily_avgs[p['date']].append(p['price_eur'])

    hourly_stats = {}
    for h in range(24):
        vals = by_hour.get(h, [])
        if vals:
            hourly_stats[h] = {
                'mean': round(np.mean(vals), 2),
                'median': round(float(np.median(vals)), 2),
                'min': round(min(vals), 2),
                'max': round(max(vals), 2),
                'std': round(float(np.std(vals)), 2),
            }

    daily_stats = {}
    for date, vals in sorted(daily_avgs.items()):
        daily_stats[date] = round(np.mean(vals), 2)

    all_prices = [p['price_eur'] for p in prices if p['price_eur'] is not None]

    return {
        'period_days': days,
        'total_records': len(all_prices),
        'overall_mean': round(np.mean(all_prices), 2) if all_prices else None,
        'overall_median': round(float(np.median(all_prices)), 2) if all_prices else None,
        'overall_min': round(min(all_prices), 2) if all_prices else None,
        'overall_max': round(max(all_prices), 2) if all_prices else None,
        'hourly': hourly_stats,
        'daily': daily_stats,
    }
