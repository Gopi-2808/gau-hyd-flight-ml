from __future__ import annotations

from datetime import date, timedelta
import json
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor,
    HistGradientBoostingRegressor, ExtraTreesClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from config import *

FEATURES = [
    'days_to_departure', 'departure_weekday', 'observed_weekday',
    'departure_month', 'departure_day',
    'current_price', 'price_vs_mean_7', 'price_vs_mean_14',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
    'mean_3', 'mean_7', 'mean_14', 'std_7', 'std_14',
    'change_1d', 'change_3d', 'change_7d',
    'trend_3', 'trend_7', 'range_7', 'range_14',
]


def read(path, dcols):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in dcols:
        df[c] = pd.to_datetime(df[c], errors='coerce').dt.date
    return df.dropna(subset=dcols)


def _slope(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or np.isnan(arr).any() or np.allclose(arr, arr[0]):
        return 0.0 if len(arr) >= 2 else np.nan
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def add_features(df):
    df = df.sort_values(['departure_date', 'observed_date']).copy()
    g = df.groupby('departure_date', group_keys=False)
    df['days_to_departure'] = [(d - r).days for r, d in zip(df.observed_date, df.departure_date)]
    df['departure_weekday'] = [d.weekday() for d in df.departure_date]
    df['observed_weekday'] = [d.weekday() for d in df.observed_date]
    df['departure_month'] = [d.month for d in df.departure_date]
    df['departure_day'] = [d.day for d in df.departure_date]
    df['current_price'] = df.price_inr
    for n in (1, 2, 3, 7, 14):
        df[f'lag_{n}'] = g.price_inr.shift(n)
    for n in (3, 7, 14):
        df[f'mean_{n}'] = g.price_inr.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=2).mean())
    for n in (7, 14):
        df[f'std_{n}'] = g.price_inr.transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=2).std())
        df[f'range_{n}'] = g.price_inr.transform(
            lambda s, n=n: (s.shift(1).rolling(n, min_periods=2).max() - s.shift(1).rolling(n, min_periods=2).min())
        )
    df['price_vs_mean_7'] = df.current_price / df.mean_7 - 1
    df['price_vs_mean_14'] = df.current_price / df.mean_14 - 1
    df['change_1d'] = df.current_price / df.lag_1 - 1
    df['change_3d'] = df.current_price / df.lag_3 - 1
    df['change_7d'] = df.current_price / df.lag_7 - 1
    df['trend_3'] = g.price_inr.transform(lambda s: s.shift(1).rolling(3, min_periods=2).apply(_slope, raw=True)) / df.current_price
    df['trend_7'] = g.price_inr.transform(lambda s: s.shift(1).rolling(7, min_periods=3).apply(_slope, raw=True)) / df.current_price
    return df


def make_training(sweep: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = add_features(sweep)
    future = sweep[['departure_date', 'observed_date', 'price_inr']].copy()
    future['label_date'] = future['observed_date'] - timedelta(days=horizon)
    future = future.rename(columns={'price_inr': 'future_price'})
    train = x.merge(
        future[['departure_date', 'label_date', 'future_price']],
        left_on=['departure_date', 'observed_date'],
        right_on=['departure_date', 'label_date'], how='inner'
    )
    train = train.replace([np.inf, -np.inf], np.nan)
    train['log_return'] = np.log(train.future_price / train.current_price)
    train['return'] = train.future_price / train.current_price - 1
    train['drop_event'] = (train['return'] <= -DROP_THRESHOLD).astype(int)
    train['rise_event'] = (train['return'] >= RISE_THRESHOLD).astype(int)
    train = train.dropna(subset=['future_price', 'log_return'])
    return train


def regression_candidates(seed=42):
    return [
        make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(
            n_estimators=160, min_samples_leaf=2, max_features=0.85, random_state=seed, n_jobs=-1)),
        make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(
            n_estimators=160, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=-1)),
        make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)),
    ]


def validation_split(train: pd.DataFrame):
    dates = sorted(train.observed_date.unique())
    if len(dates) < 8:
        return train.iloc[:0], train
    cut = max(2, int(len(dates) * (1 - VALIDATION_FRACTION)))
    cut_date = dates[min(cut, len(dates) - 1)]
    tr = train[train.observed_date < cut_date]
    va = train[train.observed_date >= cut_date]
    if tr.empty or va.empty:
        return train.iloc[:0], train
    return tr, va


def fit_regression(train: pd.DataFrame, horizon: int):
    if len(train) < MIN_TRAIN_ROWS:
        return None
    tr, va = validation_split(train)
    candidates = regression_candidates(42 + horizon)
    scores = []
    for model in candidates:
        if not tr.empty and len(va) >= 10:
            model.fit(tr[FEATURES], tr['log_return'])
            pred = model.predict(va[FEATURES])
            mae = float(np.mean(np.abs(pred - va['log_return'].to_numpy())))
            scores.append(mae)
        else:
            scores.append(np.nan)
    valid_scores = [x for x in scores if np.isfinite(x)]
    if valid_scores:
        raw = np.array([1.0 / max(s, 1e-5) if np.isfinite(s) else 0.0 for s in scores])
        weights = raw / raw.sum() if raw.sum() else np.ones(3) / 3
    else:
        weights = np.ones(3) / 3
    fitted = []
    for model in candidates:
        model.fit(train[FEATURES], train['log_return'])
        fitted.append(model)
    return {'models': fitted, 'weights': weights.tolist(), 'validation_mae': scores, 'n_train': int(len(train))}


def fit_classifier(train: pd.DataFrame, label: str, horizon: int):
    if len(train) < MIN_CLASS_ROWS or train[label].nunique() < 2:
        return None
    tr, va = validation_split(train)
    model = make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(
        n_estimators=140, min_samples_leaf=3, max_features=0.8,
        random_state=100 + horizon, class_weight='balanced', n_jobs=-1))
    # Fit once on all data. Validation is deliberately diagnostic only for the
    # classifier; it is not used to tune the forecast and therefore cannot leak.
    model.fit(train[FEATURES], train[label])
    if not va.empty and va[label].nunique() >= 1:
        val_prob = model.predict_proba(va[FEATURES])[:, 1]
        val_brier = float(np.mean((val_prob - va[label].to_numpy()) ** 2))
    else:
        val_brier = None
    return {'model': model, 'validation_brier': val_brier, 'n_train': int(len(train)), 'positive_rate': float(train[label].mean())}


def make_feature_from_history(history: pd.DataFrame, forecast_date: date, departure: date):
    h = history.sort_values('observed_date').copy()
    mp = dict(zip(h.observed_date, h.price_inr))
    cur = float(h.iloc[-1].price_inr)
    def exact_lag(n):
        return mp.get(forecast_date - timedelta(days=n), np.nan)
    vals = list(h.price_inr.astype(float))
    def roll(n, fn):
        r = vals[-n:]
        return fn(r) if len(r) >= 2 else np.nan
    lag_1, lag_2, lag_3, lag_7, lag_14 = [exact_lag(n) for n in (1,2,3,7,14)]
    mean_7, mean_14 = roll(7, np.mean), roll(14, np.mean)
    std_7, std_14 = roll(7, lambda r: np.std(r, ddof=1)), roll(14, lambda r: np.std(r, ddof=1))
    return pd.DataFrame([{
        'days_to_departure': (departure - forecast_date).days,
        'departure_weekday': departure.weekday(),
        'observed_weekday': forecast_date.weekday(),
        'departure_month': departure.month,
        'departure_day': departure.day,
        'current_price': cur,
        'price_vs_mean_7': cur / mean_7 - 1 if pd.notna(mean_7) and mean_7 else np.nan,
        'price_vs_mean_14': cur / mean_14 - 1 if pd.notna(mean_14) and mean_14 else np.nan,
        'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3, 'lag_7': lag_7, 'lag_14': lag_14,
        'mean_3': roll(3, np.mean), 'mean_7': mean_7, 'mean_14': mean_14,
        'std_7': std_7, 'std_14': std_14,
        'change_1d': cur / lag_1 - 1 if pd.notna(lag_1) and lag_1 else np.nan,
        'change_3d': cur / lag_3 - 1 if pd.notna(lag_3) and lag_3 else np.nan,
        'change_7d': cur / lag_7 - 1 if pd.notna(lag_7) and lag_7 else np.nan,
        'trend_3': _slope(vals[-3:]) / cur if len(vals) >= 2 and cur else np.nan,
        'trend_7': _slope(vals[-7:]) / cur if len(vals) >= 3 and cur else np.nan,
        'range_7': (max(vals[-7:]) - min(vals[-7:])) if len(vals) >= 2 else np.nan,
        'range_14': (max(vals[-14:]) - min(vals[-14:])) if len(vals) >= 2 else np.nan,
    }])


def ensemble_predict(bundle, x):
    preds = np.array([m.predict(x[FEATURES])[0] for m in bundle['models']], dtype=float)
    return float(np.dot(preds, np.asarray(bundle['weights'], dtype=float)))


def classifier_prob(bundle, x):
    if bundle is None:
        return None
    classes = list(bundle['model'].classes_)
    if 1 not in classes:
        return 0.0
    idx = classes.index(1)
    return float(bundle['model'].predict_proba(x[FEATURES])[0, idx])


def score_forecasts(exact):
    empty = {'n': 0, 'mae': None, 'smape': None, 'matured_30d': None, 'by_horizon': []}
    if not FORECAST_CSV.exists() or FORECAST_CSV.stat().st_size == 0 or exact.empty:
        return empty
    f = pd.read_csv(FORECAST_CSV)
    if f.empty:
        return empty
    for c in ['forecast_date', 'forecast_for_date', 'target_date']:
        f[c] = pd.to_datetime(f[c], errors='coerce').dt.date
    e = exact.copy()
    m = f.merge(
        e[['observed_date','target_date','flight_no','price_inr']].rename(columns={'price_inr':'actual_price'}),
        left_on=['forecast_for_date','target_date','flight_no'],
        right_on=['observed_date','target_date','flight_no'], how='inner'
    )
    if m.empty:
        return empty
    err = (m.actual_price - m.predicted_price_inr).abs()
    denom = (m.actual_price.abs() + m.predicted_price_inr.abs()) / 2
    smape = float((err / denom.replace(0, np.nan)).mean() * 100)
    by = []
    for h, g in m.groupby('horizon_days'):
        en = (g.actual_price - g.predicted_price_inr).abs()
        de = (g.actual_price.abs() + g.predicted_price_inr.abs()) / 2
        by.append({'horizon_days': int(h), 'n': int(len(g)), 'mae': float(en.mean()),
                   'smape': float((en / de.replace(0, np.nan)).mean() * 100)})
    matured = m[m.horizon_days >= 30]
    return {'n': int(len(m)), 'mae': float(err.mean()), 'smape': smape,
            'matured_30d': {'n': int(len(matured))} if not matured.empty else None,
            'by_horizon': by}


def main():
    today = date.today()
    exact = read(EXACT_CSV, ['observed_date', 'target_date'])
    sweep = read(SWEEP_CSV, ['observed_date', 'departure_date'])
    if exact.empty or sweep.empty:
        DASHBOARD_JSON.write_text(json.dumps({'meta': {'as_of': today.isoformat()}, 'series': [],
                                              'accuracy': score_forecasts(exact), 'model': {'status':'waiting_for_data'}}, indent=2))
        return

    bundles = {}
    drop_models = {}
    rise_models = {}
    status = {}
    for h in range(1, FORECAST_HORIZON + 1):
        train = make_training(sweep, h)
        bundles[h] = fit_regression(train, h)
        drop_models[h] = fit_classifier(train, 'drop_event', h)
        rise_models[h] = fit_classifier(train, 'rise_event', h)
        status[str(h)] = {
            'training_rows': int(len(train)),
            'regression_ready': bundles[h] is not None,
            'drop_classifier_ready': drop_models[h] is not None,
            'rise_classifier_ready': rise_models[h] is not None,
            'validation_mae_log_return': bundles[h]['validation_mae'] if bundles[h] else None,
            'regression_weights': bundles[h]['weights'] if bundles[h] else None,
            'drop_validation_brier': drop_models[h]['validation_brier'] if drop_models[h] else None,
            'rise_validation_brier': rise_models[h]['validation_brier'] if rise_models[h] else None,
        }

    rows = []
    for td_str in TARGET_DATES:
        td = date.fromisoformat(td_str)
        for flt in PREFERRED_FLIGHTS:
            hx = exact[(exact.target_date == td) & (exact.flight_no == flt) & (exact.observed_date <= today)].sort_values('observed_date')
            if hx.empty:
                continue
            anchor = float(hx.iloc[-1].price_inr)
            for h in range(1, FORECAST_HORIZON + 1):
                fd = today + timedelta(days=h)
                if fd >= td:
                    continue
                x = make_feature_from_history(hx, today, td)
                bundle = bundles.get(h)
                if bundle is None:
                    pred = anchor
                else:
                    log_ret = ensemble_predict(bundle, x)
                    pred = anchor * math.exp(float(np.clip(log_ret, -0.45, 0.45)))
                rows.append({
                    'forecast_date': today.isoformat(), 'forecast_for_date': fd.isoformat(),
                    'target_date': td.isoformat(), 'flight_no': flt,
                    'horizon_days': h, 'predicted_price_inr': round(max(0, pred), 2),
                    'drop_probability': None if drop_models.get(h) is None else round(classifier_prob(drop_models[h], x), 4),
                    'rise_probability': None if rise_models.get(h) is None else round(classifier_prob(rise_models[h], x), 4),
                    'model_training_rows': status[str(h)]['training_rows'],
                })

    fnew = pd.DataFrame(rows)
    fold = pd.read_csv(FORECAST_CSV) if FORECAST_CSV.exists() and FORECAST_CSV.stat().st_size > 0 else pd.DataFrame()
    f = pd.concat([fold, fnew], ignore_index=True)
    if not f.empty:
        for c in ['forecast_date', 'forecast_for_date', 'target_date']:
            f[c] = pd.to_datetime(f[c], errors='coerce').dt.date
        f = f.drop_duplicates(['forecast_date','forecast_for_date','target_date','flight_no'], keep='last').sort_values(
            ['forecast_date','target_date','flight_no','forecast_for_date'])
    f.to_csv(FORECAST_CSV, index=False)

    series = []
    for td_str in TARGET_DATES:
        td = date.fromisoformat(td_str)
        for flt in PREFERRED_FLIGHTS:
            hx = exact[(exact.target_date == td) & (exact.flight_no == flt)].sort_values('observed_date')
            if hx.empty:
                continue
            hist = [{'date': r.observed_date.isoformat(), 'price': round(float(r.price_inr), 2)} for r in hx.tail(HISTORY_DAYS).itertuples()]
            ff = f[(f.forecast_date == today) & (f.target_date == td) & (f.flight_no == flt)].sort_values('forecast_for_date') if not f.empty else pd.DataFrame()
            fc = []
            for r in ff.itertuples():
                fc.append({'date': r.forecast_for_date.isoformat(), 'price': round(float(r.predicted_price_inr), 2),
                           'horizon': int(r.horizon_days),
                           'drop_probability': r.drop_probability,
                           'rise_probability': r.rise_probability})
            current = float(hx.iloc[-1].price_inr)
            lower = min((z['price'] for z in fc), default=current)
            upper = max((z['price'] for z in fc), default=current)
            best = min(fc, key=lambda z: z['price'], default={'date': today.isoformat(), 'price': current})
            drop_pct = (current - lower) / current if current else 0
            rise_pct = (upper - current) / current if current else 0
            max_drop_prob = max((z['drop_probability'] or 0 for z in fc), default=0)
            max_rise_prob = max((z['rise_probability'] or 0 for z in fc), default=0)
            if max_rise_prob >= 0.65 and max_rise_prob > max_drop_prob:
                signal = 'HIGH INCREASE RISK'
            elif max_drop_prob >= 0.65 and max_drop_prob > max_rise_prob:
                signal = 'POSSIBLE LARGE DROP'
            elif drop_pct >= DROP_THRESHOLD and rise_pct >= RISE_THRESHOLD:
                signal = 'MIXED / VOLATILE'
            elif drop_pct >= DROP_THRESHOLD:
                signal = 'POSSIBLE LARGE DROP'
            elif rise_pct >= RISE_THRESHOLD:
                signal = 'HIGH INCREASE RISK'
            else:
                signal = 'NO LARGE MOVE DETECTED'
            series.append({
                'target_date': td_str, 'flight_no': flt, 'history': hist, 'forecast': fc,
                'current_price': current, 'best_predicted_day': best,
                'predicted_drop_pct': round(drop_pct * 100, 1), 'predicted_rise_pct': round(rise_pct * 100, 1),
                'max_drop_probability': round(max_drop_prob * 100, 1),
                'max_rise_probability': round(max_rise_prob * 100, 1),
                'signal': signal,
            })

    acc = score_forecasts(exact)
    payload = {
        'meta': {'as_of': today.isoformat(), 'origin': ORIGIN, 'destination': DESTINATION,
                 'target_dates': TARGET_DATES, 'decision_horizon_days': FORECAST_HORIZON,
                 'tracked_flights': PREFERRED_FLIGHTS, 'learning_window_end': (date.fromisoformat(max(TARGET_DATES)) + timedelta(days=POST_TARGET_LEARNING_DAYS)).isoformat(),
                 'learning_strategy': 'expanding daily panel of departure dates',
                 'drop_threshold_pct': DROP_THRESHOLD * 100, 'rise_threshold_pct': RISE_THRESHOLD * 100},
        'accuracy': acc, 'model': {'status': 'daily_retrained', 'horizons': status}, 'series': series,
    }
    DASHBOARD_JSON.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    MODEL_STATUS_JSON.write_text(json.dumps(status, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
