from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import *


def run_json(cmd: list[str], timeout: int = 180):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-1500:])
    s = p.stdout.strip()
    starts = [i for i in (s.find('{'), s.find('[')) if i >= 0]
    if not starts:
        raise RuntimeError('No JSON returned by fli')
    return json.loads(s[min(starts):])


def iso_date(v):
    if isinstance(v, date):
        return v.isoformat()
    if not isinstance(v, str):
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(v[:10], fmt).date().isoformat()
        except ValueError:
            pass
    return None


def as_price(v):
    try:
        x = float(v)
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None


def extract_date_price_pairs(obj):
    """Tolerate minor JSON-schema changes in fli date-search output."""
    pairs = []
    if isinstance(obj, dict):
        d = None
        p = None
        for k in ('date', 'departure_date', 'travel_date', 'depart_date'):
            if k in obj:
                d = iso_date(obj[k])
                if d:
                    break
        for k in ('price', 'price_inr', 'lowest_price', 'amount'):
            if k in obj:
                p = as_price(obj[k])
                if p is not None:
                    break
        if d and p is not None:
            pairs.append((d, p))
        for v in obj.values():
            pairs.extend(extract_date_price_pairs(v))
    elif isinstance(obj, list):
        for v in obj:
            pairs.extend(extract_date_price_pairs(v))
    return pairs


def load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    return df if not df.empty else pd.DataFrame(columns=cols)


def save_unique(df: pd.DataFrame, path: Path, keys: list[str]):
    if df.empty:
        return
    df = df.drop_duplicates(keys, keep='last').sort_values(keys)
    df.to_csv(path, index=False)


def collect_exact(stamp: datetime) -> list[dict]:
    rows = []
    for target in TARGET_DATES:
        try:
            data = run_json([
                'fli', 'flights', ORIGIN, DESTINATION, target,
                '--stops', NONSTOP, '--class', CABIN, '--sort', 'CHEAPEST',
                '--currency', CURRENCY, '--country', COUNTRY, '--language', LANGUAGE,
                '--passengers', '1', '--format', 'json'
            ])
            flights = data.get('flights', data if isinstance(data, list) else [])
            for item in flights:
                legs = item.get('legs') or []
                if not legs:
                    continue
                leg = legs[0]
                airline = str(leg.get('airline', '')).upper().replace(' ', '')
                number = str(leg.get('flight_number', '')).upper().replace(' ', '')
                flight_no = number if number.startswith('6E') else f'{airline}{number}'
                if not flight_no.startswith('6E'):
                    continue
                price = as_price(item.get('price'))
                if price is None:
                    continue
                rows.append({
                    'observed_date': stamp.date().isoformat(),
                    'observed_at_utc': stamp.isoformat(),
                    'target_date': target,
                    'flight_no': flight_no,
                    'price_inr': price,
                    'departure_time': str(leg.get('departure_datetime', ''))[-8:-3],
                    'arrival_time': str(leg.get('arrival_datetime', ''))[-8:-3],
                    'source': 'google-flights-via-fli',
                })
        except Exception as e:
            ERROR_LOG.parent.mkdir(exist_ok=True)
            with ERROR_LOG.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'kind': 'exact', 'date': target, 'error': str(e), 'at': stamp.isoformat()}) + '\n')
    return rows


def collect_sweep(stamp: datetime) -> list[dict]:
    # Broad learning panel: from tomorrow through 30 days after the latest
    # target departure. This captures the entire target month plus surrounding
    # dates, while previously collected observations remain in the CSV forever.
    start = stamp.date() + timedelta(days=1)
    hard_end = date.fromisoformat(max(TARGET_DATES)) + timedelta(days=POST_TARGET_LEARNING_DAYS)
    if start > hard_end:
        return []
    end = hard_end
    try:
        data = run_json([
            'fli', 'dates', ORIGIN, DESTINATION,
            '--from', start.isoformat(), '--to', end.isoformat(),
            '--class', CABIN, '--stops', NONSTOP, '--sort', 'CHEAPEST',
            '--currency', CURRENCY, '--country', COUNTRY, '--language', LANGUAGE,
            '--passengers', '1', '--format', 'json'
        ], timeout=600)
    except Exception as e:
        ERROR_LOG.parent.mkdir(exist_ok=True)
        with ERROR_LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'kind': 'sweep', 'error': str(e), 'at': stamp.isoformat()}) + '\n')
        return []

    rows = []
    seen = set()
    for d, p in extract_date_price_pairs(data):
        if date.fromisoformat(d) < start or date.fromisoformat(d) > end:
            continue
        key = (stamp.date().isoformat(), d)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'observed_date': stamp.date().isoformat(),
            'observed_at_utc': stamp.isoformat(),
            'departure_date': d,
            'price_inr': p,
            'source': 'google-flights-date-sweep-via-fli',
        })
    return rows


def main():
    stamp = datetime.now(timezone.utc)
    exact_cols = ['observed_date','observed_at_utc','target_date','flight_no','price_inr','departure_time','arrival_time','source']
    sweep_cols = ['observed_date','observed_at_utc','departure_date','price_inr','source']

    e = load_csv(EXACT_CSV, exact_cols)
    s = load_csv(SWEEP_CSV, sweep_cols)
    exact = pd.DataFrame(collect_exact(stamp), columns=exact_cols)
    sweep = pd.DataFrame(collect_sweep(stamp), columns=sweep_cols)

    if not exact.empty:
        e = pd.concat([e, exact], ignore_index=True)
    if not sweep.empty:
        s = pd.concat([s, sweep], ignore_index=True)

    if not e.empty:
        save_unique(e, EXACT_CSV, ['observed_date','target_date','flight_no'])
    else:
        e.to_csv(EXACT_CSV, index=False)
    if not s.empty:
        save_unique(s, SWEEP_CSV, ['observed_date','departure_date'])
    else:
        s.to_csv(SWEEP_CSV, index=False)

    subprocess.run(['python', 'model.py'], check=True)
    print(f'exact={len(exact)} sweep={len(sweep)}')


if __name__ == '__main__':
    main()
