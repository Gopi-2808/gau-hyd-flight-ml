from pathlib import Path
from datetime import date

ORIGIN = 'GAU'
DESTINATION = 'HYD'
TARGET_DATES = ['2026-11-21', '2026-11-22']
PREFERRED_FLIGHTS = ['6E565', '6E187']
CURRENCY = 'INR'
COUNTRY = 'IN'
LANGUAGE = 'en-IN'
CABIN = 'ECONOMY'
NONSTOP = 'NON_STOP'

# Learn from a broad panel of departure dates around the target month.
# The collector searches from today through 30 days after the last target date.
# Older observations remain permanently stored, so the effective training set
# expands over time even though past departure dates are no longer searchable.
POST_TARGET_LEARNING_DAYS = 30
FORECAST_HORIZON = 10
HISTORY_DAYS = 60
DROP_THRESHOLD = 0.07
RISE_THRESHOLD = 0.07
MIN_TRAIN_ROWS = 60
MIN_CLASS_ROWS = 80
VALIDATION_FRACTION = 0.20

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
EXACT_CSV = DATA / 'exact_prices.csv'
SWEEP_CSV = DATA / 'sweep_prices.csv'
FORECAST_CSV = DATA / 'forecasts.csv'
DASHBOARD_JSON = DATA / 'dashboard.json'
MODEL_STATUS_JSON = DATA / 'model_status.json'
ERROR_LOG = DATA / 'errors.jsonl'
