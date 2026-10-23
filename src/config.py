"""Project settings in one place. Change values here, not inside the code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

SEED = 42

# Churn ("lapse") definition, see src/features.py
OBSERVATION_WEEKS = 12   # a household is "at risk" if it bought at least once in the last 12 weeks
PREDICTION_WEEKS = 8     # churn = 1 if it does not buy in the next 8 weeks
SNAPSHOT_STEP = 8        # one snapshot every 8 weeks, so label windows never overlap
FIRST_CUTOFF = 30        # earlier weeks are still "warming up" (households are still appearing)

# Time-based split over the snapshot cutoff weeks
VALID_CUTOFFS = [70, 78, 86]   # choose the model here: for each week, train on the earlier snapshots and score this one
TEST_CUTOFF = 94               # report the final result here, once

TOP_FRACTION = 0.10      # assumed budget: we can send a coupon to 10% of at-risk households
