"""Use the saved model on the latest week of data and write the coupon list.

Households are ranked by expected sales lost = churn probability x sales in the last 12 weeks (see src/business.py and
notebook 03: ranking by risk alone picks many low-spending households, which are a small loss if they stop coming).

The saved model was refit on every labelled snapshot (see src/train.py), so it has seen the most recent data.

Usage (from the repo root, after python -m src.train):
    python -m src.predict                 # score the households at risk in the last week of the data
    python -m src.predict --cutoff 90     # or at another week
Output: reports/coupon_list.csv with the TOP_FRACTION households with the highest expected sales lost.
"""

import argparse

import joblib

from src.config import MODEL_PATH, REPORTS_DIR, TOP_FRACTION
from src.features import FEATURES, build_scoring_table, load_inputs


def main(argv=None):
    """Score the households at risk at the cutoff week and write the top TOP_FRACTION by expected sales lost."""
    parser = argparse.ArgumentParser(description="Score at-risk households with the saved churn model.")
    parser.add_argument("--cutoff", type=int, help="week to score (default: the last week in the data)")
    args = parser.parse_args(argv)

    weekly, _ = load_inputs()          # the model does not use demographics
    cutoff = args.cutoff or int(weekly["week_no"].max())
    model = joblib.load(MODEL_PATH)

    table = build_scoring_table(weekly, cutoff)
    table["churn_probability"] = model.predict_proba(table[FEATURES])[:, 1]
    table["expected_sales_lost"] = table["churn_probability"] * table["sales_12w"]
    table = table.sort_values("expected_sales_lost", ascending=False)

    k = int(len(table) * TOP_FRACTION)
    columns = ["household_key", "churn_probability", "sales_12w", "expected_sales_lost",
               "recency_weeks", "active_weeks_12w"]
    coupon_list = table.head(k)[columns].round(3)
    coupon_list.to_csv(REPORTS_DIR / "coupon_list.csv", index=False)
    print(f"Week {cutoff}: {len(table):,} households at risk -> "
          f"coupon list of the top {TOP_FRACTION:.0%} ({k} households)")
    print(f"mean churn probability in the list {coupon_list['churn_probability'].mean():.1%} "
          f"vs {table['churn_probability'].mean():.1%} for all at-risk households")
    print("Wrote reports/coupon_list.csv")


if __name__ == "__main__":
    main()
