"""Churn labels and features at a cutoff week, and the model table that stacks several cutoffs.

For a cutoff week c:
  * active week : a household-week with net_sales > 0
  * at risk     : at least one active week in weeks c-11 .. c   (OBSERVATION_WEEKS = 12)
  * churn = 1   : no active week in weeks c+1 .. c+8            (PREDICTION_WEEKS = 8)
Behaviour features only use weeks <= c, so the model never sees the period it predicts (no leakage).
Demographics do not change over time, so they are simply joined on household_key.

Usage (from the repo root, after python -m src.data):
    python -m src.features
"""

import pandas as pd

from src.config import (FIRST_CUTOFF, OBSERVATION_WEEKS, PREDICTION_WEEKS,
                        PROCESSED_DIR, SNAPSHOT_STEP)

BEHAVIOUR_FEATURES = [
    "recency_weeks",          # weeks since the last purchase
    "tenure_weeks",           # weeks since the first purchase in the data
    "active_share",           # share of weeks with a purchase since the first purchase
    "active_weeks_4w",        # purchase weeks in the last 4 weeks
    "active_weeks_12w",       # purchase weeks in the last 12 weeks
    "baskets_12w",            # baskets in the last 12 weeks
    "sales_12w",              # net sales in the last 12 weeks
    "avg_basket_value_12w",   # sales_12w / baskets_12w
    "trend_4w_vs_12w",        # active_weeks_4w / 4 - active_weeks_12w / 12  (< 0 = buying less often lately)
    "discount_share_12w",     # discount / (sales + discount): how price-sensitive the household is
    "coupon_items_12w",       # items bought with a coupon in the last 12 weeks
    "avg_departments_12w",    # departments per purchase week: does the household do its full shopping here?
]
NUMERIC_DEMOGRAPHICS = ["has_demographics", "income_k", "household_size"]   # NaN for ~2/3 of households
CATEGORICAL_DEMOGRAPHICS = ["age_group", "homeowner"]
NUMERIC_FEATURES = BEHAVIOUR_FEATURES + NUMERIC_DEMOGRAPHICS
FEATURES = NUMERIC_FEATURES + CATEGORICAL_DEMOGRAPHICS


def active_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    return weekly[weekly["net_sales"] > 0]


def at_risk_households(weekly: pd.DataFrame, cutoff: int) -> list:
    """Households with at least one purchase week in the last OBSERVATION_WEEKS weeks up to the cutoff."""
    weeks = active_weeks(weekly)
    recent = weeks[(weeks["week_no"] > cutoff - OBSERVATION_WEEKS) & (weeks["week_no"] <= cutoff)]
    return sorted(recent["household_key"].unique())


def build_labels(weekly: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """household_key, cutoff_week, churn for every household at risk at this cutoff."""
    if cutoff + PREDICTION_WEEKS > weekly["week_no"].max():
        raise ValueError(f"cutoff {cutoff}: the next {PREDICTION_WEEKS} weeks are not in the data yet")
    active = active_weeks(weekly)
    in_pred = active[(active["week_no"] > cutoff) & (active["week_no"] <= cutoff + PREDICTION_WEEKS)]

    labels = pd.DataFrame({"household_key": at_risk_households(weekly, cutoff)})
    labels["cutoff_week"] = cutoff
    labels["churn"] = (~labels["household_key"].isin(in_pred["household_key"])).astype(int)
    return labels


def build_features(weekly: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """One row of behaviour features per household with a purchase on or before the cutoff."""
    hist = active_weeks(weekly)
    hist = hist[hist["week_no"] <= cutoff]          # never look past the cutoff
    last4 = hist[hist["week_no"] > cutoff - 4]
    last12 = hist[hist["week_no"] > cutoff - 12]

    g = hist.groupby("household_key")
    feats = pd.DataFrame({
        "recency_weeks": cutoff - g["week_no"].max(),
        "tenure_weeks": cutoff - g["week_no"].min() + 1,
    })
    feats["active_share"] = g["week_no"].nunique() / feats["tenure_weeks"]
    feats["active_weeks_4w"] = last4.groupby("household_key").size()

    g12 = last12.groupby("household_key")
    feats["active_weeks_12w"] = g12.size()
    feats["baskets_12w"] = g12["n_baskets"].sum()
    feats["sales_12w"] = g12["net_sales"].sum()
    feats["discount_12w"] = g12["discount_amount"].sum().clip(lower=0)
    feats["coupon_items_12w"] = g12["n_coupon_items"].sum()
    feats["avg_departments_12w"] = g12["n_departments"].mean()

    # Households with no purchase in a window get NaN from the groupby above: that means 0
    window_cols = ["active_weeks_4w", "active_weeks_12w", "baskets_12w", "sales_12w",
                   "discount_12w", "coupon_items_12w", "avg_departments_12w"]
    feats[window_cols] = feats[window_cols].fillna(0)

    feats["avg_basket_value_12w"] = (feats["sales_12w"] / feats["baskets_12w"]).fillna(0)
    feats["trend_4w_vs_12w"] = feats["active_weeks_4w"] / 4 - feats["active_weeks_12w"] / 12
    feats["discount_share_12w"] = (feats["discount_12w"] / (feats["sales_12w"] + feats["discount_12w"])).fillna(0)
    return feats[BEHAVIOUR_FEATURES].reset_index()


def snapshot_cutoffs(weekly: pd.DataFrame) -> list[int]:
    """Cutoff weeks FIRST_CUTOFF, +8, +16, ... while the next 8 weeks are still inside the data."""
    last = weekly["week_no"].max() - PREDICTION_WEEKS
    return list(range(FIRST_CUTOFF, last + 1, SNAPSHOT_STEP))


def build_model_table(weekly: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """Labels + features for every snapshot. One row per (household, cutoff_week)."""
    frames = []
    for cutoff in snapshot_cutoffs(weekly):
        labels = build_labels(weekly, cutoff)
        frames.append(labels.merge(build_features(weekly, cutoff), on="household_key", how="left"))
    table = pd.concat(frames, ignore_index=True)
    return table.merge(demographics, on="household_key", how="left", validate="many_to_one")


def build_scoring_table(weekly: pd.DataFrame, demographics: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Features for the households at risk at `cutoff`, without a label (the future is unknown): used to score new data."""
    households = pd.DataFrame({"household_key": at_risk_households(weekly, cutoff)})
    table = households.merge(build_features(weekly, cutoff), on="household_key", how="left")
    return table.merge(demographics, on="household_key", how="left", validate="one_to_one")


def load_inputs():
    weekly = pd.read_csv(PROCESSED_DIR / "weekly_activity.csv")
    demographics = pd.read_csv(PROCESSED_DIR / "household_demographics.csv")
    return weekly, demographics


def main():
    table = build_model_table(*load_inputs())
    table.to_csv(PROCESSED_DIR / "model_table.csv", index=False)
    print(table.groupby("cutoff_week")["churn"].agg(households="size", churn_rate="mean").round(3))
    print(f"Wrote {len(table):,} rows to data/processed/model_table.csv")


if __name__ == "__main__":
    main()
