"""Business views of the results: where sales come from (RFM segments), and who should get a coupon.

Who should get a coupon? Compare ways of ranking the at-risk households on a snapshot with known labels.
  * by churn risk          : churn probability
  * by expected sales lost : churn probability x sales_12w   (losing a big spender costs more than losing a small one)
  * baselines without the model: recency rule x sales_12w, or sales_12w alone

For the top-k households of each ranking we count the real churners reached and their "sales at risk"
(what those churners spent in the last 12 weeks, i.e. what the store loses if they stop coming).
No coupon cost or success rate is assumed: whether a coupon really keeps a household can only be measured with an A/B test.
"""

import numpy as np
import pandas as pd


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Per RFM segment: households, sales, and the segment's share of all households and of all sales."""
    out = (rfm.groupby("segment")
              .agg(households=("household_key", "size"),
                   sales=("monetary", "sum"),
                   median_active_weeks=("frequency_weeks", "median"),
                   median_recency_weeks=("recency_weeks", "median"))
              .sort_values("sales", ascending=False))
    out["share_of_households"] = out["households"] / out["households"].sum()
    out["share_of_sales"] = out["sales"] / out["sales"].sum()
    out["sales_per_household"] = out["sales"] / out["households"]
    return out.reset_index()


def targeting_table(y, score, sales, fractions=(0.05, 0.10, 0.20, 0.30, 0.50, 1.0)) -> pd.DataFrame:
    """One row per share of households targeted (highest score first): churners reached and their sales at risk."""
    y, sales = np.asarray(y), np.asarray(sales, dtype=float)
    churner_sales = y * sales                      # 0 for households that do not churn
    order = np.argsort(-np.asarray(score), kind="stable")   # stable: ties keep the table order on every machine
    rows = []
    for f in fractions:
        top = order[: max(1, int(len(y) * f))]
        rows.append({
            "share_targeted": f,
            "households": len(top),
            "churners_reached": int(y[top].sum()),
            "share_of_all_churners": y[top].sum() / y.sum(),
            "sales_at_risk_reached": churner_sales[top].sum(),
            "share_of_sales_at_risk": churner_sales[top].sum() / churner_sales.sum(),
        })
    return pd.DataFrame(rows)


def compare_rankings(y, rankings: dict, sales, fraction=0.10) -> pd.DataFrame:
    """One row per ranking ({name: score}), all with the same budget (`fraction` of the households)."""
    rows = {name: targeting_table(y, score, sales, fractions=[fraction]).iloc[0] for name, score in rankings.items()}
    return pd.DataFrame(rows).T[["households", "churners_reached", "share_of_all_churners", "share_of_sales_at_risk"]]
