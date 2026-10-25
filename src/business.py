"""Business views of the results: where sales come from (RFM segments), and who should get a coupon.

Who should get a coupon? Compare ways of ranking the at-risk households on a snapshot with known labels.
  * by churn risk          : churn probability
  * by expected sales lost : churn probability x sales_12w   (losing a big spender costs more than losing a small one)
  * baselines without the model: recency rule x sales_12w, or sales_12w alone

For the top-k households of each ranking we count the real churners reached and their "sales at risk"
(what those churners spent in the last 12 weeks, i.e. what the store loses if they stop coming).

Is the campaign worth it? expected_value_table makes the unknowns explicit (share of churners a coupon keeps,
coupon cost), and ab_sample_size says how many households an A/B test needs to measure the first one.
"""

import numpy as np
import pandas as pd


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Per RFM segment: households, sales, and the segment's share of all households and of all sales."""
    per_segment = rfm.groupby("segment").agg(
        households=("household_key", "size"),
        sales=("monetary", "sum"),
        median_active_weeks=("frequency_weeks", "median"),
        median_recency_weeks=("recency_weeks", "median"),
    )
    out = per_segment.sort_values("sales", ascending=False)
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
    """One row per ranking ({name: score}), all with the same budget (`fraction` of the households).

    share_not_churning = households in the list that would not have churned anyway:
    for them the coupon is only a discount.
    """
    rows = {}
    for name, score in rankings.items():
        rows[name] = targeting_table(y, score, sales, fractions=[fraction]).iloc[0]
    out = pd.DataFrame(rows).T
    out = out[["households", "churners_reached", "share_of_all_churners", "share_of_sales_at_risk"]]
    out["share_not_churning"] = 1 - out["churners_reached"] / out["households"]
    return out


def expected_value_table(prob, sales, retention_rates=(0.10, 0.20, 0.30), coupon_costs=(2, 5, 10)) -> pd.DataFrame:
    """Total expected value of sending one coupon to every household of a list, for assumed values of r and c.

    Per household: p x r x sales - c, where p = churn probability (model), r = share of would-be churners the coupon
    keeps, c = cost of one coupon, sales = sales_12w (the sales we keep if the household stays).
    r and c are assumptions: r can only be measured with an A/B test.
    """
    prob, sales = np.asarray(prob, dtype=float), np.asarray(sales, dtype=float)
    table = {}
    for c in coupon_costs:
        table[c] = [float(np.sum(prob * r * sales - c)) for r in retention_rates]
    out = pd.DataFrame(table, index=pd.Index(retention_rates, name="retention_rate"))
    return out.rename_axis(columns="coupon_cost")


def ab_sample_size(p1: float, p2: float, z_alpha=1.96, z_power=0.84) -> int:
    """Households needed in EACH group of an A/B test to detect a churn rate going from p1 (control) to p2 (coupon).

    Two-sided test of two proportions at 5% (z = 1.96) with 80% power (z = 0.84), rounded up:
    n = (z_alpha + z_power)^2 x (p1 (1 - p1) + p2 (1 - p2)) / (p1 - p2)^2
    """
    n = (z_alpha + z_power) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
    return int(np.ceil(n))
