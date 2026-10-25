"""Segment shares and coupon targeting, on tiny hand-made data."""

import pandas as pd

from src.business import compare_rankings, segment_summary, targeting_table


def test_segment_summary_shares():
    rfm = pd.DataFrame({
        "household_key":   [1, 2, 3, 4],
        "segment":         ["champions", "champions", "lost", "lost"],
        "monetary":        [300.0, 500.0, 100.0, 100.0],
        "frequency_weeks": [50, 60, 5, 7],
        "recency_weeks":   [0, 1, 10, 20],
    })
    out = segment_summary(rfm).set_index("segment")
    assert out.index[0] == "champions"                  # biggest sales first
    assert out.loc["champions", "share_of_households"] == 0.5
    assert out.loc["champions", "share_of_sales"] == 0.8
    assert out.loc["lost", "sales_per_household"] == 100.0


def test_targeting_table():
    y = [1, 0, 1, 0]
    score = [0.9, 0.8, 0.2, 0.1]
    sales = [100, 50, 300, 10]
    row = targeting_table(y, score, sales, fractions=[0.5]).iloc[0]
    assert row["households"] == 2                      # the two highest scores
    assert row["churners_reached"] == 1
    assert row["sales_at_risk_reached"] == 100         # only churners count, the non-churner's 50 is not at risk
    assert row["share_of_sales_at_risk"] == 0.25       # 100 of the 400 spent by all churners


def test_compare_rankings():
    y = [1, 0, 1, 0]
    sales = [100, 50, 300, 10]
    out = compare_rankings(y, {"by_risk": [0.9, 0.8, 0.2, 0.1], "by_sales": sales}, sales, fraction=0.25)
    assert out.loc["by_risk", "share_of_sales_at_risk"] == 0.25      # picks household 1 (100 of 400)
    assert out.loc["by_sales", "share_of_sales_at_risk"] == 0.75     # picks household 3 (300 of 400)
