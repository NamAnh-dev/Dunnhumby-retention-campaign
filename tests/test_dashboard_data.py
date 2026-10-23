"""The summary tables behind the dashboard, on tiny hand-made data."""

import pandas as pd

from src.dashboard_data import churn_by_snapshot, segment_summary


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


def test_churn_by_snapshot():
    table = pd.DataFrame({"household_key": [1, 2, 1, 2], "cutoff_week": [30, 30, 38, 38], "churn": [0, 1, 0, 0]})
    out = churn_by_snapshot(table).set_index("cutoff_week")
    assert out.loc[30, "churn_rate"] == 0.5
    assert out.loc[38, "households_at_risk"] == 2
