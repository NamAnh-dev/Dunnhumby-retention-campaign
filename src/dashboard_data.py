"""Small summary tables for the Power BI report (powerbi/), saved to reports/dashboard/.

They only hold aggregates (per week, per segment, per department, per snapshot), so they are committed to the repo:
the Power BI report can load them right after cloning, without downloading the raw data.

Usage (from the repo root, after python -m src.data and python -m src.features):
    python -m src.dashboard_data
"""

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR

DASHBOARD_DIR = REPORTS_DIR / "dashboard"


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


def churn_by_snapshot(model_table: pd.DataFrame) -> pd.DataFrame:
    """Per cutoff week: households at risk, and the share of them that did not buy in the next 8 weeks."""
    return (model_table.groupby("cutoff_week")
                       .agg(households_at_risk=("household_key", "size"), churn_rate=("churn", "mean"))
                       .reset_index())


def main():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "weekly_kpis": pd.read_csv(PROCESSED_DIR / "weekly_kpis.csv"),
        "period_retention": pd.read_csv(PROCESSED_DIR / "period_retention.csv"),
        "department_sales": pd.read_csv(PROCESSED_DIR / "department_sales.csv"),
        "segments": segment_summary(pd.read_csv(PROCESSED_DIR / "rfm.csv")),
        "churn_by_snapshot": churn_by_snapshot(pd.read_csv(PROCESSED_DIR / "model_table.csv")),
    }
    for name, df in tables.items():
        df.round(4).to_csv(DASHBOARD_DIR / f"{name}.csv", index=False)
        print(f"{name:18s} -> reports/dashboard/{name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
