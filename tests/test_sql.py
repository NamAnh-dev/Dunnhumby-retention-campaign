"""Run the real SQL files on tiny hand-made tables, where the right answer is easy to check by hand."""

import duckdb
import pandas as pd
import pytest

from src.config import SQL_DIR
from src.data import run_sql


def connect_with(**tables) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con


def test_weekly_activity_aggregates_lines_into_baskets():
    tx = pd.DataFrame({
        "household_key":     [1, 1, 1, 2],
        "basket_id":         [10, 10, 11, 20],
        "week_no":           [1, 1, 1, 1],
        "product_id":        [100, 200, 100, 100],
        "sales_value":       [2.0, 3.0, 5.0, 4.0],
        "retail_disc":       [-0.5, 0.0, 0.0, 0.0],
        "coupon_disc":       [0.0, -1.0, 0.0, 0.0],
        "coupon_match_disc": [0.0, 0.0, 0.0, 0.0],
    })
    product = pd.DataFrame({"product_id": [100, 200], "department": ["GROCERY", "PRODUCE"]})
    out = run_sql(SQL_DIR / "01_weekly_activity.sql", connect_with(transaction_data=tx, product=product))
    hh1 = out.set_index("household_key").loc[1]
    assert hh1["n_baskets"] == 2
    assert hh1["n_items"] == 3
    assert hh1["n_departments"] == 2
    assert hh1["net_sales"] == 10.0
    assert hh1["discount_amount"] == 1.5        # discounts are negative in the raw data
    assert hh1["n_coupon_items"] == 1


def test_rfm_recency_score_uses_fixed_weeks():
    # the data ends in week 20; households 1-6 last bought 0, 1, 2, 3, 5 and 9 weeks before; household 7 = same as household 1
    tx = pd.DataFrame({"household_key": [1, 2, 3, 4, 5, 6, 7],
                       "week_no":       [20, 19, 18, 17, 15, 11, 20],
                       "sales_value":   [1.0] * 7})
    out = run_sql(SQL_DIR / "02_rfm.sql", connect_with(transaction_data=tx)).set_index("household_key")
    assert out["r_score"].tolist() == [5, 4, 3, 3, 2, 1, 5]      # same recency -> same score, whatever the household_key
    assert "m_score" not in out.columns                         # segments use R and F only


def test_rfm_frequency_score_and_segments():
    # 10 households that all bought in the last week, with 1..10 purchase weeks
    rows = [(hh, week) for hh in range(1, 11) for week in range(21 - hh, 21)]
    tx = pd.DataFrame(rows, columns=["household_key", "week_no"]).assign(sales_value=1.0)
    out = run_sql(SQL_DIR / "02_rfm.sql", connect_with(transaction_data=tx)).set_index("household_key")
    assert out["f_score"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert out.loc[10, "segment"] == "champions"                 # bought last week (R = 5) and most often (F = 5)
    assert out.loc[1, "segment"] == "new_or_light"               # bought last week, but only once


def test_weekly_kpis():
    # week 1: household 1 buys in 2 baskets, household 2 only returns an item
    tx = pd.DataFrame({
        "household_key":     [1, 1, 2],
        "basket_id":         [10, 11, 20],
        "week_no":           [1, 1, 1],
        "sales_value":       [6.0, 4.0, -2.0],
        "retail_disc":       [-1.0, 0.0, 0.0],
        "coupon_disc":       [0.0, 0.0, 0.0],
        "coupon_match_disc": [0.0, 0.0, 0.0],
    })
    out = run_sql(SQL_DIR / "07_weekly_kpis.sql", connect_with(transaction_data=tx)).set_index("week_no")
    assert out.loc[1, "active_households"] == 1     # a week with only returns is not a purchase
    assert out.loc[1, "baskets"] == 3
    assert out.loc[1, "net_sales"] == 8.0
    assert out.loc[1, "discount_share"] == pytest.approx(1 / 9)    # discount / (net sales + discount)


def test_period_retention():
    # 4-week periods: household 1 is active in periods 1 and 2, household 2 only in period 1
    tx = pd.DataFrame({
        "household_key": [1, 1, 2, 3],
        "week_no":       [1, 5, 2, 9],
        "sales_value":   [1.0, 1.0, 1.0, 1.0],
    })
    out = run_sql(SQL_DIR / "03_period_retention.sql", connect_with(transaction_data=tx)).set_index("period")
    assert out.loc[1, "active_households"] == 2
    assert out.loc[1, "retained_next_period"] == 1
    assert out.loc[1, "retention_rate"] == 0.5


def test_purchase_gaps_with_lag():
    tx = pd.DataFrame({
        "household_key": [1, 1, 1, 1],
        "week_no":       [1, 2, 6, 6],
        "sales_value":   [1.0, 1.0, 1.0, 1.0],     # two lines in week 6 = still one purchase week
    })
    out = run_sql(SQL_DIR / "05_purchase_gaps.sql", connect_with(transaction_data=tx))
    assert out["silent_weeks"].tolist() == [0, 3]  # 1 -> 2: no gap, 2 -> 6: weeks 3, 4, 5 silent


def test_demographics_cleaning():
    tx = pd.DataFrame({"household_key": [1, 2], "week_no": [1, 1], "sales_value": [1.0, 1.0]})
    demo = pd.DataFrame({
        "household_key": [1], "age_desc": ["45-54"], "income_desc": ["50-74K"],
        "household_size_desc": ["5+"], "homeowner_desc": ["Probable Renter"],
    })
    out = run_sql(SQL_DIR / "06_household_demographics.sql",
                  connect_with(transaction_data=tx, hh_demographic=demo)).set_index("household_key")
    assert out.loc[1, "income_k"] == 62
    assert out.loc[1, "household_size"] == 5
    assert out.loc[1, "homeowner"] == "renter"
    assert out.loc[2, "has_demographics"] == 0      # household 2 is kept, with missing demographics
    assert pd.isna(out.loc[2, "income_k"])


def test_department_sales_share():
    tx = pd.DataFrame({
        "household_key": [1, 1, 2],
        "product_id":    [100, 200, 100],
        "sales_value":   [3.0, 1.0, 4.0],
    })
    product = pd.DataFrame({"product_id": [100, 200], "department": ["GROCERY", "PRODUCE"]})
    out = run_sql(SQL_DIR / "04_department_sales.sql",
                  connect_with(transaction_data=tx, product=product)).set_index("department")
    assert out.loc["GROCERY", "sales"] == 7.0
    assert out.loc["GROCERY", "n_households"] == 2
    assert out.loc["GROCERY", "share_of_sales"] == 7 / 8
    assert out["share_of_sales"].sum() == 1.0
