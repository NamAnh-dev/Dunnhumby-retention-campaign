import numpy as np
import pandas as pd

from src.business import compare_rankings, targeting_table
from src.config import PREDICTION_WEEKS, TEST_CUTOFF, VALID_CUTOFFS
from src.features import FEATURES
from src.train import evaluate, make_preprocessor, precision_at_top, time_split


def test_precision_at_top():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert precision_at_top(y, score, fraction=0.2) == 0.5      # top 2: one churner
    assert precision_at_top(y, score, fraction=0.3) == 2 / 3


def test_lift_of_a_perfect_model():
    y = [1] * 10 + [0] * 90
    score = list(range(100, 0, -1))      # churners get the highest scores
    result = evaluate(y, score)
    assert result["roc_auc"] == 1.0
    assert result["lift_top10"] == 10.0  # base rate 10%, precision in the top 10% = 100%


def test_time_split_keeps_the_test_week_apart():
    table = pd.DataFrame({"cutoff_week": range(30, 95, 8)})
    dev, test = time_split(table)
    assert set(test["cutoff_week"]) == {TEST_CUTOFF}
    assert dev["cutoff_week"].max() + PREDICTION_WEEKS <= TEST_CUTOFF     # no dev label window reaches the test week
    assert set(VALID_CUTOFFS) <= set(dev["cutoff_week"])                    # every validation week is in dev


def test_preprocessor_handles_missing_and_new_categories():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.random((20, len(FEATURES))), columns=FEATURES)
    train["age_group"] = ["25-34", None] * 10
    train["homeowner"] = ["owner", None] * 10
    train.loc[::2, ["income_k", "household_size"]] = np.nan
    new = train.head(3).copy()
    new["age_group"] = "an age group never seen"

    prep = make_preprocessor().fit(train)
    out = prep.transform(new)
    assert not np.isnan(out).any()                     # every missing value is filled
    assert out.shape[1] == prep.transform(train).shape[1]


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
