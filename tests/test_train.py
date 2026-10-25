import numpy as np
import pandas as pd
import pytest

from src.config import PREDICTION_WEEKS, TEST_CUTOFF, VALID_CUTOFF
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


def test_time_split_train_valid_test():
    table = pd.DataFrame({"cutoff_week": range(30, 95, 8)})      # the real snapshots: 30, 38, ..., 94
    train, valid, test = time_split(table)
    assert train["cutoff_week"].tolist() == [30, 38, 46, 54, 62, 70, 78]
    assert set(valid["cutoff_week"]) == {VALID_CUTOFF}
    assert set(test["cutoff_week"]) == {TEST_CUTOFF}
    assert train["cutoff_week"].max() + PREDICTION_WEEKS <= VALID_CUTOFF   # train labels end before validation
    assert VALID_CUTOFF + PREDICTION_WEEKS <= TEST_CUTOFF                   # validation labels end before test


def test_time_split_refuses_overlapping_label_windows():
    table = pd.DataFrame({"cutoff_week": [74, 82, 86, 94]})      # the label of week 82 looks at weeks 83..90
    with pytest.raises(AssertionError):
        time_split(table)


def test_preprocessor_logs_skewed_columns_and_scales_every_feature():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.exponential(10, (200, len(FEATURES))), columns=FEATURES)
    prep = make_preprocessor().fit(train)
    out = pd.DataFrame(prep.transform(train), columns=prep.get_feature_names_out())

    assert sorted(out.columns) == sorted(FEATURES)                    # one column per feature, nothing added
    assert np.allclose(out.mean(), 0) and np.allclose(out.std(ddof=0), 1)
    logged = np.log1p(train["sales_12w"])                             # skewed columns get log1p before scaling
    assert np.allclose(out["sales_12w"], (logged - logged.mean()) / logged.std(ddof=0))
