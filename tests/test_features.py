import pandas as pd
import pytest

from src.features import BEHAVIOUR_FEATURES, build_features, build_labels, build_model_table


def make_weekly(rows):
    """rows: list of (household_key, week_no, net_sales). Each row = 1 basket, 1 department, no discount."""
    df = pd.DataFrame(rows, columns=["household_key", "week_no", "net_sales"])
    df["n_baskets"] = 1
    df["n_departments"] = 1
    df["discount_amount"] = 0.0
    df["n_coupon_items"] = 0
    return df


def test_labels_churn_and_at_risk():
    weekly = make_weekly([
        (1, 20, 10), (1, 25, 10),   # active before cutoff 20, buys again in week 25 -> churn 0
        (2, 15, 10),                # active before cutoff, never comes back         -> churn 1
        (3, 2, 10), (3, 25, 10),    # silent in weeks 9..20                          -> not at risk
        (4, 40, 10),                # makes the data long enough
    ])
    labels = build_labels(weekly, cutoff=20).set_index("household_key")["churn"]
    assert labels.to_dict() == {1: 0, 2: 1}


def test_return_week_is_not_a_purchase():
    weekly = make_weekly([(1, 20, 10), (1, 25, -5), (2, 40, 10)])   # week 25 is only a return
    labels = build_labels(weekly, cutoff=20)
    assert labels.loc[labels["household_key"] == 1, "churn"].item() == 1


def test_cutoff_too_late_raises():
    weekly = make_weekly([(1, 20, 10), (1, 25, 10)])
    with pytest.raises(ValueError):
        build_labels(weekly, cutoff=20)    # weeks 21..28 are not all in the data


def test_features_values():
    weekly = make_weekly([(1, 11, 30), (1, 17, 20), (1, 20, 10)])
    weekly.loc[2, "discount_amount"] = 20       # week 20: 10 paid + 20 discount
    f = build_features(weekly, cutoff=20).iloc[0]
    assert f["recency_weeks"] == 0
    assert f["tenure_weeks"] == 10
    assert f["active_weeks_4w"] == 2
    assert f["active_weeks_12w"] == 3
    assert f["sales_12w"] == 60
    assert f["avg_basket_value_12w"] == 20
    assert f["discount_share_12w"] == 0.25       # 20 / (60 + 20)


def test_features_ignore_the_future():
    past = make_weekly([(1, 10, 10), (1, 18, 10)])
    with_future = make_weekly([(1, 10, 10), (1, 18, 10), (1, 21, 999), (1, 30, 999)])
    a = build_features(past, cutoff=20)[BEHAVIOUR_FEATURES]
    b = build_features(with_future, cutoff=20)[BEHAVIOUR_FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_model_table_keeps_households_without_demographics():
    weekly = make_weekly([(1, 25, 10), (2, 28, 10), (1, 38, 10)])     # data ends in week 38
    demographics = pd.DataFrame({"household_key": [1, 2], "has_demographics": [1, 0], "income_k": [62, None]})
    table = build_model_table(weekly, demographics)
    assert len(table) == 2                                  # one snapshot (week 30), two households
    assert table.set_index("household_key")["income_k"].isna().to_dict() == {1: False, 2: True}
