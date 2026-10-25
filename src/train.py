"""Train and compare churn models with a time-based split, then save the final model.

  1. Split by time: train = snapshots 30 .. 78, validation = week 86 (VALID_CUTOFF), test = week 94 (TEST_CUTOFF).
  2. For each model, fit every setting of a small grid on train, score it on validation (PR-AUC) and keep the best.
  3. Refit each model's best setting on train + validation (every week before the test) and evaluate it once on test.
  4. The final model (FINAL_MODEL in config.py, with its best validation setting) is refit on ALL labelled
     snapshots (including the test week) and saved to models/churn_model.joblib: this is the model used to score new weeks.
We split by time (not at random) because the same household appears in several snapshots,
and in real use the model always predicts the future from the past.

Usage (from the repo root, after python -m src.features):
    python -m src.train
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (FINAL_MODEL, MODEL_PATH, OBSERVATION_WEEKS, PREDICTION_WEEKS, PROCESSED_DIR,
                        REPORTS_DIR, SEED, TEST_CUTOFF, TOP_FRACTION, VALID_CUTOFF)
from src.features import FEATURES

# Long right tail (a few households spend a lot): log1p makes them easier for the linear model
SKEWED = ["baskets_12w", "sales_12w", "avg_basket_value_12w", "coupon_items_12w"]
OTHER = [c for c in FEATURES if c not in SKEWED]


def time_split(table: pd.DataFrame):
    """train = snapshots before the validation week, valid = the validation week, test = the test week."""
    train = table[table["cutoff_week"] < VALID_CUTOFF]
    valid = table[table["cutoff_week"] == VALID_CUTOFF]
    test = table[table["cutoff_week"] == TEST_CUTOFF]
    # A label looks PREDICTION_WEEKS ahead: it must end before the next part's cutoff,
    # otherwise the model would learn from weeks it is later asked to predict
    assert train["cutoff_week"].max() + PREDICTION_WEEKS <= VALID_CUTOFF
    assert VALID_CUTOFF + PREDICTION_WEEKS <= TEST_CUTOFF
    return train, valid, test


def make_preprocessor() -> ColumnTransformer:
    """Same preprocessing for every model, fitted on the training data only (inside the pipeline).

    Standardizing puts every feature on the same scale, so the logistic coefficients can be compared.
    The behaviour features have no missing values (no purchase in a window = 0), so nothing needs to be filled.
    """
    return ColumnTransformer([
        ("skewed", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                 StandardScaler()), SKEWED),
        ("other", StandardScaler(), OTHER),
    ], verbose_feature_names_out=False)


class RecencyRule:
    """Baseline without machine learning: the longer since the last purchase, the higher the risk.

    recency_weeks has only 12 values (0..11), so many households tie, and the top 10% would then depend on the
    row order of the table. Ties are broken by active_weeks_12w: fewer purchase weeks = riskier. Its weight (0.01)
    is smaller than one week of recency (1/12), so it never changes the recency order.
    """

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        recency = X["recency_weeks"].to_numpy(dtype=float) / OBSERVATION_WEEKS           # 0 .. 11/12
        inactive = 1 - X["active_weeks_12w"].to_numpy(dtype=float) / OBSERVATION_WEEKS   # 0 .. 11/12
        risk = recency + 0.01 * inactive
        return np.column_stack([1 - risk, risk])


def make_model(name: str) -> Pipeline:
    """A new, untrained pipeline (preprocessing + classifier).

    Two models are enough to compare a linear model with a tree model. LightGBM was tried before and tied with both.
    """
    classifiers = {
        "logistic": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    return Pipeline([("prep", make_preprocessor()), ("clf", classifiers[name])])


# Settings tried for each model. Grids are kept small on purpose.
GRIDS = {
    "logistic": {"clf__C": [0.01, 0.1, 1, 10]},
    "random_forest": {"clf__max_depth": [4, 8, 12], "clf__min_samples_leaf": [20, 50, 100]},
}


def precision_at_top(y, score, fraction=TOP_FRACTION) -> float:
    """Share of real churners among the `fraction` of households with the highest score."""
    k = max(1, int(len(y) * fraction))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]     # stable: ties keep the table order on every machine
    return float(np.asarray(y)[top].mean())


def evaluate(y, score) -> dict:
    base_rate = float(np.mean(y))
    p_top = precision_at_top(y, score)
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "precision_top10": p_top,
        "lift_top10": p_top / base_rate,     # how many times better than picking at random
    }


def validation_score(model, train: pd.DataFrame, valid: pd.DataFrame) -> float:
    """PR-AUC on the validation week of a model fitted on the training weeks."""
    model.fit(train[FEATURES], train["churn"])
    return float(average_precision_score(valid["churn"], model.predict_proba(valid[FEATURES])[:, 1]))


def tune(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Score every setting of every model on validation. Returns one row per model with its best setting."""
    rows = [{"model": "recency_rule", "params": {}, "valid_pr_auc": validation_score(RecencyRule(), train, valid)}]
    for name, grid in GRIDS.items():
        for params in ParameterGrid(grid):
            model = make_model(name).set_params(**params)
            rows.append({"model": name, "params": params, "valid_pr_auc": validation_score(model, train, valid)})
            print(f"{name:14s} {str(params):55s} valid PR-AUC {rows[-1]['valid_pr_auc']:.3f}")
    results = pd.DataFrame(rows)
    best_rows = results.groupby("model", sort=False)["valid_pr_auc"].idxmax()
    return results.loc[best_rows].set_index("model")


def build_model(name: str, params: dict):
    return RecencyRule() if name == "recency_rule" else make_model(name).set_params(**params)


def main():
    table = pd.read_csv(PROCESSED_DIR / "model_table.csv")
    train, valid, test = time_split(table)
    print(f"train {len(train):,} rows (cutoffs < {VALID_CUTOFF}) | valid {len(valid):,} rows (cutoff {VALID_CUTOFF}) "
          f"| test {len(test):,} rows (cutoff {TEST_CUTOFF})\n")

    best = tune(train, valid)
    print(f"\nBest setting per model (PR-AUC on validation week {VALID_CUTOFF}):")
    print(best.round(3))

    # Test: refit each best setting on train + validation (every week before the test), score the test week once
    before_test = pd.concat([train, valid])
    scores = {}
    for name, row in best.iterrows():
        model = build_model(name, row["params"]).fit(before_test[FEATURES], before_test["churn"])
        scores[name] = model.predict_proba(test[FEATURES])[:, 1]
    test_results = pd.DataFrame({name: evaluate(test["churn"], s) for name, s in scores.items()}).T
    print("\nTest (reported once):")
    print(test_results.round(3))

    # Final model (why logistic regression: see FINAL_MODEL in config.py), refit on every labelled snapshot
    params = best.loc[FINAL_MODEL, "params"]
    final = build_model(FINAL_MODEL, params).fit(table[FEATURES], table["churn"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"\nFinal model: {FINAL_MODEL} {params}, refit on all {len(table):,} rows -> {MODEL_PATH.name}")

    REPORTS_DIR.mkdir(exist_ok=True)
    best.to_csv(REPORTS_DIR / "validation_results.csv")
    test_results.to_csv(REPORTS_DIR / "test_results.csv")
    test_scores = test[["household_key", "churn"] + FEATURES].copy()
    for name, s in scores.items():
        test_scores[f"score_{name}"] = s
    test_scores.to_csv(PROCESSED_DIR / "test_scores.csv", index=False)


if __name__ == "__main__":
    main()
