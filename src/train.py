"""Train and compare churn models with a time-based split, then save the final model.

  1. Split by time: dev = snapshots before the test week, test = the TEST_CUTOFF snapshot.
  2. Validation: for each validation week (70, 78, 86) train on the earlier snapshots only and score that week.
     A setting's validation score = its average PR-AUC over the 3 weeks (one week alone is noisy).
  3. For each model, keep the setting with the best validation score.
  4. Train each model on all dev snapshots and evaluate it once on the test snapshot.
  5. The model with the best validation score is refit on ALL labelled snapshots (including the test week)
     and saved to models/churn_model.joblib: this is the model used to score new weeks.
We split by time (not at random) because the same household appears in several snapshots,
and in real use the model always predicts the future from the past.

Usage (from the repo root, after python -m src.features):
    python -m src.train
"""

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (MODEL_PATH, OBSERVATION_WEEKS, PREDICTION_WEEKS, PROCESSED_DIR,
                        REPORTS_DIR, SEED, TEST_CUTOFF, TOP_FRACTION, VALID_CUTOFFS)
from src.features import CATEGORICAL_DEMOGRAPHICS, FEATURES, NUMERIC_FEATURES

# Long right tail (a few households spend a lot): log1p makes them easier for the linear model
SKEWED = ["baskets_12w", "sales_12w", "avg_basket_value_12w", "coupon_items_12w"]
OTHER_NUMERIC = [c for c in NUMERIC_FEATURES if c not in SKEWED]


def time_split(table: pd.DataFrame):
    """dev = every snapshot before the test week (training + validation), test = the test week."""
    # The label of the last dev snapshot looks 8 weeks ahead; it must end before the test week starts
    assert max(VALID_CUTOFFS) + PREDICTION_WEEKS <= TEST_CUTOFF
    dev = table[table["cutoff_week"] < TEST_CUTOFF]
    test = table[table["cutoff_week"] == TEST_CUTOFF]
    return dev, test


def make_preprocessor() -> ColumnTransformer:
    """Same preprocessing for every model, fitted on the training data only (inside the pipeline)."""
    return ColumnTransformer([
        ("skewed", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                 StandardScaler()), SKEWED),
        # income_k / household_size are missing for ~2/3 of households: fill with the median,
        # the has_demographics column tells the model which values were filled
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), OTHER_NUMERIC),
        ("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="unknown"),
                                      OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
         CATEGORICAL_DEMOGRAPHICS),
    ], verbose_feature_names_out=False)


class RecencyRule:
    """Baseline without machine learning: the longer since the last purchase, the higher the risk."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        risk = X["recency_weeks"].to_numpy(dtype=float) / OBSERVATION_WEEKS   # at-risk households have recency 0..11
        return np.column_stack([1 - risk, risk])


def make_model(name: str) -> Pipeline:
    """A new, untrained pipeline (preprocessing + classifier)."""
    classifiers = {
        "logistic": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "lightgbm": LGBMClassifier(n_estimators=300, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=SEED, verbose=-1),
    }
    return Pipeline([("prep", make_preprocessor()), ("clf", classifiers[name])])


# Settings tried for each model. Grids are kept small on purpose.
GRIDS = {
    "logistic": {"clf__C": [0.01, 0.1, 1, 10]},
    "random_forest": {"clf__max_depth": [4, 8, 12], "clf__min_samples_leaf": [20, 50, 100]},
    "lightgbm": {"clf__num_leaves": [7, 15, 31], "clf__learning_rate": [0.02, 0.05]},
}


def precision_at_top(y, score, fraction=TOP_FRACTION) -> float:
    """Share of real churners among the `fraction` of households with the highest score."""
    k = max(1, int(len(y) * fraction))
    top = np.argsort(-np.asarray(score))[:k]
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


def validation_score(model, dev: pd.DataFrame) -> float:
    """Average PR-AUC over the validation weeks. For each week, the model only learns from earlier snapshots."""
    scores = []
    for week in VALID_CUTOFFS:
        train = dev[dev["cutoff_week"] < week]
        valid = dev[dev["cutoff_week"] == week]
        model.fit(train[FEATURES], train["churn"])
        prob = model.predict_proba(valid[FEATURES])[:, 1]
        scores.append(average_precision_score(valid["churn"], prob))
    return float(np.mean(scores))


def tune(dev: pd.DataFrame) -> pd.DataFrame:
    """Score every setting of every model on validation. Returns one row per model with its best setting."""
    rows = [{"model": "recency_rule", "params": {}, "valid_pr_auc": validation_score(RecencyRule(), dev)}]
    for name, grid in GRIDS.items():
        for params in ParameterGrid(grid):
            model = make_model(name).set_params(**params)
            rows.append({"model": name, "params": params, "valid_pr_auc": validation_score(model, dev)})
            print(f"{name:14s} {str(params):55s} valid PR-AUC {rows[-1]['valid_pr_auc']:.3f}")
    results = pd.DataFrame(rows)
    best_rows = results.groupby("model", sort=False)["valid_pr_auc"].idxmax()
    return results.loc[best_rows].set_index("model")


def build_model(name: str, params: dict):
    return RecencyRule() if name == "recency_rule" else make_model(name).set_params(**params)


def main():
    table = pd.read_csv(PROCESSED_DIR / "model_table.csv")
    dev, test = time_split(table)
    print(f"dev {len(dev):,} rows (cutoffs < {TEST_CUTOFF}) | test {len(test):,} rows (cutoff {TEST_CUTOFF})\n")

    best = tune(dev)
    print("\nBest setting per model (average PR-AUC over validation weeks", VALID_CUTOFFS, "):")
    print(best.round(3))

    # Test: train each model on all dev snapshots, score the test week once
    scores = {}
    for name, row in best.iterrows():
        model = build_model(name, row["params"]).fit(dev[FEATURES], dev["churn"])
        scores[name] = model.predict_proba(test[FEATURES])[:, 1]
    test_results = pd.DataFrame({name: evaluate(test["churn"], s) for name, s in scores.items()}).T
    print("\nTest (reported once):")
    print(test_results.round(3))

    # Final model: chosen on validation (not on test), then refit on every labelled snapshot
    chosen = best.drop("recency_rule")["valid_pr_auc"].idxmax()
    final = build_model(chosen, best.loc[chosen, "params"]).fit(table[FEATURES], table["churn"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"\nChosen on validation: {chosen}, refit on all {len(table):,} rows -> {MODEL_PATH.name}")

    REPORTS_DIR.mkdir(exist_ok=True)
    best.to_csv(REPORTS_DIR / "validation_results.csv")
    test_results.to_csv(REPORTS_DIR / "test_results.csv")
    test_scores = test[["household_key", "churn"] + FEATURES].copy()
    for name, s in scores.items():
        test_scores[f"score_{name}"] = s
    test_scores.to_csv(PROCESSED_DIR / "test_scores.csv", index=False)


if __name__ == "__main__":
    main()
