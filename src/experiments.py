"""Extra experiments used in notebook 04: class weights, bootstrap confidence intervals and the decision threshold."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import SEED
from src.features import FEATURES
from src.train import precision_at_top


# --------------------------------------------------------------------------------------------- 1. class weights
def compare_class_weight(models: dict, train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Fit every model twice (normal / class_weight="balanced") on train and score it on valid.

    Two questions: does the ranking get better (PR-AUC, precision in the top 10%)?
    And do the probabilities still match the real churn rate (mean predicted probability)?
    """
    rows = []
    for name, model in models.items():
        for class_weight in [None, "balanced"]:
            model.set_params(clf__class_weight=class_weight)
            model.fit(train[FEATURES], train["churn"])
            prob = model.predict_proba(valid[FEATURES])[:, 1]
            rows.append({
                "model": name,
                "class_weight": class_weight or "none",
                "pr_auc": average_precision_score(valid["churn"], prob),
                "precision_top10": precision_at_top(valid["churn"], prob),
                "mean_predicted": prob.mean(),        # compare with the real churn rate
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- 2. bootstrap intervals
def bootstrap_ci(y, score, metric=average_precision_score, n_boot=1000, seed=SEED, alpha=0.05):
    """95% interval of a metric: resample the households with replacement, recompute the metric, take percentiles."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:            # a sample without churners has no PR-AUC
            continue
        values.append(metric(y[idx], score[idx]))
    return tuple(np.quantile(values, [alpha / 2, 1 - alpha / 2]))


# --------------------------------------------------------------------------------------------- 3. decision threshold
def best_f1_threshold(y, prob) -> float:
    """Probability threshold with the highest F1 score (balance of precision and recall)."""
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])      # the last precision/recall pair has no threshold


def threshold_report(y, prob, thresholds: dict) -> pd.DataFrame:
    """For each named threshold: how many households are flagged, and precision / recall / F1 among them."""
    y, prob = np.asarray(y), np.asarray(prob)
    rows = []
    for name, t in thresholds.items():
        flagged = prob >= t
        tp = (flagged & (y == 1)).sum()
        precision = tp / max(flagged.sum(), 1)
        recall = tp / y.sum()
        rows.append({"rule": name, "threshold": t, "households_flagged": int(flagged.sum()),
                     "share_flagged": flagged.mean(), "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / max(precision + recall, 1e-12)})
    return pd.DataFrame(rows).set_index("rule")
