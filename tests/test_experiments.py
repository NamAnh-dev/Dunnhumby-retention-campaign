import numpy as np
from sklearn.metrics import average_precision_score

from src.experiments import best_f1_threshold, bootstrap_ci, threshold_report


def make_scores(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    score = y * 0.5 + rng.random(n)          # informative but noisy
    return y, score


def test_bootstrap_interval_contains_the_estimate():
    y, score = make_scores()
    low, high = bootstrap_ci(y, score, n_boot=200)
    assert low < average_precision_score(y, score) < high


def test_bootstrap_interval_is_narrower_with_more_data():
    small = bootstrap_ci(*make_scores(n=200), n_boot=200)
    large = bootstrap_ci(*make_scores(n=5000), n_boot=200)
    assert large[1] - large[0] < small[1] - small[0]


def test_threshold_rules():
    y = np.array([0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    assert best_f1_threshold(y, prob) == 0.6     # flags exactly the two churners
    report = threshold_report(y, prob, {"low": 0.25, "high": 0.95})
    assert report.loc["low", "households_flagged"] == 3
    assert report.loc["low", "precision"] == 2 / 3
    assert report.loc["high", "recall"] == 0
