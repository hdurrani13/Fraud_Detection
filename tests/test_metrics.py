import numpy as np
import pytest

from fraud.metrics import best_threshold, evaluate_threshold, summary_metrics

y = np.array([0, 0, 1, 1, 0, 1])
scores = np.array([0.1, 0.8, 0.9, 0.3, 0.2, 0.7])
amounts = np.array([10.0, 20.0, 100.0, 50.0, 5.0, 200.0])


def test_counts_and_dollars_at_threshold():
    r = evaluate_threshold(y, scores, amounts, threshold=0.5, review_cost=5)
    # flagged: idx 1 (legit), 2 (fraud), 5 (fraud); missed fraud: idx 3 ($50)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (2, 1, 1)
    assert r.fraud_dollars_caught == 300
    assert r.fraud_dollars_missed == 50
    assert r.review_cost == 15
    assert r.total_cost == 65
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 3)


def test_flag_nothing_costs_all_fraud():
    r = evaluate_threshold(y, scores, amounts, threshold=np.inf)
    assert r.flagged == 0
    assert r.total_cost == 350


def test_best_threshold_beats_extremes():
    best = best_threshold(y, scores, amounts, review_cost=5)
    flag_all = evaluate_threshold(y, scores, amounts, 0.0, 5)
    flag_none = evaluate_threshold(y, scores, amounts, np.inf, 5)
    assert best.total_cost <= min(flag_all.total_cost, flag_none.total_cost)


def test_expensive_reviews_raise_the_threshold():
    cheap = best_threshold(y, scores, amounts, review_cost=1)
    pricey = best_threshold(y, scores, amounts, review_cost=500)
    assert pricey.flagged <= cheap.flagged


def test_summary_metrics_perfect_model():
    m = summary_metrics(y, y.astype(float))
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0 and m["recall_at_90_precision"] == 1.0
