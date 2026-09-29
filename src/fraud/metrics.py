"""Evaluation for a problem where accuracy is meaningless.

With 0.17% fraud, a model that says "legit" every time is 99.8% accurate. So we
report precision-recall AUC and, more usefully, the dollar cost of each decision
threshold:

- Missed fraud (false negative): the bank loses the transaction amount.
- Flagged transaction (true or false positive): an analyst reviews it, at a fixed cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

DEFAULT_REVIEW_COST = 5.0  # dollars per flagged transaction


@dataclass
class ThresholdResult:
    threshold: float
    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    fraud_dollars_caught: float
    fraud_dollars_missed: float
    review_cost: float
    total_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_threshold(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, threshold: float, review_cost: float = DEFAULT_REVIEW_COST
) -> ThresholdResult:
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(scores) >= threshold
    amounts = np.asarray(amounts, dtype=float)

    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    fn = int((~flagged & y).sum())
    caught = float(amounts[flagged & y].sum())
    missed = float(amounts[~flagged & y].sum())
    reviews = float(flagged.sum() * review_cost)
    return ThresholdResult(
        threshold=float(threshold),
        flagged=int(flagged.sum()),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        fraud_dollars_caught=round(caught, 2),
        fraud_dollars_missed=round(missed, 2),
        review_cost=round(reviews, 2),
        total_cost=round(missed + reviews, 2),
    )


def threshold_grid(scores: np.ndarray, n: int = 400) -> np.ndarray:
    """Candidate thresholds: quantiles of the scores, concentrated where they matter."""
    qs = np.unique(np.quantile(scores, np.linspace(0.9, 1.0, n)))
    return np.unique(np.concatenate([[0.0], qs, [np.inf]]))


def cost_curve(y, scores, amounts, review_cost: float = DEFAULT_REVIEW_COST) -> list[ThresholdResult]:
    return [evaluate_threshold(y, scores, amounts, t, review_cost) for t in threshold_grid(scores)]


def best_threshold(y, scores, amounts, review_cost: float = DEFAULT_REVIEW_COST) -> ThresholdResult:
    return min(cost_curve(y, scores, amounts, review_cost), key=lambda r: (r.total_cost, -r.threshold))


def summary_metrics(y, scores) -> dict:
    precision, recall, _ = precision_recall_curve(y, scores)
    # Best recall achievable while keeping precision >= 90%
    ok = precision >= 0.9
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_90_precision": float(recall[ok].max()) if ok.any() else 0.0,
    }
