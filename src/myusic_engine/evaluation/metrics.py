"""Classification, calibration, and period-level ranking metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


class EvaluationError(ValueError):
    """Raised when prediction metrics receive invalid or incomparable rows."""


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Held-out metrics with null discrimination scores for one-class splits."""

    row_count: int
    positive_count: int
    positive_rate: float
    roc_auc: float | None
    average_precision: float | None
    log_loss: float
    brier_score: float
    expected_calibration_error: float
    ranking_k: int
    ranking_periods: int
    precision_at_k: float | None
    recall_at_k: float | None
    ndcg_at_k: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "positive_count": self.positive_count,
            "positive_rate": self.positive_rate,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "ranking_k": self.ranking_k,
            "ranking_periods": self.ranking_periods,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
        }


def _rounded(value: float) -> float:
    return round(float(value), 8)


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        count = int(np.sum(selected))
        if count:
            confidence = float(np.mean(probabilities[selected]))
            accuracy = float(np.mean(labels[selected]))
            error += count / total * abs(confidence - accuracy)
    return error


def _ranking_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    period_indices: Sequence[int],
    ranking_k: int,
) -> tuple[int, float | None, float | None, float | None]:
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for label, probability, period_index in zip(
        labels, probabilities, period_indices, strict=True
    ):
        groups[period_index].append((label, probability))
    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    for rows in groups.values():
        positives = sum(label for label, _ in rows)
        if positives == 0:
            continue
        ordered = sorted(rows, key=lambda item: item[1], reverse=True)
        limit = min(ranking_k, len(ordered))
        top = ordered[:limit]
        hits = sum(label for label, _ in top)
        precision_values.append(hits / limit)
        recall_values.append(hits / positives)
        dcg = math.fsum(
            label / math.log2(rank + 2) for rank, (label, _) in enumerate(top)
        )
        ideal_hits = min(positives, limit)
        ideal = math.fsum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        ndcg_values.append(dcg / ideal if ideal else 0.0)
    if not precision_values:
        return 0, None, None, None
    return (
        len(precision_values),
        _rounded(float(np.mean(precision_values))),
        _rounded(float(np.mean(recall_values))),
        _rounded(float(np.mean(ndcg_values))),
    )


def evaluate_predictions(
    labels: Sequence[int],
    probabilities: Sequence[float],
    period_indices: Sequence[int],
    *,
    ranking_k: int,
) -> PredictionMetrics:
    """Evaluate one aligned held-out prediction vector without using future context."""

    if not labels or len(labels) != len(probabilities) or len(labels) != len(period_indices):
        raise EvaluationError("Prediction labels, probabilities, and periods must align")
    if ranking_k < 1:
        raise EvaluationError("ranking_k must be positive")
    if any(label not in {0, 1} for label in labels):
        raise EvaluationError("Prediction labels must be binary")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(probability_array)) or np.any(
        (probability_array < 0) | (probability_array > 1)
    ):
        raise EvaluationError("Prediction probabilities must be finite and in [0, 1]")
    label_array = np.asarray(labels, dtype=np.int64)
    clipped = np.clip(probability_array, 1e-12, 1.0 - 1e-12)
    two_classes = len(set(labels)) == 2
    ranking_periods, precision, recall, ndcg = _ranking_metrics(
        labels, probabilities, period_indices, ranking_k
    )
    return PredictionMetrics(
        row_count=len(labels),
        positive_count=int(np.sum(label_array)),
        positive_rate=_rounded(float(np.mean(label_array))),
        roc_auc=_rounded(roc_auc_score(label_array, probability_array)) if two_classes else None,
        average_precision=(
            _rounded(average_precision_score(label_array, probability_array))
            if two_classes
            else None
        ),
        log_loss=_rounded(log_loss(label_array, clipped, labels=[0, 1])),
        brier_score=_rounded(brier_score_loss(label_array, probability_array)),
        expected_calibration_error=_rounded(
            _expected_calibration_error(label_array, probability_array)
        ),
        ranking_k=ranking_k,
        ranking_periods=ranking_periods,
        precision_at_k=precision,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
    )
