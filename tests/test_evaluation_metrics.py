from __future__ import annotations

import pytest

from myusic_engine.evaluation import (
    EvaluationError,
    compare_paired_rankings,
    evaluate_predictions,
)


def test_paired_period_bootstrap_detects_consistent_ranking_lift() -> None:
    labels = [1, 1, 0, 0] * 4
    periods = [period for period in range(4) for _ in range(4)]
    baseline = [0.1, 0.2, 0.9, 0.8] * 4
    contender = [0.9, 0.8, 0.2, 0.1] * 4

    comparison = compare_paired_rankings(
        labels,
        baseline,
        contender,
        periods,
        baseline_model_name="baseline",
        contender_model_name="contender",
        split="validation",
        ranking_k=2,
        bootstrap_resamples=500,
    )

    assert comparison.period_count == 4
    assert comparison.mean_ndcg_delta == pytest.approx(1.0)
    assert comparison.confidence_interval_low == pytest.approx(1.0)
    assert comparison.confidence_interval_high == pytest.approx(1.0)
    assert comparison.contender_win_rate == 1.0
    assert comparison.to_dict()["metric"] == "ndcg_at_k"


def test_evaluation_handles_one_class_split_without_inventing_discrimination() -> None:
    metrics = evaluate_predictions(
        [1, 1, 1],
        [0.6, 0.8, 0.7],
        [1, 1, 1],
        ranking_k=2,
    )
    assert metrics.roc_auc is None
    assert metrics.average_precision is None
    assert metrics.ndcg_at_k == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bootstrap_resamples": 0},
        {"confidence_level": 1.0},
        {"random_seed": -1},
    ],
)
def test_paired_period_bootstrap_rejects_invalid_controls(kwargs: dict[str, object]) -> None:
    with pytest.raises(EvaluationError):
        compare_paired_rankings(
            [1, 0],
            [0.6, 0.4],
            [0.7, 0.3],
            [1, 1],
            baseline_model_name="baseline",
            contender_model_name="contender",
            split="validation",
            ranking_k=1,
            **kwargs,  # type: ignore[arg-type]
        )
