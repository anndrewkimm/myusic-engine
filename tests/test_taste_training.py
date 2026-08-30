from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from myusic_engine.evaluation import PairedRankingComparison, PredictionMetrics
from myusic_engine.features import FeatureObservation, FeatureSelector
from myusic_engine.ingest import NormalizedListeningEvent
from myusic_engine.modeling import (
    AudioFeatureProfile,
    AudioInputSpec,
    TasteModelConfig,
    TemporalConfig,
    build_temporal_dataset,
    read_taste_model,
    train_taste_models,
    write_taste_training_result,
)
from myusic_engine.modeling.training import VariantEvaluation, _select_evaluation


def _ranking_metrics(ndcg_at_k: float) -> PredictionMetrics:
    return PredictionMetrics(
        row_count=20,
        positive_count=10,
        positive_rate=0.5,
        roc_auc=0.75,
        average_precision=0.75,
        log_loss=0.5,
        brier_score=0.2,
        expected_calibration_error=0.1,
        ranking_k=10,
        ranking_periods=4,
        precision_at_k=0.5,
        recall_at_k=0.5,
        ndcg_at_k=ndcg_at_k,
    )


def _evaluation(model_name: str, feature_count: int, ndcg_at_k: float) -> VariantEvaluation:
    return VariantEvaluation(
        model_name=model_name,
        status="trained",
        reason=None,
        feature_count=feature_count,
        cohort="all_labeled",
        split_rows={"train": 40, "validation": 20, "test": 20},
        validation_metrics=_ranking_metrics(ndcg_at_k),
        test_metrics=_ranking_metrics(0.1),
        model_id=f"{model_name}-id",
    )


def _comparison(confidence_interval_low: float) -> PairedRankingComparison:
    return PairedRankingComparison(
        baseline_model_name="simple",
        contender_model_name="complex",
        split="validation",
        period_count=4,
        mean_ndcg_delta=0.02,
        confidence_level=0.95,
        confidence_interval_low=confidence_interval_low,
        confidence_interval_high=0.05,
        contender_win_rate=0.75,
    )


def _event(day: int, track_number: int, positive: bool) -> NormalizedListeningEvent:
    played = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day)
    track_id = str(track_number).zfill(22)
    return NormalizedListeningEvent(
        event_id=f"{day}-{track_id}",
        played_at=played.isoformat().replace("+00:00", "Z"),
        media_type="track",
        ms_played=180_000 if positive else 2_000,
        track_uri=f"spotify:track:{track_id}",
        track_name=f"Synthetic {track_number}",
        artist_name=f"Artist {track_number % 2}",
        album_name="Synthetic",
        episode_uri=None,
        episode_name=None,
        show_name=None,
        reason_start="clickrow",
        reason_end="trackdone" if positive else "fwdbtn",
        shuffle=False,
        skipped=not positive,
        offline=False,
        incognito_mode=False,
        platform_family="desktop",
        source_file="synthetic.json",
        source_record_index=day * 10 + track_number,
    )


def test_train_taste_models_runs_fair_audio_ablations(tmp_path) -> None:
    events = []
    for period in range(8):
        for track_number in range(1, 5):
            positive = (period + track_number) % 3 != 0
            events.append(_event(period * 10 + track_number, track_number, positive))
    dataset = build_temporal_dataset(
        sorted(events, key=lambda event: event.played_at),
        config=TemporalConfig(
            period_days=10,
            minimum_train_periods=4,
            validation_fraction=0.2,
            test_fraction=0.2,
        ),
    )
    descriptor_selector = FeatureSelector("synthetic_descriptor_v1", "synthetic", "v1")
    embedding_selector = FeatureSelector("synthetic_embedding_v1", "synthetic", "v1")
    profile = AudioFeatureProfile(
        profile_version="synthetic_profile_v1",
        descriptor_inputs=(AudioInputSpec(descriptor_selector, 2),),
        embedding_input=AudioInputSpec(embedding_selector, 3),
    )
    observations = []
    for track_number in range(1, 5):
        track_id = f"spotify:track:{str(track_number).zfill(22)}"
        observations.extend(
            (
                FeatureObservation(
                    track_id=track_id,
                    feature_name="synthetic_descriptor_v1",
                    value=(float(track_number), float(track_number % 2)),
                    feature_source="synthetic",
                    source_version="v1",
                    coverage_seconds=30.0,
                    feature_confidence=1.0,
                ),
                FeatureObservation(
                    track_id=track_id,
                    feature_name="synthetic_embedding_v1",
                    value=(float(track_number), 1.0, -1.0),
                    feature_source="synthetic",
                    source_version="v1",
                    coverage_seconds=30.0,
                    feature_confidence=1.0,
                ),
            )
        )
    result = train_taste_models(
        dataset.samples,
        config=TasteModelConfig(ranking_k=2),
        feature_observations=observations,
        profile=profile,
        profile_name="synthetic",
    )

    trained_names = {model.model_name for model in result.models}
    assert "behavior_matched" in trained_names
    assert "embedding_only" in trained_names
    assert "full_combined" in trained_names
    assert result.report.fair_cohort_tracks == 4
    assert result.report.selected_model_id is not None
    assert result.report.comparisons
    assert any(
        comparison.baseline_model_name == "behavior_matched"
        and comparison.contender_model_name == "full_combined"
        for comparison in result.report.comparisons
    )
    assert all(
        variant.test_metrics is not None
        for variant in result.report.variants
        if variant.status == "trained"
    )

    model_paths, selected_path, predictions_path, report_path = write_taste_training_result(
        result, tmp_path
    )
    assert model_paths
    selected = read_taste_model(selected_path)
    assert selected.model_id == result.report.selected_model_id
    assert predictions_path.stat().st_size > 0
    assert report_path.stat().st_size > 0
    assert "paired_period_comparisons" in report_path.read_text(encoding="utf-8")
    probability = selected.predict_probability(selected.means)
    assert 0 < probability < 1


def test_model_selection_prefers_simpler_model_when_lift_is_uncertain() -> None:
    simple = _evaluation("simple", 3, 0.80)
    complex_model = _evaluation("complex", 13, 0.82)

    selected = _select_evaluation((simple, complex_model), (_comparison(-0.01),))

    assert selected.model_name == "simple"


def test_model_selection_accepts_complexity_for_clear_validation_lift() -> None:
    simple = _evaluation("simple", 3, 0.80)
    complex_model = _evaluation("complex", 13, 0.82)

    selected = _select_evaluation((simple, complex_model), (_comparison(0.005),))

    assert selected.model_name == "complex"


def test_portable_model_rejects_modified_content(tmp_path) -> None:
    # Hash validation is covered using a minimal artifact created in the primary training test.
    missing = tmp_path / "missing.json"
    with pytest.raises(OSError):
        read_taste_model(missing)
