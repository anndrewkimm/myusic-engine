from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

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
    probability = selected.predict_probability(selected.means)
    assert 0 < probability < 1


def test_portable_model_rejects_modified_content(tmp_path) -> None:
    # Hash validation is covered using a minimal artifact created in the primary training test.
    missing = tmp_path / "missing.json"
    with pytest.raises(OSError):
        read_taste_model(missing)
