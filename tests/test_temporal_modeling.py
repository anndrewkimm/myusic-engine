from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from myusic_engine.ingest import NormalizedListeningEvent
from myusic_engine.modeling import (
    BEHAVIOR_FEATURE_NAMES,
    TemporalConfig,
    TemporalDatasetError,
    build_temporal_dataset,
    load_modeling_config,
    read_behavior_snapshots,
    read_temporal_samples,
    write_temporal_dataset,
)


def _event(
    day: int,
    track: str,
    *,
    positive: bool,
    artist: str = "Synthetic Artist",
) -> NormalizedListeningEvent:
    played = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day)
    return NormalizedListeningEvent(
        event_id=f"event-{day}-{track}",
        played_at=played.isoformat().replace("+00:00", "Z"),
        media_type="track",
        ms_played=180_000 if positive else 5_000,
        track_uri=f"spotify:track:{track:0>22}",
        track_name=f"Track {track}",
        artist_name=artist,
        album_name="Synthetic Album",
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
        source_record_index=day,
    )


def test_temporal_dataset_freezes_features_before_target_period(tmp_path) -> None:
    events = []
    for period in range(6):
        events.append(_event(period * 10, "1", positive=period != 4))
        events.append(_event(period * 10 + 1, "2", positive=period % 2 == 0))
    config = TemporalConfig(
        period_days=10,
        minimum_train_periods=2,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    result = build_temporal_dataset(events, config=config)

    track_one = [sample for sample in result.samples if sample.track_id.endswith("1".zfill(22))]
    assert len(track_one) == 6
    assert track_one[0].behavior_features[BEHAVIOR_FEATURE_NAMES.index("prior_seen")] == 0
    assert track_one[1].behavior_features[
        BEHAVIOR_FEATURE_NAMES.index("prior_log_play_count")
    ] == pytest.approx(0.69314718)
    assert track_one[-1].split == "test"
    assert track_one[-2].label == 0
    assert result.report.split_period_counts == {"train": 4, "validation": 1, "test": 1}

    paths = write_temporal_dataset(result, tmp_path)
    assert read_temporal_samples(paths[0]) == result.samples
    assert read_behavior_snapshots(paths[1]) == result.snapshots
    assert json.loads(paths[2].read_text(encoding="utf-8"))["periods_with_samples"] == 6


def test_temporal_dataset_rejects_out_of_order_history() -> None:
    events = [_event(10, "1", positive=True), _event(0, "1", positive=True)]
    with pytest.raises(TemporalDatasetError, match="sorted chronologically"):
        build_temporal_dataset(events, config=TemporalConfig(minimum_train_periods=1))


def test_modeling_config_locks_audio_provenance() -> None:
    config = load_modeling_config("configs/modeling.yaml")
    profile = config.profiles["local_discogs_effnet"]
    assert profile.embedding_input is not None
    assert profile.embedding_input.dimensions == 1280
    assert sum(item.dimensions for item in profile.descriptor_inputs) == 61
