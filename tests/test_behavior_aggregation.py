import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from myusic_engine.ingest import load_history
from myusic_engine.ranking import (
    AffinityConfig,
    BehaviorAggregationError,
    aggregate_track_behavior,
    load_affinity_config,
    load_duration_map,
    score_affinity,
    write_track_affinities,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "spotify_history_synthetic.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRST_TRACK_URI = "spotify:track:0000000000000000000001"
SECOND_TRACK_URI = "spotify:track:0000000000000000000002"


def test_project_affinity_config_loads_as_the_versioned_defaults() -> None:
    loaded = load_affinity_config(PROJECT_ROOT / "configs" / "recommendation.yaml")

    assert loaded == AffinityConfig()


def test_behavior_aggregates_tracks_but_not_episodes() -> None:
    events = load_history(FIXTURE_PATH).events

    affinities = aggregate_track_behavior(events)
    by_track = {record.track_id: record for record in affinities}

    assert set(by_track) == {FIRST_TRACK_URI, SECOND_TRACK_URI}
    completed = by_track[FIRST_TRACK_URI]
    skipped = by_track[SECOND_TRACK_URI]
    assert completed.play_count == 1
    assert completed.total_ms_played == 180_000
    assert completed.completion_rate == 1.0
    assert completed.median_completion_ratio is None
    assert completed.duration_coverage_rate == 0.0
    assert completed.intentional_start_rate == 1.0
    assert completed.early_skip_rate == 0.0
    assert skipped.completion_rate == 0.0
    assert skipped.intentional_start_rate == 0.0
    assert skipped.early_skip_rate == 1.0
    assert completed.affinity_score > skipped.affinity_score


def test_duration_metadata_enables_completion_ratios() -> None:
    events = load_history(FIXTURE_PATH).events
    durations = {FIRST_TRACK_URI: 200_000, SECOND_TRACK_URI: 240_000}

    by_track = {
        record.track_id: record
        for record in aggregate_track_behavior(events, durations_ms=durations)
    }

    assert by_track[FIRST_TRACK_URI].duration_ms == 200_000
    assert by_track[FIRST_TRACK_URI].median_completion_ratio == 0.9
    assert by_track[FIRST_TRACK_URI].completion_rate == 1.0
    assert by_track[SECOND_TRACK_URI].median_completion_ratio == 0.05
    assert by_track[SECOND_TRACK_URI].completion_rate == 0.0


def test_missing_behavior_signals_remain_null_and_reduce_confidence() -> None:
    event = next(
        event for event in load_history(FIXTURE_PATH).events if event.track_uri == FIRST_TRACK_URI
    )
    unknown_signals = replace(event, reason_start=None, reason_end=None, skipped=None)

    known = aggregate_track_behavior([event])[0]
    unknown = aggregate_track_behavior([unknown_signals])[0]

    assert unknown.completion_rate is None
    assert unknown.early_skip_rate is None
    assert unknown.intentional_start_rate is None
    assert unknown.completion_signal_coverage == 0.0
    assert unknown.affinity_confidence < known.affinity_confidence


def test_repeat_rate_resets_after_the_session_gap() -> None:
    base = next(
        event for event in load_history(FIXTURE_PATH).events if event.track_uri == FIRST_TRACK_URI
    )
    events = [
        replace(base, event_id="a" * 64, played_at="2025-01-01T10:00:00.000Z"),
        replace(base, event_id="b" * 64, played_at="2025-01-01T10:10:00.000Z"),
        replace(base, event_id="c" * 64, played_at="2025-01-01T11:00:00.000Z"),
    ]

    affinity = aggregate_track_behavior(events)[0]

    assert affinity.play_count == 3
    assert affinity.repeat_within_session_rate == pytest.approx(1 / 3, abs=1e-6)


def test_affinity_formula_exposes_each_component() -> None:
    score, confidence, components = score_affinity(
        play_count=4,
        completion_rate=0.75,
        completion_coverage=1.0,
        intentional_start_rate=0.50,
        intentional_start_coverage=1.0,
        recency_score=0.80,
        early_skip_rate=0.25,
        early_skip_coverage=1.0,
        config=AffinityConfig(),
    )

    assert score == pytest.approx(math.log1p(4) + 1.5 * 0.75 + 0.5 + 0.5 * 0.8 - 2 * 0.25)
    assert 0 < confidence < 1
    assert set(components) == {
        "play_count",
        "completion",
        "intentional_start",
        "recency",
        "early_skip",
    }


def test_as_of_must_not_precede_the_latest_event() -> None:
    events = load_history(FIXTURE_PATH).events

    with pytest.raises(BehaviorAggregationError, match="latest event"):
        aggregate_track_behavior(
            events,
            as_of=datetime(2024, 12, 31, tzinfo=UTC),
        )


def test_duration_map_validation_and_affinity_output(tmp_path: Path) -> None:
    duration_path = tmp_path / "durations.json"
    duration_path.write_text(json.dumps({FIRST_TRACK_URI: 200_000}), encoding="utf-8")
    durations = load_duration_map(duration_path)
    affinities = aggregate_track_behavior(load_history(FIXTURE_PATH).events, durations_ms=durations)

    output_path = write_track_affinities(affinities, tmp_path / "affinity.jsonl")
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 2
    assert records[0]["affinity_score"] >= records[1]["affinity_score"]
    assert all(record["scoring_version"] == "behavior_affinity_v1" for record in records)


def test_invalid_duration_map_is_rejected(tmp_path: Path) -> None:
    duration_path = tmp_path / "durations.json"
    duration_path.write_text(json.dumps({FIRST_TRACK_URI: 0}), encoding="utf-8")

    with pytest.raises(BehaviorAggregationError, match="positive integer"):
        load_duration_map(duration_path)
