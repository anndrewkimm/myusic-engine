import json
from pathlib import Path

import pytest

from myusic_engine.features import (
    FeatureCatalog,
    FeatureObservation,
    FeatureRecordError,
    FeatureSelector,
    read_feature_observations,
    write_feature_observations,
)


def _observation(track_id: str, feature_name: str, value: object) -> FeatureObservation:
    return FeatureObservation(
        track_id=track_id,
        feature_name=feature_name,
        value=value,  # type: ignore[arg-type]
        feature_source="clean_room_audio",
        source_version="extractor-0.1.0",
        coverage_seconds=30,
        feature_confidence=0.9,
    )


@pytest.mark.parametrize(
    ("feature_name", "value", "value_field"),
    [
        ("tempo_bpm_estimate_v1", 120, "value_number"),
        ("key_estimate_v1", "C major", "value_text"),
        ("discogs_effnet_embedding_v1", (1, 2, 3), "value_vector"),
    ],
)
def test_feature_values_round_trip_through_the_contract(
    feature_name: str, value: object, value_field: str
) -> None:
    observation = _observation("track-a", feature_name, value)

    record = observation.to_dict()
    restored = FeatureObservation.from_dict(record)

    assert set(record) & {"value_number", "value_text", "value_vector"} == {value_field}
    assert restored == observation
    assert restored.coverage_seconds == 30.0
    assert restored.feature_confidence == 0.9


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"feature_name": "tempo"}, "versioned"),
        ({"feature_source": ""}, "feature_source"),
        ({"source_version": ""}, "source_version"),
        ({"coverage_seconds": -1}, "non-negative"),
        ({"feature_confidence": 1.1}, r"\[0, 1\]"),
        ({"value": (0.0, float("nan"))}, "finite"),
    ],
)
def test_invalid_feature_records_are_rejected(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "track_id": "track-a",
        "feature_name": "tempo_bpm_estimate_v1",
        "value": 120.0,
        "feature_source": "clean_room_audio",
        "source_version": "extractor-0.1.0",
        "coverage_seconds": 30.0,
        "feature_confidence": 0.9,
    }
    arguments.update(overrides)

    with pytest.raises(FeatureRecordError, match=message):
        FeatureObservation(**arguments)  # type: ignore[arg-type]


def test_exact_selector_keeps_sources_and_versions_separate() -> None:
    clean_room = _observation("track-a", "tempo_bpm_estimate_v1", 120.0)
    acousticbrainz = FeatureObservation(
        track_id="track-a",
        feature_name="tempo_bpm_estimate_v1",
        value=121.0,
        feature_source="acousticbrainz",
        source_version="essentia-2.1-beta2",
        coverage_seconds=210.0,
        feature_confidence=0.7,
    )
    catalog = FeatureCatalog([clean_room, acousticbrainz])

    assert catalog.get("track-a", clean_room.selector) == clean_room
    assert catalog.get("track-a", acousticbrainz.selector) == acousticbrainz
    assert (
        catalog.get(
            "track-a",
            FeatureSelector(
                feature_name="tempo_bpm_estimate_v1",
                feature_source="clean_room_audio",
                source_version="extractor-9.9.9",
            ),
        )
        is None
    )


def test_duplicate_exact_observations_are_rejected() -> None:
    observation = _observation("track-a", "tempo_bpm_estimate_v1", 120.0)

    with pytest.raises(FeatureRecordError, match="Duplicate observation"):
        FeatureCatalog([observation, observation])


def test_json_lines_are_sorted_and_round_trip(tmp_path: Path) -> None:
    observations = [
        _observation("track-b", "tempo_bpm_estimate_v1", 130.0),
        _observation("track-a", "tempo_bpm_estimate_v1", 120.0),
    ]

    output_path = write_feature_observations(observations, tmp_path / "features.jsonl")
    restored = read_feature_observations(output_path)
    serialized = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert [record.track_id for record in restored] == ["track-a", "track-b"]
    assert serialized[0]["feature_source"] == "clean_room_audio"
    assert serialized[0]["source_version"] == "extractor-0.1.0"
