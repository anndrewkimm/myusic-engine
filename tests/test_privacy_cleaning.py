import json
from pathlib import Path

import pytest

from myusic_engine.ingest import HistoryRecordError, normalize_history_record
from myusic_engine.privacy import PrivacyBoundaryError, assert_privacy_safe, find_sensitive_fields


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "spotify_history_synthetic.json"


def _fixture_records() -> list[dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_track_record_is_normalized_through_an_allowlist() -> None:
    raw_record = _fixture_records()[0]

    event = normalize_history_record(
        raw_record,
        source_file="private/nested/Streaming_History_Audio_0.json",
        source_record_index=0,
    )
    cleaned = event.to_dict()

    assert cleaned["media_type"] == "track"
    assert cleaned["played_at"] == "2025-01-01T10:00:00.000Z"
    assert cleaned["platform_family"] == "other"
    assert cleaned["source_file"] == "Streaming_History_Audio_0.json"
    assert find_sensitive_fields(cleaned) == set()
    assert "conn_country" not in cleaned
    assert "platform" not in cleaned
    assert "offline_timestamp" not in cleaned


def test_sensitive_values_do_not_affect_deduplication_identity() -> None:
    first = _fixture_records()[0]
    second = dict(first)
    second.update(
        {
            "ip_addr": "203.0.113.200",
            "username": "different_synthetic_user",
            "user_agent": "DifferentSyntheticPlayer/2.0",
        }
    )

    first_event = normalize_history_record(first, source_file="first.json", source_record_index=0)
    second_event = normalize_history_record(
        second, source_file="second.json", source_record_index=99
    )

    assert first_event.event_id == second_event.event_id


def test_episode_is_classified_separately_from_music() -> None:
    event = normalize_history_record(
        _fixture_records()[2], source_file="history.json", source_record_index=2
    )

    assert event.media_type == "episode"
    assert event.track_uri is None
    assert event.episode_uri is not None


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_code"),
    [
        ("ms_played", -1, "invalid_ms_played"),
        ("ts", "2025-01-01T10:00:00", "invalid_timestamp"),
        ("skipped", "false", "invalid_boolean"),
        ("spotify_track_uri", "spotify:track:not-valid", "invalid_spotify_uri"),
    ],
)
def test_invalid_values_are_rejected_without_echoing_them(
    field_name: str, invalid_value: object, error_code: str
) -> None:
    raw_record = _fixture_records()[0]
    raw_record[field_name] = invalid_value

    with pytest.raises(HistoryRecordError) as error:
        normalize_history_record(raw_record, source_file="history.json", source_record_index=0)

    assert error.value.code == error_code
    assert str(invalid_value) not in str(error.value)


def test_privacy_audit_checks_nested_records() -> None:
    unsafe_record = {"safe": [{"nested": {"ip_addr": "192.0.2.1"}}]}

    with pytest.raises(PrivacyBoundaryError, match="ip_addr"):
        assert_privacy_safe(unsafe_record)
