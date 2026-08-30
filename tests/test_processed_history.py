from __future__ import annotations

import json
from pathlib import Path

import pytest

from myusic_engine.ingest import (
    ProcessedHistoryError,
    iter_normalized_events,
    load_history,
    read_normalized_events,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "spotify_history_synthetic.json"


def test_processed_history_round_trip_streams_cleaned_events(tmp_path: Path) -> None:
    expected = load_history(FIXTURE_PATH).events
    path = tmp_path / "listening_events.jsonl"
    path.write_text(
        "\n".join(json.dumps(event.to_dict(), sort_keys=True) for event in expected) + "\n\n",
        encoding="utf-8",
    )

    assert tuple(iter_normalized_events(path)) == expected
    assert read_normalized_events(path) == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unknown": 1}, "unknown fields"),
        ({"schema_version": 2}, "schema_version"),
        ({"media_type": "video"}, "media_type"),
        ({"event_id": ""}, "event_id"),
        ({"track_name": 123}, "track_name"),
        ({"ms_played": -1}, "ms_played"),
        ({"source_record_index": True}, "source_record_index"),
        ({"shuffle": "yes"}, "shuffle"),
    ],
)
def test_processed_history_rejects_contract_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    record = load_history(FIXTURE_PATH).events[0].to_dict()
    record.update(mutation)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ProcessedHistoryError, match=message):
        read_normalized_events(path)


@pytest.mark.parametrize("line", ["not-json", "[]"])
def test_processed_history_rejects_invalid_json_lines(tmp_path: Path, line: str) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ProcessedHistoryError):
        read_normalized_events(path)
