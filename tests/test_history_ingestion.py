import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from myusic_engine.cli import main
from myusic_engine.ingest import HistoryRecordError, load_history, prepare_history


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "spotify_history_synthetic.json"


def test_json_file_is_classified_and_audited() -> None:
    result = load_history(FIXTURE_PATH)

    assert len(result.events) == 3
    assert result.report.records_seen == 3
    assert result.report.records_rejected == 0
    assert result.report.duplicate_events_removed == 0
    assert result.report.media_counts == {"track": 2, "episode": 1}
    assert result.report.sensitive_fields_seen == {
        "ip_addr": 3,
        "user_agent": 3,
        "username": 3,
    }


def test_zip_reads_only_history_files_and_deduplicates_overlaps(tmp_path: Path) -> None:
    archive_path = tmp_path / "private-export.zip"
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Spotify Account Data/Streaming_History_Audio_0.json", fixture_text)
        archive.writestr("Spotify Account Data/Streaming_History_Audio_1.json", fixture_text)
        archive.writestr("Spotify Account Data/Userdata.json", '{"username": "do-not-read"}')

    result = load_history(archive_path)

    assert len(result.events) == 3
    assert result.report.records_seen == 6
    assert result.report.duplicate_events_removed == 3
    assert result.report.source_files == (
        "Streaming_History_Audio_0.json",
        "Streaming_History_Audio_1.json",
    )


def test_invalid_records_are_reported_safely_or_raise_in_strict_mode(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps([{"ts": "not-a-timestamp", "ms_played": 100, "ip_addr": "192.0.2.9"}]),
        encoding="utf-8",
    )

    result = load_history(invalid_path)

    assert result.events == ()
    assert result.report.records_rejected == 1
    assert result.report.issues[0].code == "invalid_timestamp"
    assert "192.0.2.9" not in json.dumps(result.report.to_dict())

    with pytest.raises(HistoryRecordError, match="record 0"):
        load_history(invalid_path, strict=True)


def test_prepare_history_writes_only_cleaned_events_and_aggregate_report(tmp_path: Path) -> None:
    output_directory = tmp_path / "processed"

    result = prepare_history(FIXTURE_PATH, output_directory)

    event_lines = (output_directory / "listening_events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    report = json.loads((output_directory / "ingestion_report.json").read_text(encoding="utf-8"))
    serialized_output = "\n".join(event_lines) + json.dumps(report)

    assert len(event_lines) == result.report.events_written == 3
    assert report["media_counts"] == {"episode": 1, "track": 2}
    assert report["sensitive_field_counts"] == [
        {"field_name": "ip_addr", "record_count": 3},
        {"field_name": "user_agent", "record_count": 3},
        {"field_name": "username", "record_count": 3},
    ]
    assert "192.0.2.10" not in serialized_output
    assert "synthetic_user" not in serialized_output
    assert "SyntheticPlayer" not in serialized_output


def test_cli_prepares_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_directory = tmp_path / "cli-output"

    exit_code = main(
        [
            "prepare-history",
            str(FIXTURE_PATH),
            "--output-dir",
            str(output_directory),
        ]
    )

    assert exit_code == 0
    assert (output_directory / "listening_events.jsonl").exists()
    assert (output_directory / "user_track_affinity.jsonl").exists()
    output = capsys.readouterr().out
    assert "Prepared 3 events" in output
    assert "Aggregated 2 track affinity records" in output
