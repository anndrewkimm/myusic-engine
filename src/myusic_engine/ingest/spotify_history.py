"""Privacy-first ingestion for Spotify Extended Streaming History exports."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TextIO, cast
from zipfile import BadZipFile, ZipFile

from myusic_engine.ingest.models import (
    IngestionIssue,
    IngestionReport,
    IngestionResult,
    MediaType,
    NormalizedListeningEvent,
)
from myusic_engine.io import atomic_write_text
from myusic_engine.privacy import assert_privacy_safe, find_sensitive_fields

_SPOTIFY_URI_PATTERN = re.compile(r"^spotify:(track|episode):[A-Za-z0-9]{22}$")
_MAX_REPORTED_ISSUES = 100

_ACCOUNT_DATA_FIELD_ALIASES = {
    "endTime": "ts",
    "msPlayed": "ms_played",
    "trackName": "master_metadata_track_name",
    "artistName": "master_metadata_album_artist_name",
    "episodeName": "episode_name",
    "podcastName": "episode_show_name",
}


class HistoryIngestionError(ValueError):
    """Base error for invalid inputs or records."""


class HistoryInputError(HistoryIngestionError):
    """Raised when an export source cannot be read."""


class HistoryRecordError(HistoryIngestionError):
    """Raised when a playback record violates the raw data contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _canonicalize_account_data_record(record: Mapping[str, object]) -> Mapping[str, object]:
    """Map Spotify's compact Account Data fields onto the extended-history names.

    Account Data timestamps omit an offset, but Spotify defines ``endTime`` as UTC. Other
    timestamp fields still have to carry an explicit offset and are validated unchanged.
    """

    if not any(field_name in record for field_name in _ACCOUNT_DATA_FIELD_ALIASES):
        return record

    canonical = dict(record)
    for account_field, canonical_field in _ACCOUNT_DATA_FIELD_ALIASES.items():
        if canonical_field in canonical or account_field not in record:
            continue
        value = record[account_field]
        if account_field == "endTime" and isinstance(value, str):
            candidate = value.strip()
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                pass
            else:
                if parsed.tzinfo is None:
                    value = parsed.replace(tzinfo=UTC).isoformat()
        canonical[canonical_field] = value
    return canonical


def _optional_string(record: Mapping[str, object], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HistoryRecordError("invalid_string", f"Field {field_name!r} must be text or null")
    stripped = value.strip()
    return stripped or None


def _optional_boolean(record: Mapping[str, object], field_name: str) -> bool | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise HistoryRecordError("invalid_boolean", f"Field {field_name!r} must be boolean or null")
    return value


def _milliseconds_played(record: Mapping[str, object]) -> int:
    value = record.get("ms_played")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HistoryRecordError(
            "invalid_ms_played", "Field 'ms_played' must be a non-negative integer"
        )
    return value


def _utc_timestamp(record: Mapping[str, object]) -> str:
    value = record.get("ts")
    if not isinstance(value, str):
        raise HistoryRecordError("invalid_timestamp", "Field 'ts' must be timestamp text")
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise HistoryRecordError("invalid_timestamp", "Field 'ts' is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HistoryRecordError("invalid_timestamp", "Field 'ts' must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _spotify_uri(record: Mapping[str, object], field_name: str, expected_type: str) -> str | None:
    uri = _optional_string(record, field_name)
    if uri is None:
        return None
    match = _SPOTIFY_URI_PATTERN.fullmatch(uri)
    if match is None or match.group(1) != expected_type:
        raise HistoryRecordError(
            "invalid_spotify_uri", f"Field {field_name!r} is not a supported Spotify URI"
        )
    return uri


def _platform_family(platform: str | None) -> str | None:
    if platform is None:
        return None
    normalized = platform.casefold()
    families = (
        (("iphone", "ipad", "ios"), "ios"),
        (("android",), "android"),
        (("windows",), "windows"),
        (("macintosh", "macos", "os x"), "macos"),
        (("linux",), "linux"),
        (("browser", "web player", "web_player"), "web"),
        (("television", "smart tv", "smart_tv"), "tv"),
        (("playstation", "xbox", "game console"), "game_console"),
    )
    for markers, family in families:
        if any(marker in normalized for marker in markers):
            return family
    return "other"


def _media_type(
    *,
    track_uri: str | None,
    track_name: str | None,
    artist_name: str | None,
    album_name: str | None,
    episode_uri: str | None,
    episode_name: str | None,
    show_name: str | None,
) -> MediaType:
    has_track = track_uri is not None or any((track_name, artist_name, album_name))
    has_episode = episode_uri is not None or any((episode_name, show_name))
    if has_track and has_episode:
        raise HistoryRecordError(
            "ambiguous_media_type", "Record contains both track and episode metadata"
        )
    if has_track:
        return "track"
    if has_episode:
        return "episode"
    return "unknown"


def _event_id(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_history_record(
    record: Mapping[str, object], *, source_file: str, source_record_index: int
) -> NormalizedListeningEvent:
    """Normalize one raw record through an explicit privacy allowlist."""

    record = _canonicalize_account_data_record(record)
    played_at = _utc_timestamp(record)
    ms_played = _milliseconds_played(record)
    track_uri = _spotify_uri(record, "spotify_track_uri", "track")
    episode_uri = _spotify_uri(record, "spotify_episode_uri", "episode")
    track_name = _optional_string(record, "master_metadata_track_name")
    artist_name = _optional_string(record, "master_metadata_album_artist_name")
    album_name = _optional_string(record, "master_metadata_album_album_name")
    episode_name = _optional_string(record, "episode_name")
    show_name = _optional_string(record, "episode_show_name")
    media_type = _media_type(
        track_uri=track_uri,
        track_name=track_name,
        artist_name=artist_name,
        album_name=album_name,
        episode_uri=episode_uri,
        episode_name=episode_name,
        show_name=show_name,
    )
    reason_start = _optional_string(record, "reason_start")
    reason_end = _optional_string(record, "reason_end")
    shuffle = _optional_boolean(record, "shuffle")
    skipped = _optional_boolean(record, "skipped")
    offline = _optional_boolean(record, "offline")
    incognito_mode = _optional_boolean(record, "incognito_mode")

    identity = {
        "played_at": played_at,
        "media_type": media_type,
        "media_identity": track_uri
        or episode_uri
        or [track_name, artist_name, album_name, episode_name, show_name],
        "ms_played": ms_played,
        "reason_start": reason_start,
        "reason_end": reason_end,
        "shuffle": shuffle,
        "skipped": skipped,
        "offline": offline,
        "incognito_mode": incognito_mode,
    }
    event = NormalizedListeningEvent(
        event_id=_event_id(identity),
        played_at=played_at,
        media_type=media_type,
        ms_played=ms_played,
        track_uri=track_uri,
        track_name=track_name,
        artist_name=artist_name,
        album_name=album_name,
        episode_uri=episode_uri,
        episode_name=episode_name,
        show_name=show_name,
        reason_start=reason_start,
        reason_end=reason_end,
        shuffle=shuffle,
        skipped=skipped,
        offline=offline,
        incognito_mode=incognito_mode,
        platform_family=_platform_family(_optional_string(record, "platform")),
        source_file=PurePosixPath(source_file.replace("\\", "/")).name,
        source_record_index=source_record_index,
    )
    assert_privacy_safe(event.to_dict())
    return event


def _looks_like_history_file(name: str) -> bool:
    lowered = PurePosixPath(name).name.casefold()
    return lowered.endswith(".json") and (
        lowered.startswith("streaming_history_audio_")
        or lowered.startswith("streaminghistory_music_")
        or lowered.startswith("streaminghistory_podcast_")
        or lowered.startswith("endsong")
    )


def _records_from_stream(stream: TextIO, source_label: str) -> Sequence[object]:
    try:
        payload = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoryInputError(f"Could not decode JSON in {source_label!r}") from exc
    if not isinstance(payload, list):
        raise HistoryInputError(f"History file {source_label!r} must contain a JSON array")
    return cast(Sequence[object], payload)


def _source_batches(source: Path) -> Iterator[tuple[str, Sequence[object]]]:
    if not source.exists():
        raise HistoryInputError("History source does not exist")

    if source.is_dir():
        candidates = sorted(
            (path for path in source.rglob("*.json") if _looks_like_history_file(path.name)),
            key=lambda path: str(path).casefold(),
        )
        if not candidates:
            raise HistoryInputError("Directory contains no recognized Spotify history JSON files")
        for candidate in candidates:
            with candidate.open("r", encoding="utf-8-sig") as stream:
                yield candidate.name, _records_from_stream(stream, candidate.name)
        return

    if source.suffix.casefold() == ".zip":
        try:
            with ZipFile(source) as archive:
                members = sorted(
                    (
                        member
                        for member in archive.infolist()
                        if not member.is_dir() and _looks_like_history_file(member.filename)
                    ),
                    key=lambda member: member.filename.casefold(),
                )
                if not members:
                    raise HistoryInputError("ZIP contains no recognized Spotify history JSON files")
                for member in members:
                    source_label = PurePosixPath(member.filename).name
                    with (
                        archive.open(member) as binary_stream,
                        io.TextIOWrapper(binary_stream, encoding="utf-8-sig") as stream,
                    ):
                        yield source_label, _records_from_stream(stream, source_label)
        except BadZipFile as exc:
            raise HistoryInputError("History source is not a valid ZIP archive") from exc
        return

    if source.suffix.casefold() != ".json":
        raise HistoryInputError("History source must be a JSON file, directory, or ZIP archive")
    with source.open("r", encoding="utf-8-sig") as stream:
        yield source.name, _records_from_stream(stream, source.name)


def load_history(source: str | Path, *, strict: bool = False) -> IngestionResult:
    """Load, normalize, classify, and deduplicate a Spotify history source."""

    events_by_id: dict[str, NormalizedListeningEvent] = {}
    source_files: list[str] = []
    sensitive_counts: Counter[str] = Counter()
    issues: list[IngestionIssue] = []
    records_seen = 0
    records_rejected = 0
    duplicate_events_removed = 0
    issues_truncated = False

    for source_file, records in _source_batches(Path(source)):
        source_files.append(source_file)
        for source_record_index, raw_record in enumerate(records):
            records_seen += 1
            if isinstance(raw_record, Mapping):
                for field_name in find_sensitive_fields(raw_record):
                    sensitive_counts[field_name] += 1
            try:
                if not isinstance(raw_record, Mapping):
                    raise HistoryRecordError("invalid_record", "History record must be an object")
                record = cast(Mapping[str, object], raw_record)
                event = normalize_history_record(
                    record,
                    source_file=source_file,
                    source_record_index=source_record_index,
                )
            except HistoryRecordError as exc:
                if strict:
                    raise HistoryRecordError(
                        exc.code,
                        f"{source_file} record {source_record_index}: {exc}",
                    ) from exc
                records_rejected += 1
                if len(issues) < _MAX_REPORTED_ISSUES:
                    issues.append(
                        IngestionIssue(
                            source_file=source_file,
                            source_record_index=source_record_index,
                            code=exc.code,
                            message=str(exc),
                        )
                    )
                else:
                    issues_truncated = True
                continue

            if event.event_id in events_by_id:
                duplicate_events_removed += 1
                continue
            events_by_id[event.event_id] = event

    events = tuple(
        sorted(events_by_id.values(), key=lambda event: (event.played_at, event.event_id))
    )
    media_counts = Counter(event.media_type for event in events)
    report = IngestionReport(
        source_files=tuple(source_files),
        records_seen=records_seen,
        records_rejected=records_rejected,
        duplicate_events_removed=duplicate_events_removed,
        events_written=len(events),
        media_counts={media_type: count for media_type, count in media_counts.items()},
        sensitive_fields_seen=dict(sensitive_counts),
        issues=tuple(issues),
        issues_truncated=issues_truncated,
    )
    return IngestionResult(events=events, report=report)


def write_ingestion_result(
    result: IngestionResult, output_directory: str | Path
) -> tuple[Path, Path]:
    """Atomically write normalized JSON Lines and a non-sensitive quality report."""

    output_path = Path(output_directory)
    events_path = output_path / "listening_events.jsonl"
    report_path = output_path / "ingestion_report.json"

    serialized_events: list[str] = []
    for event in result.events:
        record = event.to_dict()
        assert_privacy_safe(record)
        serialized_events.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    events_content = "\n".join(serialized_events)
    if events_content:
        events_content += "\n"
    report_record = result.report.to_dict()
    assert_privacy_safe(report_record)
    report_content = json.dumps(report_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    atomic_write_text(events_path, events_content)
    atomic_write_text(report_path, report_content)
    return events_path, report_path


def prepare_history(
    source: str | Path, output_directory: str | Path, *, strict: bool = False
) -> IngestionResult:
    """Run the complete local history preparation step and write its outputs."""

    result = load_history(source, strict=strict)
    write_ingestion_result(result, output_directory)
    return result
