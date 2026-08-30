"""Strict streaming reader for privacy-cleaned normalized listening events."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

from myusic_engine.ingest.models import MediaType, NormalizedListeningEvent

_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "played_at",
        "media_type",
        "ms_played",
        "track_uri",
        "track_name",
        "artist_name",
        "album_name",
        "episode_uri",
        "episode_name",
        "show_name",
        "reason_start",
        "reason_end",
        "shuffle",
        "skipped",
        "offline",
        "incognito_mode",
        "platform_family",
        "source_file",
        "source_record_index",
    }
)


class ProcessedHistoryError(ValueError):
    """Raised when a cleaned event table no longer matches its strict contract."""


def _text(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ProcessedHistoryError(f"Cleaned event field {field_name} must be text")
    return value


def _optional_text(record: Mapping[str, object], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProcessedHistoryError(f"Cleaned event field {field_name} must be null or text")
    return value


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProcessedHistoryError(
            f"Cleaned event field {field_name} must be a non-negative integer"
        )
    return value


def _optional_boolean(record: Mapping[str, object], field_name: str) -> bool | None:
    value = record.get(field_name)
    if value is not None and not isinstance(value, bool):
        raise ProcessedHistoryError(
            f"Cleaned event field {field_name} must be null or boolean"
        )
    return value


def _event_from_record(record: Mapping[str, object]) -> NormalizedListeningEvent:
    unknown = set(record) - _EVENT_FIELDS
    if unknown:
        raise ProcessedHistoryError("Cleaned event contains unknown fields")
    if record.get("schema_version") != 1:
        raise ProcessedHistoryError("Cleaned event schema_version must be 1")
    media_type = record.get("media_type")
    if media_type not in {"track", "episode", "unknown"}:
        raise ProcessedHistoryError("Cleaned event has an unsupported media_type")
    return NormalizedListeningEvent(
        schema_version=1,
        event_id=_text(record, "event_id"),
        played_at=_text(record, "played_at"),
        media_type=cast(MediaType, media_type),
        ms_played=_integer(record, "ms_played"),
        track_uri=_optional_text(record, "track_uri"),
        track_name=_optional_text(record, "track_name"),
        artist_name=_optional_text(record, "artist_name"),
        album_name=_optional_text(record, "album_name"),
        episode_uri=_optional_text(record, "episode_uri"),
        episode_name=_optional_text(record, "episode_name"),
        show_name=_optional_text(record, "show_name"),
        reason_start=_optional_text(record, "reason_start"),
        reason_end=_optional_text(record, "reason_end"),
        shuffle=_optional_boolean(record, "shuffle"),
        skipped=_optional_boolean(record, "skipped"),
        offline=_optional_boolean(record, "offline"),
        incognito_mode=_optional_boolean(record, "incognito_mode"),
        platform_family=_optional_text(record, "platform_family"),
        source_file=_text(record, "source_file"),
        source_record_index=_integer(record, "source_record_index"),
    )


def iter_normalized_events(path: str | Path) -> Iterator[NormalizedListeningEvent]:
    """Yield validated normalized events without loading the complete private table."""

    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProcessedHistoryError(
                    f"Cleaned event line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ProcessedHistoryError(
                    f"Cleaned event line {line_number} must be an object"
                )
            yield _event_from_record(cast(Mapping[str, object], payload))


def read_normalized_events(path: str | Path) -> tuple[NormalizedListeningEvent, ...]:
    """Read all validated normalized events when an in-memory stage requires sorting."""

    return tuple(iter_normalized_events(path))
