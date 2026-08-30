"""Data structures emitted by listening-history ingestion."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

MediaType: TypeAlias = Literal["track", "episode", "unknown"]


@dataclass(frozen=True, slots=True)
class NormalizedListeningEvent:
    """Privacy-cleaned representation of one playback event."""

    event_id: str
    played_at: str
    media_type: MediaType
    ms_played: int
    track_uri: str | None
    track_name: str | None
    artist_name: str | None
    album_name: str | None
    episode_uri: str | None
    episode_name: str | None
    show_name: str | None
    reason_start: str | None
    reason_end: str | None
    shuffle: bool | None
    skipped: bool | None
    offline: bool | None
    incognito_mode: bool | None
    platform_family: str | None
    source_file: str
    source_record_index: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record matching the versioned schema."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "played_at": self.played_at,
            "media_type": self.media_type,
            "ms_played": self.ms_played,
            "track_uri": self.track_uri,
            "track_name": self.track_name,
            "artist_name": self.artist_name,
            "album_name": self.album_name,
            "episode_uri": self.episode_uri,
            "episode_name": self.episode_name,
            "show_name": self.show_name,
            "reason_start": self.reason_start,
            "reason_end": self.reason_end,
            "shuffle": self.shuffle,
            "skipped": self.skipped,
            "offline": self.offline,
            "incognito_mode": self.incognito_mode,
            "platform_family": self.platform_family,
            "source_file": self.source_file,
            "source_record_index": self.source_record_index,
        }


@dataclass(frozen=True, slots=True)
class IngestionIssue:
    """A safe validation issue that never includes a raw field value."""

    source_file: str
    source_record_index: int
    code: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "source_record_index": self.source_record_index,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Aggregate, non-sensitive diagnostics for an ingestion run."""

    source_files: tuple[str, ...]
    records_seen: int
    records_rejected: int
    duplicate_events_removed: int
    events_written: int
    media_counts: dict[str, int]
    sensitive_fields_seen: dict[str, int]
    issues: tuple[IngestionIssue, ...]
    issues_truncated: bool
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_files": list(self.source_files),
            "records_seen": self.records_seen,
            "records_rejected": self.records_rejected,
            "duplicate_events_removed": self.duplicate_events_removed,
            "events_written": self.events_written,
            "media_counts": dict(sorted(self.media_counts.items())),
            "sensitive_field_counts": [
                {"field_name": field_name, "record_count": record_count}
                for field_name, record_count in sorted(self.sensitive_fields_seen.items())
            ],
            "issues": [issue.to_dict() for issue in self.issues],
            "issues_truncated": self.issues_truncated,
        }


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Normalized events and their run-level quality report."""

    events: tuple[NormalizedListeningEvent, ...]
    report: IngestionReport
