"""Strict MusicBrainz mapping of URI-backed history tracks through a provider boundary."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from myusic_engine.io import atomic_write_text
from myusic_engine.matching.models import IdentityInputError, TrackQuery
from myusic_engine.matching.resolver import normalize_metadata
from myusic_engine.privacy import assert_privacy_safe
from myusic_engine.providers import ListenBrainzMapping, MusicBrainzMapper

ExternalMatchStatus = Literal["exact", "fuzzy", "ambiguous", "unmatched"]
ExternalMatchMethod = Literal[
    "exact_title_artist_release",
    "exact_title_artist",
    "provider_fuzzy",
    "provider_unmatched",
]
_MATCH_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "source_track_id",
        "source_track_uri",
        "track_name",
        "artist_name",
        "album_name",
        "play_count",
        "total_ms_played",
        "match_status",
        "match_method",
        "recording_mbid",
        "release_mbid",
        "artist_mbids",
        "provider",
        "provider_confidence",
        "match_confidence",
        "mapped_recording_name",
        "mapped_artist_name",
        "mapped_release_name",
        "review_required",
    }
)


@dataclass(frozen=True, slots=True)
class ExternalIdentityPolicy:
    """Precision-first boundary for provider-suggested MusicBrainz recording matches."""

    policy_version: str = "listenbrainz_mapper_exact_metadata_v1"
    minimum_provider_confidence: float = 0.90
    review_sample_per_status: int = 20

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise IdentityInputError("External identity policy_version must be non-empty")
        if not math.isfinite(self.minimum_provider_confidence) or not (
            0 <= self.minimum_provider_confidence <= 1
        ):
            raise IdentityInputError("External minimum_provider_confidence must be in [0, 1]")
        if (
            isinstance(self.review_sample_per_status, bool)
            or not isinstance(self.review_sample_per_status, int)
            or self.review_sample_per_status < 0
        ):
            raise IdentityInputError(
                "External review_sample_per_status must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class ExternalIdentityMatch:
    """One auditable Spotify-history identity to MusicBrainz recording decision."""

    policy_version: str
    source_track_id: str
    source_track_uri: str | None
    track_name: str | None
    artist_name: str | None
    album_name: str | None
    play_count: int
    total_ms_played: int
    match_status: ExternalMatchStatus
    match_method: ExternalMatchMethod
    recording_mbid: str | None
    release_mbid: str | None
    artist_mbids: tuple[str, ...]
    provider: str
    provider_confidence: float | None
    match_confidence: float | None
    mapped_recording_name: str | None
    mapped_artist_name: str | None
    mapped_release_name: str | None
    review_required: bool
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "source_track_id": self.source_track_id,
            "source_track_uri": self.source_track_uri,
            "track_name": self.track_name,
            "artist_name": self.artist_name,
            "album_name": self.album_name,
            "play_count": self.play_count,
            "total_ms_played": self.total_ms_played,
            "match_status": self.match_status,
            "match_method": self.match_method,
            "recording_mbid": self.recording_mbid,
            "release_mbid": self.release_mbid,
            "artist_mbids": list(self.artist_mbids),
            "provider": self.provider,
            "provider_confidence": self.provider_confidence,
            "match_confidence": self.match_confidence,
            "mapped_recording_name": self.mapped_recording_name,
            "mapped_artist_name": self.mapped_artist_name,
            "mapped_release_name": self.mapped_release_name,
            "review_required": self.review_required,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> ExternalIdentityMatch:
        unknown = set(record) - _MATCH_FIELDS
        if unknown:
            raise IdentityInputError("External match input contains unknown fields")
        raw_artist_mbids = record.get("artist_mbids")
        if not isinstance(raw_artist_mbids, list) or not all(
            isinstance(value, str) for value in raw_artist_mbids
        ):
            raise IdentityInputError("External match artist_mbids must be an array of text")
        status = record.get("match_status")
        if status not in {"exact", "fuzzy", "ambiguous", "unmatched"}:
            raise IdentityInputError("External match has an unsupported status")
        method = record.get("match_method")
        if method not in {
            "exact_title_artist_release",
            "exact_title_artist",
            "provider_fuzzy",
            "provider_unmatched",
        }:
            raise IdentityInputError("External match has an unsupported method")
        return cls(
            schema_version=_integer(record, "schema_version"),
            policy_version=_text(record, "policy_version"),
            source_track_id=_text(record, "source_track_id"),
            source_track_uri=_optional_text(record, "source_track_uri"),
            track_name=_optional_text(record, "track_name"),
            artist_name=_optional_text(record, "artist_name"),
            album_name=_optional_text(record, "album_name"),
            play_count=_integer(record, "play_count"),
            total_ms_played=_integer(record, "total_ms_played"),
            match_status=cast(ExternalMatchStatus, status),
            match_method=cast(ExternalMatchMethod, method),
            recording_mbid=_optional_text(record, "recording_mbid"),
            release_mbid=_optional_text(record, "release_mbid"),
            artist_mbids=tuple(cast(list[str], raw_artist_mbids)),
            provider=_text(record, "provider"),
            provider_confidence=_optional_number(record, "provider_confidence"),
            match_confidence=_optional_number(record, "match_confidence"),
            mapped_recording_name=_optional_text(record, "mapped_recording_name"),
            mapped_artist_name=_optional_text(record, "mapped_artist_name"),
            mapped_release_name=_optional_text(record, "mapped_release_name"),
            review_required=_boolean(record, "review_required"),
        )


@dataclass(frozen=True, slots=True)
class ExternalIdentityReport:
    """Aggregate identity and behavior-weighted coverage for a provider mapping run."""

    policy_version: str
    provider: str
    queries_available: int
    queries_processed: int
    exact_count: int
    review_required_count: int
    status_counts: dict[str, int]
    status_rates: dict[str, float]
    history_play_count: int
    exact_play_count: int
    exact_play_rate: float
    history_ms_played: int
    exact_ms_played: int
    exact_ms_played_rate: float
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "provider": self.provider,
            "queries_available": self.queries_available,
            "queries_processed": self.queries_processed,
            "exact_count": self.exact_count,
            "review_required_count": self.review_required_count,
            "status_counts": dict(sorted(self.status_counts.items())),
            "status_rates": dict(sorted(self.status_rates.items())),
            "history_play_count": self.history_play_count,
            "exact_play_count": self.exact_play_count,
            "exact_play_rate": self.exact_play_rate,
            "history_ms_played": self.history_ms_played,
            "exact_ms_played": self.exact_ms_played,
            "exact_ms_played_rate": self.exact_ms_played_rate,
        }


@dataclass(frozen=True, slots=True)
class ExternalIdentityResult:
    matches: tuple[ExternalIdentityMatch, ...]
    report: ExternalIdentityReport


def _text(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise IdentityInputError(f"External match field {field_name} must be text")
    return value.strip()


def _optional_text(record: Mapping[str, object], field_name: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise IdentityInputError(f"External match field {field_name} must be null or text")
    return value.strip()


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IdentityInputError(f"External match field {field_name} must be non-negative")
    return value


def _optional_number(record: Mapping[str, object], field_name: str) -> float | None:
    value = record.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IdentityInputError(f"External match field {field_name} must be null or numeric")
    number = float(value)
    if not math.isfinite(number):
        raise IdentityInputError(f"External match field {field_name} must be finite")
    return number


def _boolean(record: Mapping[str, object], field_name: str) -> bool:
    value = record.get(field_name)
    if not isinstance(value, bool):
        raise IdentityInputError(f"External match field {field_name} must be boolean")
    return value


def _match_query(
    query: TrackQuery,
    mapping: ListenBrainzMapping | None,
    policy: ExternalIdentityPolicy,
) -> ExternalIdentityMatch:
    if mapping is None:
        return ExternalIdentityMatch(
            policy_version=policy.policy_version,
            source_track_id=query.source_track_id,
            source_track_uri=query.track_uri,
            track_name=query.track_name,
            artist_name=query.artist_name,
            album_name=query.album_name,
            play_count=query.play_count,
            total_ms_played=query.total_ms_played,
            provider="listenbrainz_musicbrainz_mapper",
            match_status="unmatched",
            match_method="provider_unmatched",
            recording_mbid=None,
            release_mbid=None,
            artist_mbids=(),
            provider_confidence=None,
            match_confidence=None,
            mapped_recording_name=None,
            mapped_artist_name=None,
            mapped_release_name=None,
            review_required=True,
        )
    title_exact = normalize_metadata(query.track_name) == normalize_metadata(
        mapping.recording_name
    )
    artist_exact = normalize_metadata(query.artist_name) == normalize_metadata(
        mapping.artist_credit_name
    )
    release_exact = (
        query.album_name is not None
        and mapping.release_name is not None
        and normalize_metadata(query.album_name) == normalize_metadata(mapping.release_name)
    )
    confident = mapping.confidence >= policy.minimum_provider_confidence
    if title_exact and artist_exact and release_exact and confident:
        status: ExternalMatchStatus = "exact"
        method: ExternalMatchMethod = "exact_title_artist_release"
        accepted_mbid: str | None = mapping.recording_mbid
        review_required = False
    elif title_exact and artist_exact and query.album_name is None and confident:
        status = "exact"
        method = "exact_title_artist"
        accepted_mbid = mapping.recording_mbid
        review_required = False
    else:
        status = "fuzzy"
        method = "provider_fuzzy"
        accepted_mbid = None
        review_required = True
    return ExternalIdentityMatch(
        policy_version=policy.policy_version,
        source_track_id=query.source_track_id,
        source_track_uri=query.track_uri,
        track_name=query.track_name,
        artist_name=query.artist_name,
        album_name=query.album_name,
        play_count=query.play_count,
        total_ms_played=query.total_ms_played,
        provider="listenbrainz_musicbrainz_mapper",
        match_status=status,
        match_method=method,
        recording_mbid=accepted_mbid,
        release_mbid=mapping.release_mbid if accepted_mbid else None,
        artist_mbids=mapping.artist_mbids if accepted_mbid else (),
        provider_confidence=mapping.confidence,
        match_confidence=mapping.confidence if accepted_mbid else None,
        mapped_recording_name=mapping.recording_name,
        mapped_artist_name=mapping.artist_credit_name,
        mapped_release_name=mapping.release_name,
        review_required=review_required,
    )


def resolve_external_identities(
    queries: Iterable[TrackQuery],
    mapper: MusicBrainzMapper,
    *,
    policy: ExternalIdentityPolicy | None = None,
    maximum_tracks: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ExternalIdentityResult:
    """Map highest-value history tracks and accept only exact metadata evidence."""

    active_policy = policy or ExternalIdentityPolicy()
    ordered_queries = sorted(
        queries,
        key=lambda item: (-item.play_count, -item.total_ms_played, item.source_track_id),
    )
    if maximum_tracks is not None:
        if isinstance(maximum_tracks, bool) or not isinstance(maximum_tracks, int):
            raise IdentityInputError("maximum_tracks must be an integer")
        if maximum_tracks < 1:
            raise IdentityInputError("maximum_tracks must be positive")
        selected_queries = ordered_queries[:maximum_tracks]
    else:
        selected_queries = ordered_queries
    matches: list[ExternalIdentityMatch] = []
    for index, query in enumerate(selected_queries, start=1):
        mapping = (
            mapper.lookup(
                artist_name=query.artist_name,
                recording_name=query.track_name,
                release_name=query.album_name,
            )
            if query.artist_name and query.track_name
            else None
        )
        match = _match_query(query, mapping, active_policy)
        assert_privacy_safe(match.to_dict())
        matches.append(match)
        if progress is not None:
            progress(index, len(selected_queries))

    counts = Counter(match.match_status for match in matches)
    status_counts: dict[str, int] = {
        "exact": counts["exact"],
        "fuzzy": counts["fuzzy"],
        "ambiguous": counts["ambiguous"],
        "unmatched": counts["unmatched"],
    }
    exact_matches = [match for match in matches if match.match_status == "exact"]
    history_play_count = sum(match.play_count for match in matches)
    exact_play_count = sum(match.play_count for match in exact_matches)
    history_ms_played = sum(match.total_ms_played for match in matches)
    exact_ms_played = sum(match.total_ms_played for match in exact_matches)
    processed = len(matches)
    report = ExternalIdentityReport(
        policy_version=active_policy.policy_version,
        provider="listenbrainz_musicbrainz_mapper",
        queries_available=len(ordered_queries),
        queries_processed=processed,
        exact_count=len(exact_matches),
        review_required_count=sum(match.review_required for match in matches),
        status_counts=status_counts,
        status_rates={
            status: round(count / processed, 6) if processed else 0.0
            for status, count in status_counts.items()
        },
        history_play_count=history_play_count,
        exact_play_count=exact_play_count,
        exact_play_rate=(
            round(exact_play_count / history_play_count, 6) if history_play_count else 0.0
        ),
        history_ms_played=history_ms_played,
        exact_ms_played=exact_ms_played,
        exact_ms_played_rate=(
            round(exact_ms_played / history_ms_played, 6) if history_ms_played else 0.0
        ),
    )
    assert_privacy_safe(report.to_dict())
    return ExternalIdentityResult(matches=tuple(matches), report=report)


def external_review_sample(
    matches: Iterable[ExternalIdentityMatch], *, sample_size_per_status: int
) -> tuple[ExternalIdentityMatch, ...]:
    if sample_size_per_status < 0:
        raise IdentityInputError("sample_size_per_status must be non-negative")
    grouped: dict[str, list[ExternalIdentityMatch]] = {
        status: [] for status in ("exact", "fuzzy", "ambiguous", "unmatched")
    }
    for match in matches:
        grouped[match.match_status].append(match)
    sampled: list[ExternalIdentityMatch] = []
    for status in ("exact", "fuzzy", "ambiguous", "unmatched"):
        ordered = sorted(
            grouped[status],
            key=lambda item: hashlib.sha256(item.source_track_id.encode("utf-8")).hexdigest(),
        )
        sampled.extend(ordered[:sample_size_per_status])
    return tuple(sampled)


def _json_lines(matches: Iterable[ExternalIdentityMatch]) -> str:
    rows = [
        json.dumps(match.to_dict(), ensure_ascii=False, sort_keys=True) for match in matches
    ]
    return "\n".join(rows) + ("\n" if rows else "")


def write_external_identity_resolution(
    result: ExternalIdentityResult,
    output_dir: str | Path,
    *,
    review_sample_per_status: int,
) -> tuple[Path, Path, Path]:
    destination = Path(output_dir)
    matches_path = atomic_write_text(
        destination / "external_identity_matches.jsonl", _json_lines(result.matches)
    )
    report_path = atomic_write_text(
        destination / "external_identity_report.json",
        json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    review_path = atomic_write_text(
        destination / "external_identity_review_sample.jsonl",
        _json_lines(
            external_review_sample(
                result.matches, sample_size_per_status=review_sample_per_status
            )
        ),
    )
    return matches_path, report_path, review_path


def read_external_identity_matches(path: str | Path) -> tuple[ExternalIdentityMatch, ...]:
    matches: list[ExternalIdentityMatch] = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IdentityInputError(
                    f"External identity line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise IdentityInputError(
                    f"External identity line {line_number} must be an object"
                )
            matches.append(
                ExternalIdentityMatch.from_dict(cast(Mapping[str, object], payload))
            )
    return tuple(matches)
