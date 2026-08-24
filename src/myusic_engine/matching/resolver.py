"""Deterministic metadata matching with explicit ambiguity and review states."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

import yaml

from myusic_engine.io import atomic_write_text
from myusic_engine.matching.models import (
    CatalogLoadResult,
    CatalogTrack,
    IdentityInputError,
    IdentityMatch,
    IdentityResolutionError,
    IdentityResolutionReport,
    IdentityResolutionResult,
    IdentitySource,
    MatchCandidate,
    MatchMethod,
    MatchStatus,
    TrackQuery,
)
from myusic_engine.privacy import assert_privacy_safe

_TRACK_URI_PATTERN = re.compile(r"^spotify:track:[A-Za-z0-9]{22}$")
_MATCH_STATUSES: tuple[MatchStatus, ...] = ("exact", "fuzzy", "ambiguous", "unmatched")
_POLICY_FIELDS = frozenset(
    {
        "policy_version",
        "title_artist_exact_confidence",
        "fuzzy_title_threshold",
        "fuzzy_album_weight",
        "fuzzy_min_margin",
        "max_candidates",
        "review_sample_per_status",
    }
)


@dataclass(frozen=True, slots=True)
class IdentityPolicy:
    """Versioned thresholds governing automatic and review-only matches."""

    policy_version: str = "offline_spotify_account_catalog_v1"
    title_artist_exact_confidence: float = 0.90
    fuzzy_title_threshold: float = 0.86
    fuzzy_album_weight: float = 0.15
    fuzzy_min_margin: float = 0.04
    max_candidates: int = 5
    review_sample_per_status: int = 20

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise IdentityResolutionError("policy_version must be non-empty")
        for field_name in (
            "title_artist_exact_confidence",
            "fuzzy_title_threshold",
            "fuzzy_album_weight",
            "fuzzy_min_margin",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise IdentityResolutionError(f"{field_name} must be finite and in [0, 1]")
        if self.fuzzy_title_threshold == 0:
            raise IdentityResolutionError("fuzzy_title_threshold must be positive")
        if self.max_candidates < 2:
            raise IdentityResolutionError("max_candidates must be at least 2")
        if self.review_sample_per_status < 0:
            raise IdentityResolutionError("review_sample_per_status must be non-negative")


def _number(section: Mapping[str, object], field_name: str, default: float) -> float:
    value = section.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IdentityResolutionError(f"Identity policy field {field_name!r} must be numeric")
    return float(value)


def _integer(section: Mapping[str, object], field_name: str, default: int) -> int:
    value = section.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IdentityResolutionError(f"Identity policy field {field_name!r} must be an integer")
    return value


def load_identity_policy(path: str | Path) -> IdentityPolicy:
    """Load and validate a versioned identity-resolution YAML configuration."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise IdentityResolutionError("Could not read identity policy configuration") from exc
    if not isinstance(payload, Mapping):
        raise IdentityResolutionError("Identity policy configuration must be an object")
    if payload.get("schema_version") != 1:
        raise IdentityResolutionError("Identity policy schema_version must be 1")
    raw_section = payload.get("identity_resolution")
    if not isinstance(raw_section, Mapping):
        raise IdentityResolutionError(
            "Identity policy configuration must contain an identity_resolution object"
        )
    section = cast(Mapping[str, object], raw_section)
    unknown_fields = {str(key) for key in section if key not in _POLICY_FIELDS}
    if unknown_fields:
        unknown_list = ", ".join(sorted(unknown_fields))
        raise IdentityResolutionError(f"Unknown identity policy fields: {unknown_list}")

    defaults = IdentityPolicy()
    policy_version = section.get("policy_version", defaults.policy_version)
    if not isinstance(policy_version, str):
        raise IdentityResolutionError("Identity policy field 'policy_version' must be text")
    return replace(
        defaults,
        policy_version=policy_version,
        title_artist_exact_confidence=_number(
            section,
            "title_artist_exact_confidence",
            defaults.title_artist_exact_confidence,
        ),
        fuzzy_title_threshold=_number(
            section, "fuzzy_title_threshold", defaults.fuzzy_title_threshold
        ),
        fuzzy_album_weight=_number(section, "fuzzy_album_weight", defaults.fuzzy_album_weight),
        fuzzy_min_margin=_number(section, "fuzzy_min_margin", defaults.fuzzy_min_margin),
        max_candidates=_integer(section, "max_candidates", defaults.max_candidates),
        review_sample_per_status=_integer(
            section, "review_sample_per_status", defaults.review_sample_per_status
        ),
    )


def _optional_text(
    record: Mapping[str, object], field_name: str, *, line_number: int
) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise IdentityInputError(
            f"Affinity line {line_number} field {field_name!r} must be text or null"
        )
    return value.strip() or None


def _record_integer(
    record: Mapping[str, object],
    field_name: str,
    *,
    line_number: int,
    minimum: int,
) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise IdentityInputError(
            f"Affinity line {line_number} field {field_name!r} must be a {qualifier} integer"
        )
    return value


def read_track_queries(path: str | Path) -> tuple[TrackQuery, ...]:
    """Read the identity fields from a track-affinity JSON Lines file."""

    source = Path(path)
    if not source.is_file():
        raise IdentityInputError("Track-affinity input does not exist or is not a file")
    queries: list[TrackQuery] = []
    seen_track_ids: set[str] = set()
    try:
        stream = source.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise IdentityInputError("Could not open track-affinity input") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IdentityInputError(
                    f"Track-affinity input line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(raw_record, Mapping):
                raise IdentityInputError(
                    f"Track-affinity input line {line_number} must contain an object"
                )
            record = cast(Mapping[str, object], raw_record)
            track_id = record.get("track_id")
            if not isinstance(track_id, str) or not track_id.strip():
                raise IdentityInputError(
                    f"Affinity line {line_number} field 'track_id' must be non-empty text"
                )
            track_id = track_id.strip()
            if track_id in seen_track_ids:
                raise IdentityInputError("Track-affinity input contains a duplicate track_id")
            seen_track_ids.add(track_id)

            raw_identity_source = record.get("track_identity_source")
            if raw_identity_source == "spotify_uri":
                identity_source: IdentitySource = "spotify_uri"
            elif raw_identity_source == "metadata_hash":
                identity_source = "metadata_hash"
            else:
                raise IdentityInputError(
                    f"Affinity line {line_number} has an unsupported track_identity_source"
                )
            track_uri = _optional_text(record, "track_uri", line_number=line_number)
            if track_uri is None and identity_source == "spotify_uri":
                track_uri = track_id
            if track_uri is not None and _TRACK_URI_PATTERN.fullmatch(track_uri) is None:
                raise IdentityInputError(
                    f"Affinity line {line_number} contains an invalid Spotify track URI"
                )
            queries.append(
                TrackQuery(
                    source_track_id=track_id,
                    source_identity_source=identity_source,
                    track_uri=track_uri,
                    track_name=_optional_text(record, "track_name", line_number=line_number),
                    artist_name=_optional_text(record, "artist_name", line_number=line_number),
                    album_name=_optional_text(record, "album_name", line_number=line_number),
                    play_count=_record_integer(
                        record, "play_count", line_number=line_number, minimum=1
                    ),
                    total_ms_played=_record_integer(
                        record, "total_ms_played", line_number=line_number, minimum=0
                    ),
                )
            )
    if not queries:
        raise IdentityInputError("Track-affinity input contains no records")
    return tuple(queries)


def normalize_metadata(value: str | None) -> str:
    """Normalize catalog text conservatively for deterministic comparison."""

    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(words.split())


@dataclass(slots=True)
class _CatalogIndex:
    by_uri: dict[str, list[CatalogTrack]]
    by_title_artist: dict[tuple[str, str], list[CatalogTrack]]
    by_title_artist_album: dict[tuple[str, str, str], list[CatalogTrack]]
    by_artist: dict[str, list[CatalogTrack]]


def _catalog_index(tracks: Iterable[CatalogTrack]) -> _CatalogIndex:
    by_uri: dict[str, list[CatalogTrack]] = defaultdict(list)
    by_title_artist: dict[tuple[str, str], list[CatalogTrack]] = defaultdict(list)
    by_title_artist_album: dict[tuple[str, str, str], list[CatalogTrack]] = defaultdict(list)
    by_artist: dict[str, list[CatalogTrack]] = defaultdict(list)
    for track in tracks:
        title = normalize_metadata(track.track_name)
        artist = normalize_metadata(track.artist_name)
        album = normalize_metadata(track.album_name)
        by_uri[track.track_uri].append(track)
        by_title_artist[(title, artist)].append(track)
        if album:
            by_title_artist_album[(title, artist, album)].append(track)
        by_artist[artist].append(track)
    return _CatalogIndex(
        by_uri=dict(by_uri),
        by_title_artist=dict(by_title_artist),
        by_title_artist_album=dict(by_title_artist_album),
        by_artist=dict(by_artist),
    )


def _catalog_source(track: CatalogTrack) -> str:
    return f"{track.source_file}:{track.source_collection}"


def _candidates(
    scored_tracks: Iterable[tuple[CatalogTrack, float]], *, limit: int
) -> tuple[MatchCandidate, ...]:
    best_by_uri: dict[str, tuple[CatalogTrack, float]] = {}
    for track, score in scored_tracks:
        current = best_by_uri.get(track.track_uri)
        if current is None or score > current[1] or (
            score == current[1] and _catalog_source(track) < _catalog_source(current[0])
        ):
            best_by_uri[track.track_uri] = (track, score)
    ordered = sorted(
        best_by_uri.values(),
        key=lambda item: (-item[1], item[0].track_uri, _catalog_source(item[0])),
    )
    return tuple(
        MatchCandidate(
            track_uri=track.track_uri,
            track_name=track.track_name,
            artist_name=track.artist_name,
            album_name=track.album_name,
            similarity_score=round(score, 6),
            catalog_source=_catalog_source(track),
        )
        for track, score in ordered[:limit]
    )


def _match(
    query: TrackQuery,
    *,
    status: MatchStatus,
    method: MatchMethod,
    policy: IdentityPolicy,
    candidates: tuple[MatchCandidate, ...] = (),
    resolved_track_id: str | None = None,
    confidence: float | None = None,
) -> IdentityMatch:
    return IdentityMatch(
        source_track_id=query.source_track_id,
        source_identity_source=query.source_identity_source,
        track_name=query.track_name,
        artist_name=query.artist_name,
        album_name=query.album_name,
        match_status=status,
        resolved_track_id=resolved_track_id,
        match_method=method,
        match_confidence=confidence,
        review_required=status != "exact",
        candidates=candidates,
        policy_version=policy.policy_version,
    )


def _fuzzy_candidates(
    query: TrackQuery,
    artist_tracks: Sequence[CatalogTrack],
    policy: IdentityPolicy,
) -> tuple[MatchCandidate, ...]:
    query_title = normalize_metadata(query.track_name)
    query_album = normalize_metadata(query.album_name)
    scored: list[tuple[CatalogTrack, float]] = []
    for track in artist_tracks:
        candidate_title = normalize_metadata(track.track_name)
        title_score = SequenceMatcher(None, query_title, candidate_title, autojunk=False).ratio()
        if title_score < policy.fuzzy_title_threshold:
            continue
        candidate_album = normalize_metadata(track.album_name)
        score = title_score
        if query_album and candidate_album:
            album_score = SequenceMatcher(
                None, query_album, candidate_album, autojunk=False
            ).ratio()
            score = (1 - policy.fuzzy_album_weight) * title_score + (
                policy.fuzzy_album_weight * album_score
            )
        if score >= policy.fuzzy_title_threshold:
            scored.append((track, score))
    return _candidates(scored, limit=policy.max_candidates)


def _resolve_query(
    query: TrackQuery, index: _CatalogIndex, policy: IdentityPolicy
) -> IdentityMatch:
    if query.track_uri is not None:
        uri_candidates = _candidates(
            ((track, 1.0) for track in index.by_uri.get(query.track_uri, [])),
            limit=1,
        )
        return _match(
            query,
            status="exact",
            method="existing_spotify_uri",
            policy=policy,
            candidates=uri_candidates,
            resolved_track_id=query.track_uri,
            confidence=1.0,
        )

    title = normalize_metadata(query.track_name)
    artist = normalize_metadata(query.artist_name)
    album = normalize_metadata(query.album_name)
    if not title or not artist:
        return _match(query, status="unmatched", method="none", policy=policy)

    if album:
        exact_album = _candidates(
            (
                (track, 1.0)
                for track in index.by_title_artist_album.get((title, artist, album), [])
            ),
            limit=policy.max_candidates,
        )
        if len(exact_album) == 1:
            return _match(
                query,
                status="exact",
                method="exact_title_artist_album",
                policy=policy,
                candidates=exact_album,
                resolved_track_id=exact_album[0].track_uri,
                confidence=1.0,
            )
        if len(exact_album) > 1:
            return _match(
                query,
                status="ambiguous",
                method="exact_title_artist_album",
                policy=policy,
                candidates=exact_album,
            )

    exact_title_artist = _candidates(
        ((track, 1.0) for track in index.by_title_artist.get((title, artist), [])),
        limit=policy.max_candidates,
    )
    if not album and len(exact_title_artist) == 1:
        return _match(
            query,
            status="exact",
            method="exact_title_artist",
            policy=policy,
            candidates=exact_title_artist,
            resolved_track_id=exact_title_artist[0].track_uri,
            confidence=policy.title_artist_exact_confidence,
        )
    if len(exact_title_artist) > 1:
        return _match(
            query,
            status="ambiguous",
            method="exact_title_artist",
            policy=policy,
            candidates=exact_title_artist,
        )

    fuzzy = _fuzzy_candidates(query, index.by_artist.get(artist, []), policy)
    if not fuzzy:
        return _match(query, status="unmatched", method="none", policy=policy)
    if len(fuzzy) > 1 and fuzzy[0].similarity_score - fuzzy[1].similarity_score < (
        policy.fuzzy_min_margin
    ):
        status: MatchStatus = "ambiguous"
    else:
        status = "fuzzy"
    return _match(
        query,
        status=status,
        method="fuzzy_title_exact_artist",
        policy=policy,
        candidates=fuzzy,
    )


def resolve_identities(
    queries: Iterable[TrackQuery],
    catalog: CatalogLoadResult,
    *,
    policy: IdentityPolicy | None = None,
) -> IdentityResolutionResult:
    """Resolve stable IDs without ever auto-accepting fuzzy or ambiguous matches."""

    active_policy = policy or IdentityPolicy()
    query_records = tuple(queries)
    if not query_records:
        raise IdentityResolutionError("Identity resolution requires at least one query")
    index = _catalog_index(catalog.tracks)
    matches = tuple(
        sorted(
            (_resolve_query(query, index, active_policy) for query in query_records),
            key=lambda match: match.source_track_id,
        )
    )
    status_counter = Counter(match.match_status for match in matches)
    method_counter = Counter(match.match_method for match in matches)
    status_counts: dict[str, int] = {
        status: status_counter[status] for status in _MATCH_STATUSES
    }
    status_rates: dict[str, float] = {
        status: round(status_counts[status] / len(matches), 6) for status in _MATCH_STATUSES
    }
    query_by_id = {query.source_track_id: query for query in query_records}
    history_play_count = sum(query.play_count for query in query_records)
    history_ms_played = sum(query.total_ms_played for query in query_records)
    resolved_matches = tuple(match for match in matches if match.resolved_track_id is not None)
    resolved_play_count = sum(
        query_by_id[match.source_track_id].play_count for match in resolved_matches
    )
    resolved_ms_played = sum(
        query_by_id[match.source_track_id].total_ms_played for match in resolved_matches
    )
    report = IdentityResolutionReport(
        catalog_source_files=catalog.source_files,
        catalog_records_seen=catalog.records_seen,
        catalog_duplicates_removed=catalog.duplicates_removed,
        catalog_unique_tracks=catalog.unique_track_count,
        queries_seen=len(matches),
        resolved_count=len(resolved_matches),
        history_play_count=history_play_count,
        resolved_play_count=resolved_play_count,
        resolved_play_rate=round(resolved_play_count / history_play_count, 6),
        history_ms_played=history_ms_played,
        resolved_ms_played=resolved_ms_played,
        resolved_ms_played_rate=(
            round(resolved_ms_played / history_ms_played, 6) if history_ms_played else 0.0
        ),
        review_required_count=sum(match.review_required for match in matches),
        status_counts=status_counts,
        status_rates=status_rates,
        method_counts={method: count for method, count in method_counter.items()},
        policy_version=active_policy.policy_version,
    )
    return IdentityResolutionResult(matches=matches, report=report)


def review_sample(
    matches: Iterable[IdentityMatch], *, sample_size_per_status: int
) -> tuple[IdentityMatch, ...]:
    """Select a deterministic, status-stratified manual-review sample."""

    if sample_size_per_status < 0:
        raise IdentityResolutionError("sample_size_per_status must be non-negative")
    grouped: dict[MatchStatus, list[IdentityMatch]] = defaultdict(list)
    for match in matches:
        grouped[match.match_status].append(match)
    sampled: list[IdentityMatch] = []
    for status in _MATCH_STATUSES:
        ordered = sorted(
            grouped[status],
            key=lambda match: (
                hashlib.sha256(match.source_track_id.encode("utf-8")).hexdigest(),
                match.source_track_id,
            ),
        )
        sampled.extend(ordered[:sample_size_per_status])
    return tuple(sampled)


def _json_lines(records: Iterable[Mapping[str, object]]) -> str:
    lines: list[str] = []
    for record in records:
        assert_privacy_safe(record)
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def write_identity_resolution(
    result: IdentityResolutionResult,
    output_directory: str | Path,
    *,
    review_sample_per_status: int,
) -> tuple[Path, Path, Path]:
    """Atomically write private match rows, an aggregate report, and a review sample."""

    output = Path(output_directory)
    matches_path = output / "identity_matches.jsonl"
    report_path = output / "identity_resolution_report.json"
    review_path = output / "identity_review_sample.jsonl"
    atomic_write_text(matches_path, _json_lines(match.to_dict() for match in result.matches))
    report_record = result.report.to_dict()
    assert_privacy_safe(report_record)
    atomic_write_text(
        report_path,
        json.dumps(report_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    sample = review_sample(result.matches, sample_size_per_status=review_sample_per_status)
    atomic_write_text(review_path, _json_lines(match.to_dict() for match in sample))
    return matches_path, report_path, review_path
