"""Data structures for explicit, reviewable track identity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

IdentitySource: TypeAlias = Literal["spotify_uri", "metadata_hash"]
MatchStatus: TypeAlias = Literal["exact", "fuzzy", "ambiguous", "unmatched"]
MatchMethod: TypeAlias = Literal[
    "existing_spotify_uri",
    "exact_title_artist_album",
    "exact_title_artist",
    "fuzzy_title_exact_artist",
    "none",
]


class IdentityResolutionError(ValueError):
    """Base error for invalid identity inputs, policies, or catalog records."""


class IdentityInputError(IdentityResolutionError):
    """Raised when identity-resolution input cannot be read safely."""


@dataclass(frozen=True, slots=True)
class TrackQuery:
    """The identity evidence retained for one behavior aggregate."""

    source_track_id: str
    source_identity_source: IdentitySource
    track_uri: str | None
    track_name: str | None
    artist_name: str | None
    album_name: str | None
    play_count: int = 1
    total_ms_played: int = 0


@dataclass(frozen=True, slots=True)
class CatalogTrack:
    """One URI-bearing metadata observation from an allowed local catalog source."""

    track_uri: str
    track_name: str
    artist_name: str
    album_name: str | None
    source_file: str
    source_collection: str


@dataclass(frozen=True, slots=True)
class CatalogLoadResult:
    """Catalog entries plus aggregate loading diagnostics."""

    tracks: tuple[CatalogTrack, ...]
    source_files: tuple[str, ...]
    records_seen: int
    duplicates_removed: int

    @property
    def unique_track_count(self) -> int:
        """Return the number of distinct Spotify track URIs in the catalog."""

        return len({track.track_uri for track in self.tracks})


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    """A catalog candidate retained as inspectable evidence."""

    track_uri: str
    track_name: str
    artist_name: str
    album_name: str | None
    similarity_score: float
    catalog_source: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready candidate."""

        return {
            "track_uri": self.track_uri,
            "track_name": self.track_name,
            "artist_name": self.artist_name,
            "album_name": self.album_name,
            "similarity_score": self.similarity_score,
            "catalog_source": self.catalog_source,
        }


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    """Resolution outcome for one history track identity."""

    source_track_id: str
    source_identity_source: IdentitySource
    track_name: str | None
    artist_name: str | None
    album_name: str | None
    match_status: MatchStatus
    resolved_track_id: str | None
    match_method: MatchMethod
    match_confidence: float | None
    review_required: bool
    candidates: tuple[MatchCandidate, ...]
    policy_version: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready match record."""

        return {
            "schema_version": self.schema_version,
            "source_track_id": self.source_track_id,
            "source_identity_source": self.source_identity_source,
            "track_name": self.track_name,
            "artist_name": self.artist_name,
            "album_name": self.album_name,
            "match_status": self.match_status,
            "resolved_track_id": self.resolved_track_id,
            "match_method": self.match_method,
            "match_confidence": self.match_confidence,
            "review_required": self.review_required,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class IdentityResolutionReport:
    """Aggregate, non-sensitive diagnostics for one resolution run."""

    catalog_source_files: tuple[str, ...]
    catalog_records_seen: int
    catalog_duplicates_removed: int
    catalog_unique_tracks: int
    queries_seen: int
    resolved_count: int
    history_play_count: int
    resolved_play_count: int
    resolved_play_rate: float
    history_ms_played: int
    resolved_ms_played: int
    resolved_ms_played_rate: float
    review_required_count: int
    status_counts: dict[str, int]
    status_rates: dict[str, float]
    method_counts: dict[str, int]
    policy_version: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready report."""

        return {
            "schema_version": self.schema_version,
            "catalog_source_files": list(self.catalog_source_files),
            "catalog_records_seen": self.catalog_records_seen,
            "catalog_duplicates_removed": self.catalog_duplicates_removed,
            "catalog_unique_tracks": self.catalog_unique_tracks,
            "queries_seen": self.queries_seen,
            "resolved_count": self.resolved_count,
            "history_play_count": self.history_play_count,
            "resolved_play_count": self.resolved_play_count,
            "resolved_play_rate": self.resolved_play_rate,
            "history_ms_played": self.history_ms_played,
            "resolved_ms_played": self.resolved_ms_played,
            "resolved_ms_played_rate": self.resolved_ms_played_rate,
            "review_required_count": self.review_required_count,
            "status_counts": dict(sorted(self.status_counts.items())),
            "status_rates": dict(sorted(self.status_rates.items())),
            "method_counts": dict(sorted(self.method_counts.items())),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class IdentityResolutionResult:
    """All match rows and their aggregate quality report."""

    matches: tuple[IdentityMatch, ...]
    report: IdentityResolutionReport
