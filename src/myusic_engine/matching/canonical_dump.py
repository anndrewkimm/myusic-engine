"""Privacy-preserving MusicBrainz matching from the official canonical CC0 dump."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from myusic_engine.bulk_dump import open_dump_csv_lines
from myusic_engine.matching.models import IdentityInputError, TrackQuery
from myusic_engine.matching.resolver import normalize_metadata
from myusic_engine.providers import ListenBrainzMapping, ProviderError

_CANONICAL_MEMBER_NAME = "canonical_musicbrainz_data.csv"
_EXPECTED_COLUMNS = (
    "id",
    "artist_credit_id",
    "artist_mbids",
    "artist_credit_name",
    "release_mbid",
    "release_name",
    "recording_mbid",
    "recording_name",
    "combined_lookup",
    "score",
)
_MBID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_DUMP_TIMESTAMP = re.compile(r"(?P<date>[0-9]{8})-(?P<time>[0-9]{6})")

QueryKey = tuple[str, str, str | None]


class CanonicalDumpError(ValueError):
    """Raised when a canonical dump cannot be safely streamed or matched."""


@dataclass(frozen=True, slots=True)
class CanonicalDumpScanReport:
    """Aggregate evidence for one local canonical-data scan."""

    provider_name: str
    rows_scanned: int
    query_tracks_selected: int
    query_keys_with_metadata: int
    exact_query_keys: int
    ambiguous_query_keys: int
    candidate_rows_retained: int


@dataclass(frozen=True, slots=True)
class CanonicalDumpMapper:
    """In-memory exact mapper populated without disclosing query metadata."""

    mappings: Mapping[QueryKey, ListenBrainzMapping]

    def lookup(
        self,
        *,
        artist_name: str,
        recording_name: str,
        release_name: str | None,
    ) -> ListenBrainzMapping | None:
        return self.mappings.get(_query_key(artist_name, recording_name, release_name))


def _query_key(artist_name: str, recording_name: str, release_name: str | None) -> QueryKey:
    return (
        normalize_metadata(artist_name),
        normalize_metadata(recording_name),
        normalize_metadata(release_name) if release_name is not None else None,
    )


def _combined_lookup(artist_name: str, recording_name: str) -> str:
    # This mirrors MetaBrainz's published example for ``combined_lookup`` for Latin text. The
    # exact-artist fallback in the scanner also covers scripts that NFKD cannot transliterate.
    compact = re.sub(r"[^\w]+", "", (artist_name + recording_name).lower())
    return (
        unicodedata.normalize("NFKD", compact)
        .encode("ascii", errors="ignore")
        .decode("ascii")
        .casefold()
    )


def _selected_queries(
    queries: Iterable[TrackQuery], maximum_tracks: int | None
) -> tuple[TrackQuery, ...]:
    ordered = tuple(
        sorted(
            queries,
            key=lambda item: (-item.play_count, -item.total_ms_played, item.source_track_id),
        )
    )
    if maximum_tracks is None:
        return ordered
    if isinstance(maximum_tracks, bool) or not isinstance(maximum_tracks, int):
        raise IdentityInputError("maximum_tracks must be an integer")
    if maximum_tracks < 1:
        raise IdentityInputError("maximum_tracks must be positive")
    return ordered[:maximum_tracks]


def _provider_name(source: Path) -> str:
    timestamp = _DUMP_TIMESTAMP.search(source.name)
    if timestamp is None:
        return "musicbrainz_canonical_dump_local"
    return f"musicbrainz_canonical_dump_{timestamp.group('date')}_{timestamp.group('time')}"


def build_canonical_dump_mapper(
    queries: Iterable[TrackQuery],
    source: str | Path,
    *,
    maximum_tracks: int | None = None,
    progress: Callable[[int], None] | None = None,
    progress_interval: int = 1_000_000,
) -> tuple[CanonicalDumpMapper, CanonicalDumpScanReport]:
    """Scan a public canonical dump and retain only strict matches for selected private queries."""

    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, int)
        or progress_interval < 1
    ):
        raise CanonicalDumpError("progress_interval must be a positive integer")
    selected = _selected_queries(queries, maximum_tracks)
    if not selected:
        raise IdentityInputError("Canonical MusicBrainz matching received no queries")

    target_keys: set[QueryKey] = set()
    keys_by_combined: dict[str, set[QueryKey]] = defaultdict(set)
    keys_by_exact_artist: dict[str, set[QueryKey]] = defaultdict(set)
    for query in selected:
        if not query.artist_name or not query.track_name:
            continue
        key = _query_key(query.artist_name, query.track_name, query.album_name)
        target_keys.add(key)
        keys_by_combined[_combined_lookup(query.artist_name, query.track_name)].add(key)
        keys_by_exact_artist[query.artist_name.casefold()].add(key)

    retained: dict[QueryKey, dict[str, tuple[int, ListenBrainzMapping]]] = defaultdict(dict)
    rows_scanned = 0
    candidate_rows_retained = 0
    source_path = Path(source)
    try:
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(max(previous_limit, 10_000_000))
        with open_dump_csv_lines(
            source_path, member_name=_CANONICAL_MEMBER_NAME, error_cls=CanonicalDumpError
        ) as stream:
            reader = csv.reader(stream)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise CanonicalDumpError("Canonical MusicBrainz CSV is empty") from exc
            if header != _EXPECTED_COLUMNS:
                raise CanonicalDumpError("Canonical MusicBrainz CSV has an unsupported header")
            for rows_scanned, row in enumerate(reader, start=1):
                if len(row) != len(_EXPECTED_COLUMNS):
                    raise CanonicalDumpError(
                        f"Canonical MusicBrainz row {rows_scanned + 1} has "
                        f"{len(row)} fields instead of {len(_EXPECTED_COLUMNS)}: "
                        f"{tuple(value[:120] for value in row)!r}"
                    )
                possible_keys = set(keys_by_combined.get(row[8].casefold(), ()))
                possible_keys.update(keys_by_exact_artist.get(row[3].casefold(), ()))
                if possible_keys:
                    artist_key = normalize_metadata(row[3])
                    recording_key = normalize_metadata(row[7])
                    release_key = normalize_metadata(row[5]) if row[5] else ""
                    for key in possible_keys:
                        if key[0] != artist_key or key[1] != recording_key:
                            continue
                        if key[2] is not None and key[2] != release_key:
                            continue
                        try:
                            score = int(row[9])
                            mapping = ListenBrainzMapping(
                                recording_mbid=row[6].strip().lower(),
                                recording_name=row[7].strip(),
                                artist_credit_name=row[3].strip(),
                                release_mbid=row[4].strip().lower() or None,
                                release_name=row[5].strip() or None,
                                artist_mbids=tuple(
                                    value.lower() for value in _MBID_PATTERN.findall(row[2])
                                ),
                                confidence=1.0,
                            )
                        except (ProviderError, ValueError) as exc:
                            raise CanonicalDumpError(
                                f"Canonical MusicBrainz row {rows_scanned + 1} is invalid"
                            ) from exc
                        # Per MetaBrainz's canonical-data documentation, a higher score marks the
                        # more preferred/canonical row among duplicates for the same recording.
                        existing = retained[key].get(mapping.recording_mbid)
                        if existing is None or score > existing[0]:
                            retained[key][mapping.recording_mbid] = (score, mapping)
                        candidate_rows_retained += 1
                if progress is not None and rows_scanned % progress_interval == 0:
                    progress(rows_scanned)
    except csv.Error as exc:
        raise CanonicalDumpError("Canonical MusicBrainz CSV could not be parsed") from exc
    finally:
        csv.field_size_limit(previous_limit)
    if progress is not None and rows_scanned % progress_interval:
        progress(rows_scanned)

    mappings: dict[QueryKey, ListenBrainzMapping] = {}
    ambiguous = 0
    for key, by_recording in retained.items():
        if len(by_recording) != 1:
            ambiguous += 1
            continue
        _, mapping = max(by_recording.values(), key=lambda item: item[0])
        mappings[key] = mapping
    provider_name = _provider_name(source_path)
    report = CanonicalDumpScanReport(
        provider_name=provider_name,
        rows_scanned=rows_scanned,
        query_tracks_selected=len(selected),
        query_keys_with_metadata=len(target_keys),
        exact_query_keys=len(mappings),
        ambiguous_query_keys=ambiguous,
        candidate_rows_retained=candidate_rows_retained,
    )
    return CanonicalDumpMapper(mappings=mappings), report
