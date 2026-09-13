"""Import published MAEST audio embeddings by corroborated Spotify recording identity."""

from __future__ import annotations

import bz2
import csv
import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from myusic_engine.features.records import FeatureObservation
from myusic_engine.ranking.candidates import CandidateTrack

DIMENSIONS = 768
FEATURE_NAME = "maest_audio_embedding_v1"
FEATURE_SOURCE = "music4all_onion"
SOURCE_VERSION = "zenodo-15394646-maest-block7-l2-v1"
METADATA_REVISION = "a391160e3e17f351d5ab2d05439a7d3d7f0440eb"
FILES = {
    "id_metadata.csv": (
        f"https://huggingface.co/datasets/Leon299/music4all/resolve/{METADATA_REVISION}/id_metadata.csv",
        10623829,
        "fc32e2ce1b6af0f1dd7b68ea36ed0b2949191fd39a375edf416940de8a63e970",
    ),
    "id_information.csv": (
        f"https://huggingface.co/datasets/Leon299/music4all/resolve/{METADATA_REVISION}/id_information.csv",
        7000455,
        "11b7638e54c6f1bbb69746a087dcc62622cc9f04a7d3de13e98ffd073685be59",
    ),
    "id_maest.tsv.bz2": (
        "https://zenodo.org/records/15394646/files/id_maest.tsv.bz2?download=1",
        340816861,
        "a2db3ba58a2e637fe862f60ac50129ada34319d99da879778816823d907a3788",
    ),
}


@dataclass(frozen=True)
class Music4AllImport:
    observations: tuple[FeatureObservation, ...]
    matches: tuple[dict[str, str], ...]
    counts: dict[str, int]


def verify_source_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_size:
        raise ValueError(f"Music4All source size mismatch: {path.name}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError(f"Music4All source checksum mismatch: {path.name}")


def _metadata(path: Path, fields: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not fields <= set(reader.fieldnames or ()):
            raise ValueError(f"Missing Music4All metadata columns: {path.name}")
        for row in reader:
            if any(not isinstance(row.get(key), str) for key in fields):
                raise ValueError("Malformed Music4All metadata row")
            identity = row["id"]
            if not identity or identity in result:
                raise ValueError("Music4All metadata IDs must be nonempty and unique")
            # Retain identity metadata only, excluding legacy Spotify feature values.
            result[identity] = {key: row[key] for key in fields}
    return result


def _normalized(text: str | None) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def import_maest_features(
    tracks: Iterable[CandidateTrack], source_dir: str | Path
) -> Music4AllImport:
    """Match exact Spotify IDs plus title/artist/album; reject uncertain mappings.

    The released embeddings come from 30-second research excerpts, not full tracks.
    Confidence describes usable excerpt coverage, not verified recording identity or
    preference accuracy. Upstream recording matches still need human review.
    """
    source = Path(source_dir)
    for name, (_, size, digest) in FILES.items():
        verify_source_file(source / name, size, digest)
    identities = _metadata(source / "id_metadata.csv", {"id", "spotify_id"})
    information = _metadata(source / "id_information.csv", {"id", "artist", "song", "album_name"})
    by_spotify: dict[str, list[str]] = defaultdict(list)
    for identity, row in identities.items():
        if re.fullmatch(r"[A-Za-z0-9]{22}", row["spotify_id"]):
            by_spotify["spotify:track:" + row["spotify_id"]].append(identity)
    ordered = sorted(tracks, key=lambda track: track.track_id)
    if not ordered or len({track.track_id for track in ordered}) != len(ordered):
        raise ValueError("Music4All import requires nonempty, unique target track IDs")
    counts: Counter[str] = Counter(tracks_requested=len(ordered))
    matched: dict[str, list[CandidateTrack]] = defaultdict(list)
    for track in ordered:
        uri = track.spotify_uri or track.track_id
        choices = by_spotify.get(uri, [])
        if not choices:
            counts["spotify_id_not_found"] += 1
            continue
        counts["spotify_id_found"] += 1
        if len(choices) != 1:
            counts["ambiguous_spotify_id"] += 1
            continue
        item = information.get(choices[0], {})
        pairs = (
            (track.track_name, item.get("song")),
            (track.artist_name, item.get("artist")),
            (track.album_name, item.get("album_name")),
        )
        if any(
            not _normalized(left) or _normalized(left) != _normalized(right)
            for left, right in pairs
        ):
            counts["metadata_unconfirmed"] += 1
            continue
        matched[choices[0]].append(track)
    observations = []
    matches: list[dict[str, str]] = []
    seen: set[str] = set()
    with bz2.open(source / "id_maest.tsv.bz2", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if next(reader, None) != ["id", *map(str, range(DIMENSIONS))]:
            raise ValueError("Unexpected MAEST embedding columns")
        for embedding_row in reader:
            if not embedding_row:
                continue
            counts["embedding_rows_seen"] += 1
            identity = embedding_row[0]
            if identity in seen:
                raise ValueError("Duplicate MAEST embedding identity")
            seen.add(identity)
            if identity not in matched:
                continue
            if len(embedding_row) != DIMENSIONS + 1:
                raise ValueError("MAEST embedding dimensions do not match the published model")
            vector = tuple(float(value) for value in embedding_row[1:])
            norm = math.sqrt(math.fsum(value * value for value in vector))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise ValueError("MAEST embedding must be finite and nonzero")
            for track in matched[identity]:
                observations.append(
                    FeatureObservation(
                        track.track_id,
                        FEATURE_NAME,
                        tuple(value / norm for value in vector),
                        FEATURE_SOURCE,
                        SOURCE_VERSION,
                        30.0,
                        0.95,
                    )
                )
                matches.append(
                    {
                        "track_id": track.track_id,
                        "spotify_uri": track.spotify_uri or track.track_id,
                        "music4all_id": identity,
                        "match_method": "exact_spotify_id_and_title_artist_album",
                    }
                )
    counts["embedding_tracks"] = len(observations)
    counts["matched_metadata_without_embedding"] = sum(
        len(tracks) for identity, tracks in matched.items() if identity not in seen
    )
    return Music4AllImport(
        tuple(sorted(observations, key=lambda row: row.track_id)),
        tuple(sorted(matches, key=lambda row: row["track_id"])),
        dict(counts),
    )
