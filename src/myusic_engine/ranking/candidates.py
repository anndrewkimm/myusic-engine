"""Strict candidate intake from CSV, JSON Lines, Spotify URI, or Spotify URL text."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

_SPOTIFY_TRACK_ID = re.compile(r"^[A-Za-z0-9]{22}$")
_SPOTIFY_TRACK_URI = re.compile(r"^spotify:track:([A-Za-z0-9]{22})$")
_ALLOWED_FIELDS = frozenset(
    {
        "track_id",
        "spotify_uri",
        "uri",
        "url",
        "track_name",
        "title",
        "artist_name",
        "artist",
        "album_name",
        "album",
    }
)


class CandidateInputError(ValueError):
    """Raised when candidate input is ambiguous or cannot yield a stable track ID."""


@dataclass(frozen=True, slots=True)
class CandidateTrack:
    """One candidate identity with optional private display metadata."""

    track_id: str
    spotify_uri: str | None = None
    track_name: str | None = None
    artist_name: str | None = None
    album_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "spotify_uri": self.spotify_uri,
            "track_name": self.track_name,
            "artist_name": self.artist_name,
            "album_name": self.album_name,
        }


def _optional_text(record: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise CandidateInputError(f"Candidate field {key} must be text")
        if value.strip():
            return value.strip()
    return None


def spotify_uri(value: str) -> str | None:
    """Normalize a supported Spotify track URI/URL; return null for other IDs."""

    candidate = value.strip()
    match = _SPOTIFY_TRACK_URI.fullmatch(candidate)
    if match is not None:
        return f"spotify:track:{match.group(1)}"
    if _SPOTIFY_TRACK_ID.fullmatch(candidate) is not None:
        return f"spotify:track:{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme == "https" and parsed.netloc.casefold() in {
        "open.spotify.com",
        "www.open.spotify.com",
    }:
        components = [component for component in parsed.path.split("/") if component]
        if len(components) == 2 and components[0].casefold() == "track":
            track_id = components[1]
            if _SPOTIFY_TRACK_ID.fullmatch(track_id) is not None:
                return f"spotify:track:{track_id}"
    return None


def _candidate(record: Mapping[str, object]) -> CandidateTrack:
    unknown = set(record) - _ALLOWED_FIELDS
    if unknown:
        raise CandidateInputError(
            f"Unknown candidate fields: {', '.join(sorted(str(item) for item in unknown))}"
        )
    raw_track_id = _optional_text(record, "track_id")
    raw_spotify_identity = _optional_text(record, "spotify_uri", "uri", "url")
    if raw_track_id is None and raw_spotify_identity is None:
        raise CandidateInputError("Candidate needs track_id, URI, or URL")
    normalized_track_uri = spotify_uri(raw_track_id) if raw_track_id is not None else None
    normalized_explicit_uri = (
        spotify_uri(raw_spotify_identity) if raw_spotify_identity is not None else None
    )
    if raw_spotify_identity is not None and normalized_explicit_uri is None:
        raise CandidateInputError("Candidate Spotify URI/URL is not a supported track identity")
    if (
        normalized_track_uri is not None
        and normalized_explicit_uri is not None
        and normalized_track_uri != normalized_explicit_uri
    ):
        raise CandidateInputError("Candidate track_id and Spotify URI identify different tracks")
    track_id = raw_track_id or normalized_explicit_uri
    assert track_id is not None
    track_id = normalized_track_uri or track_id
    normalized_uri = normalized_explicit_uri or normalized_track_uri
    if not track_id.strip():
        raise CandidateInputError("Candidate track_id must not be empty")
    return CandidateTrack(
        track_id=track_id,
        spotify_uri=normalized_uri,
        track_name=_optional_text(record, "track_name", "title"),
        artist_name=_optional_text(record, "artist_name", "artist"),
        album_name=_optional_text(record, "album_name", "album"),
    )


def read_candidates(path: str | Path) -> tuple[CandidateTrack, ...]:
    """Read candidate records based on file extension and reject duplicate IDs."""

    source = Path(path)
    records: list[Mapping[str, object]] = []
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise CandidateInputError("Candidate CSV needs a header")
            for row in reader:
                records.append(cast(Mapping[str, object], row))
    elif suffix in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CandidateInputError(
                        f"Candidate line {line_number} is not valid JSON"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise CandidateInputError(f"Candidate line {line_number} must be an object")
                records.append(cast(Mapping[str, object], payload))
    else:
        with source.open("r", encoding="utf-8-sig") as stream:
            records.extend({"track_id": line.strip()} for line in stream if line.strip())
    candidates = tuple(_candidate(record) for record in records)
    if not candidates:
        raise CandidateInputError("Candidate input contains no tracks")
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.track_id in seen:
            raise CandidateInputError(f"Duplicate candidate track_id: {candidate.track_id}")
        seen.add(candidate.track_id)
    return candidates
