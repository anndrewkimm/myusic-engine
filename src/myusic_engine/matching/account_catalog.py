"""Offline Spotify account-data catalog reader for identity resolution."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TextIO, cast
from zipfile import BadZipFile, ZipFile

from myusic_engine.matching.models import (
    CatalogLoadResult,
    CatalogTrack,
    IdentityInputError,
)

_TRACK_URI_PATTERN = re.compile(r"^spotify:track:[A-Za-z0-9]{22}$")
_PLAYLIST_FILE_PATTERN = re.compile(r"^playlist[0-9]+\.json$", re.IGNORECASE)


def _supported_file(name: str) -> bool:
    basename = PurePosixPath(name.replace("\\", "/")).name
    return basename.casefold() == "yourlibrary.json" or bool(
        _PLAYLIST_FILE_PATTERN.fullmatch(basename)
    )


def _source_priority(name: str) -> tuple[int, str]:
    basename = PurePosixPath(name.replace("\\", "/")).name
    return (0 if basename.casefold() == "yourlibrary.json" else 1, basename.casefold())


def _load_json(stream: TextIO, source_file: str) -> Mapping[str, object]:
    try:
        payload = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityInputError(f"Could not decode catalog JSON in {source_file!r}") from exc
    if not isinstance(payload, Mapping):
        raise IdentityInputError(f"Catalog file {source_file!r} must contain an object")
    return cast(Mapping[str, object], payload)


def _source_documents(source: Path) -> Iterable[tuple[str, Mapping[str, object]]]:
    if not source.exists():
        raise IdentityInputError("Account-data catalog source does not exist")

    if source.is_dir():
        candidates = sorted(
            (path for path in source.rglob("*.json") if _supported_file(path.name)),
            key=lambda path: _source_priority(path.name),
        )
        if not candidates:
            raise IdentityInputError("Directory contains no supported account catalog files")
        for candidate in candidates:
            with candidate.open("r", encoding="utf-8-sig") as stream:
                yield candidate.name, _load_json(stream, candidate.name)
        return

    if source.suffix.casefold() == ".zip":
        try:
            with ZipFile(source) as archive:
                members = sorted(
                    (
                        member
                        for member in archive.infolist()
                        if not member.is_dir() and _supported_file(member.filename)
                    ),
                    key=lambda member: _source_priority(member.filename),
                )
                if not members:
                    raise IdentityInputError("ZIP contains no supported account catalog files")
                for member in members:
                    source_file = PurePosixPath(member.filename).name
                    with (
                        archive.open(member) as binary_stream,
                        io.TextIOWrapper(binary_stream, encoding="utf-8-sig") as stream,
                    ):
                        yield source_file, _load_json(stream, source_file)
        except BadZipFile as exc:
            raise IdentityInputError("Account-data catalog source is not a valid ZIP") from exc
        return

    if source.suffix.casefold() != ".json" or not _supported_file(source.name):
        raise IdentityInputError(
            "Account-data catalog source must be a ZIP, directory, YourLibrary.json, "
            "or PlaylistN.json"
        )
    with source.open("r", encoding="utf-8-sig") as stream:
        yield source.name, _load_json(stream, source.name)


def _records(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise IdentityInputError(f"{label} must be a JSON array")
    return cast(Sequence[object], value)


def _required_text(record: Mapping[str, object], field_name: str, *, label: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise IdentityInputError(f"{label} field {field_name!r} must be non-empty text")
    return value.strip()


def _optional_text(record: Mapping[str, object], field_name: str, *, label: str) -> str | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise IdentityInputError(f"{label} field {field_name!r} must be text or null")
    return value.strip() or None


def _validated_uri(value: str, *, label: str) -> str:
    if _TRACK_URI_PATTERN.fullmatch(value) is None:
        raise IdentityInputError(f"{label} contains an invalid Spotify track URI")
    return value


def _library_tracks(payload: Mapping[str, object], source_file: str) -> Iterable[CatalogTrack]:
    sections = (("tracks", "saved_library"), ("bannedTracks", "banned_library"))
    for field_name, source_collection in sections:
        for index, raw_record in enumerate(_records(payload.get(field_name, []), label=field_name)):
            label = f"{source_file} {field_name} record {index}"
            if not isinstance(raw_record, Mapping):
                raise IdentityInputError(f"{label} must be an object")
            record = cast(Mapping[str, object], raw_record)
            uri = _validated_uri(_required_text(record, "uri", label=label), label=label)
            yield CatalogTrack(
                track_uri=uri,
                track_name=_required_text(record, "track", label=label),
                artist_name=_required_text(record, "artist", label=label),
                album_name=_optional_text(record, "album", label=label),
                source_file=source_file,
                source_collection=source_collection,
            )


def _playlist_tracks(payload: Mapping[str, object], source_file: str) -> Iterable[CatalogTrack]:
    for playlist_index, raw_playlist in enumerate(
        _records(payload.get("playlists"), label="playlists")
    ):
        playlist_label = f"{source_file} playlist {playlist_index}"
        if not isinstance(raw_playlist, Mapping):
            raise IdentityInputError(f"{playlist_label} must be an object")
        playlist = cast(Mapping[str, object], raw_playlist)
        for item_index, raw_item in enumerate(
            _records(playlist.get("items"), label=f"{playlist_label} items")
        ):
            item_label = f"{playlist_label} item {item_index}"
            if not isinstance(raw_item, Mapping):
                raise IdentityInputError(f"{item_label} must be an object")
            raw_track = raw_item.get("track")
            if raw_track is None:
                continue
            if not isinstance(raw_track, Mapping):
                raise IdentityInputError(f"{item_label} track must be an object")
            track = cast(Mapping[str, object], raw_track)
            uri = _validated_uri(
                _required_text(track, "trackUri", label=item_label), label=item_label
            )
            yield CatalogTrack(
                track_uri=uri,
                track_name=_required_text(track, "trackName", label=item_label),
                artist_name=_required_text(track, "artistName", label=item_label),
                album_name=_optional_text(track, "albumName", label=item_label),
                source_file=source_file,
                source_collection="playlist",
            )


def load_account_catalog(source: str | Path) -> CatalogLoadResult:
    """Load URI-bearing tracks from local Spotify library and playlist export files."""

    tracks: list[CatalogTrack] = []
    source_files: list[str] = []
    records_seen = 0
    duplicates_removed = 0
    seen_entries: set[tuple[str, str, str, str | None]] = set()

    for source_file, payload in _source_documents(Path(source)):
        source_files.append(source_file)
        if source_file.casefold() == "yourlibrary.json":
            parsed_tracks = _library_tracks(payload, source_file)
        else:
            parsed_tracks = _playlist_tracks(payload, source_file)
        for track in parsed_tracks:
            records_seen += 1
            identity = (
                track.track_uri,
                track.track_name,
                track.artist_name,
                track.album_name,
            )
            if identity in seen_entries:
                duplicates_removed += 1
                continue
            seen_entries.add(identity)
            tracks.append(track)

    if not tracks:
        raise IdentityInputError("Account-data catalog contains no Spotify tracks")
    return CatalogLoadResult(
        tracks=tuple(tracks),
        source_files=tuple(source_files),
        records_seen=records_seen,
        duplicates_removed=duplicates_removed,
    )
