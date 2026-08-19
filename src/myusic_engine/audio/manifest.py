"""Strict JSON Lines boundary for private permitted-audio mappings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from myusic_engine.audio.models import AudioAsset, AudioInputError

_MANIFEST_FIELDS = frozenset({"track_id", "audio_path", "rights_basis"})


def _required_text(record: Mapping[str, object], field_name: str, line_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise AudioInputError(
            f"Audio manifest line {line_number} field {field_name!r} must be non-empty text"
        )
    return value


def read_audio_manifest(path: str | Path) -> tuple[AudioAsset, ...]:
    """Load a private manifest and resolve relative audio paths beside that manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise AudioInputError(f"Audio manifest is not a file: {manifest_path}")
    assets: list[AudioAsset] = []
    seen_track_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AudioInputError(
                    f"Audio manifest line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise AudioInputError(f"Audio manifest line {line_number} must be an object")
            record = cast(Mapping[str, object], payload)
            unknown_fields = set(record) - _MANIFEST_FIELDS
            if unknown_fields:
                names = ", ".join(sorted(unknown_fields))
                raise AudioInputError(
                    f"Audio manifest line {line_number} has unknown fields: {names}"
                )
            track_id = _required_text(record, "track_id", line_number)
            if track_id in seen_track_ids:
                raise AudioInputError(f"Duplicate track_id in audio manifest: {track_id}")
            raw_path = Path(_required_text(record, "audio_path", line_number))
            audio_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise AudioInputError(
                    f"Audio manifest line {line_number} does not reference a file: {audio_path}"
                )
            assets.append(
                AudioAsset(
                    track_id=track_id,
                    path=audio_path,
                    rights_basis=_required_text(record, "rights_basis", line_number),
                )
            )
            seen_track_ids.add(track_id)
    if not assets:
        raise AudioInputError("Audio manifest contains no assets")
    return tuple(assets)
