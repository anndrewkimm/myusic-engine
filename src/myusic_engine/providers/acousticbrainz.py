"""Bulk, cached access to frozen CC0 AcousticBrainz feature documents."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from myusic_engine.providers.http import JsonCacheTransport, ProviderError

_MBID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_LOW_LEVEL_FEATURES = (
    "lowlevel.average_loudness",
    "lowlevel.dynamic_complexity",
    "metadata.audio_properties.replay_gain",
    "rhythm.beats_count",
    "rhythm.beats_loudness.mean",
    "rhythm.bpm",
    "rhythm.danceability",
    "rhythm.onset_rate",
    "tonal.chords_changes_rate",
    "tonal.key_key",
    "tonal.key_scale",
    "tonal.key_strength",
    "tonal.tuning_equal_tempered_deviation",
    "tonal.tuning_frequency",
)


@dataclass(frozen=True, slots=True)
class AcousticBrainzDocument:
    """Selected offset-zero low/high-level documents for one recording MBID."""

    recording_mbid: str
    low_level: Mapping[str, object] | None
    high_level: Mapping[str, object] | None


class AcousticBrainzProvider(Protocol):
    """Bulk provider boundary used by the feature conversion stage."""

    def fetch(self, recording_mbids: Sequence[str]) -> Mapping[str, AcousticBrainzDocument]: ...


class AcousticBrainzClient:
    """Fetch up to 25 MBIDs per official AcousticBrainz bulk request."""

    low_level_endpoint = "https://acousticbrainz.org/api/v1/low-level"
    high_level_endpoint = "https://acousticbrainz.org/api/v1/high-level"
    maximum_batch_size = 25

    def __init__(self, transport: JsonCacheTransport) -> None:
        self.transport = transport

    def fetch(self, recording_mbids: Sequence[str]) -> Mapping[str, AcousticBrainzDocument]:
        normalized = tuple(dict.fromkeys(mbid.casefold() for mbid in recording_mbids))
        if not normalized:
            return {}
        if len(normalized) > self.maximum_batch_size:
            raise ProviderError("AcousticBrainz accepts at most 25 MBIDs per bulk request")
        if any(_MBID_PATTERN.fullmatch(mbid) is None for mbid in normalized):
            raise ProviderError("AcousticBrainz request contains an invalid recording MBID")
        recording_ids = ";".join(normalized)
        low_payload = self.transport.get_json(
            "acousticbrainz-low-level",
            self.low_level_endpoint,
            {
                "recording_ids": recording_ids,
                "features": ";".join(_LOW_LEVEL_FEATURES),
            },
        )
        high_payload = self.transport.get_json(
            "acousticbrainz-high-level",
            self.high_level_endpoint,
            {"recording_ids": recording_ids, "map_classes": "false"},
        )
        low_documents = self._documents(low_payload, "low-level")
        high_documents = self._documents(high_payload, "high-level")
        return {
            mbid: AcousticBrainzDocument(
                recording_mbid=mbid,
                low_level=low_documents.get(mbid),
                high_level=high_documents.get(mbid),
            )
            for mbid in normalized
            if mbid in low_documents or mbid in high_documents
        }

    @classmethod
    def _documents(cls, payload: object | None, label: str) -> dict[str, Mapping[str, object]]:
        if payload is None:
            return {}
        if not isinstance(payload, Mapping):
            raise ProviderError(f"AcousticBrainz {label} response must be an object")
        documents: dict[str, Mapping[str, object]] = {}
        for raw_mbid, raw_offsets in payload.items():
            mbid = str(raw_mbid).casefold()
            if mbid == "mbid_mapping":
                continue
            if _MBID_PATTERN.fullmatch(mbid) is None or not isinstance(raw_offsets, Mapping):
                raise ProviderError(f"AcousticBrainz {label} response has an invalid entry")
            offsets = cast(Mapping[str, object], raw_offsets)
            raw_document = offsets.get("0")
            if raw_document is None and offsets:
                numeric_keys = sorted((str(key) for key in offsets if str(key).isdigit()), key=int)
                raw_document = offsets.get(numeric_keys[0]) if numeric_keys else None
            if raw_document is None:
                continue
            if not isinstance(raw_document, Mapping):
                raise ProviderError(f"AcousticBrainz {label} document must be an object")
            documents[mbid] = cast(Mapping[str, object], raw_document)
        return documents
