"""Cached client for the public ListenBrainz MusicBrainz metadata mapper."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from myusic_engine.providers.http import JsonCacheTransport, ProviderError

_MBID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ListenBrainzMapping:
    """One canonical MusicBrainz mapping returned for a metadata query."""

    recording_mbid: str
    recording_name: str
    artist_credit_name: str
    release_mbid: str | None
    release_name: str | None
    artist_mbids: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        if _MBID_PATTERN.fullmatch(self.recording_mbid) is None:
            raise ProviderError("ListenBrainz returned an invalid recording MBID")
        for value, field_name in (
            (self.recording_name, "recording_name"),
            (self.artist_credit_name, "artist_credit_name"),
        ):
            if not value.strip():
                raise ProviderError(f"ListenBrainz returned an empty {field_name}")
        if self.release_mbid is not None and _MBID_PATTERN.fullmatch(self.release_mbid) is None:
            raise ProviderError("ListenBrainz returned an invalid release MBID")
        if any(_MBID_PATTERN.fullmatch(value) is None for value in self.artist_mbids):
            raise ProviderError("ListenBrainz returned an invalid artist MBID")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ProviderError("ListenBrainz mapping confidence must be in [0, 1]")


class MusicBrainzMapper(Protocol):
    """Provider boundary used by deterministic external identity resolution."""

    def lookup(
        self,
        *,
        artist_name: str,
        recording_name: str,
        release_name: str | None,
    ) -> ListenBrainzMapping | None: ...


class ListenBrainzMappingClient:
    """Read MetaBrainz Labs' public semi-exact metadata mapper through the local cache."""

    artist_recording_endpoint = "https://labs.api.listenbrainz.org/acr-lookup/json"
    artist_recording_release_endpoint = "https://labs.api.listenbrainz.org/acrr-lookup/json"

    def __init__(self, transport: JsonCacheTransport) -> None:
        self.transport = transport

    def lookup(
        self,
        *,
        artist_name: str,
        recording_name: str,
        release_name: str | None,
    ) -> ListenBrainzMapping | None:
        parameters: dict[str, str] = {
            "artist_credit_name": artist_name,
            "recording_name": recording_name,
        }
        if release_name:
            parameters["release_name"] = release_name
        endpoint = (
            self.artist_recording_release_endpoint
            if release_name
            else self.artist_recording_endpoint
        )
        namespace = "listenbrainz-acrr" if release_name else "listenbrainz-acr"
        payload = self.transport.get_json(namespace, endpoint, parameters)
        if payload is None:
            return None
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
            raise ProviderError("ListenBrainz mapping response must be an array")
        if not payload:
            return None
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise ProviderError("ListenBrainz mapping response must contain one result")
        record = cast(Mapping[str, object], payload[0])
        raw_mbid = record.get("recording_mbid")
        if raw_mbid is None:
            return None
        artist_mbids = self._text_sequence(
            record.get("artist_credit_mbids", record.get("artist_mbids", [])),
            "artist_credit_mbids",
        )
        return ListenBrainzMapping(
            recording_mbid=self._text(raw_mbid, "recording_mbid"),
            recording_name=self._text(record.get("recording_name"), "recording_name"),
            artist_credit_name=self._text(record.get("artist_credit_name"), "artist_credit_name"),
            release_mbid=self._optional_text(record.get("release_mbid"), "release_mbid"),
            release_name=self._optional_text(record.get("release_name"), "release_name"),
            artist_mbids=artist_mbids,
            # These endpoints perform normalized semi-exact lookups. The downstream policy still
            # compares the returned canonical metadata to every original field before acceptance.
            confidence=1.0,
        )

    @staticmethod
    def _text(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ProviderError(f"ListenBrainz response field {field_name} must be text")
        return value.strip()

    @classmethod
    def _optional_text(cls, value: object, field_name: str) -> str | None:
        if value is None:
            return None
        return cls._text(value, field_name)

    @classmethod
    def _text_sequence(cls, value: object, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ProviderError(f"ListenBrainz response field {field_name} must be an array")
        return tuple(cls._text(item, field_name) for item in value)
