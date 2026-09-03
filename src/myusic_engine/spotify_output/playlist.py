"""Explicit, resumable publication of ranked tracks to a private Spotify playlist."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from myusic_engine.io import atomic_write_text

_API_BASE_URL = "https://api.spotify.com/v1"
_MAX_RESPONSE_BYTES = 5_000_000
_MAX_ITEMS_PER_REQUEST = 100
_PLAYLIST_PAGE_SIZE = 50
_SPOTIFY_ID = re.compile(r"^[A-Za-z0-9]{1,128}$")
_SPOTIFY_TRACK_URI = re.compile(r"^spotify:track:[A-Za-z0-9]{22}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")

PublicationStatus: TypeAlias = Literal["created", "publishing", "complete"]


class SpotifyPlaylistError(ValueError):
    """Raised when a playlist plan, API response, or remote state is unsafe to use."""


@dataclass(frozen=True, slots=True)
class PlaylistPublicationPlan:
    """Deterministic private-playlist intent produced before any network mutation."""

    plan_id: str
    playlist_name: str
    description: str
    spotify_uris: tuple[str, ...]
    schema_version: int = 1

    @property
    def item_count(self) -> int:
        return len(self.spotify_uris)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "playlist_name": self.playlist_name,
            "description": self.description,
            "public": False,
            "item_count": self.item_count,
            "spotify_uris": list(self.spotify_uris),
        }


@dataclass(frozen=True, slots=True)
class CreatedPlaylist:
    """Minimal identity returned after Spotify creates a private playlist."""

    playlist_id: str
    playlist_uri: str
    playlist_url: str | None


@dataclass(frozen=True, slots=True)
class PlaylistPublicationReceipt:
    """Secret-free checkpoint used to reconcile and resume a publication."""

    plan_id: str
    playlist_id: str
    playlist_uri: str
    playlist_url: str | None
    requested_item_count: int
    confirmed_item_count: int
    status: PublicationStatus
    latest_snapshot_id: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "playlist_id": self.playlist_id,
            "playlist_uri": self.playlist_uri,
            "playlist_url": self.playlist_url,
            "requested_item_count": self.requested_item_count,
            "confirmed_item_count": self.confirmed_item_count,
            "status": self.status,
            "latest_snapshot_id": self.latest_snapshot_id,
        }


class SpotifyPlaylistGateway(Protocol):
    """Narrow mutable boundary used by the publication state machine."""

    def create_private_playlist(self, name: str, description: str) -> CreatedPlaylist:
        """Create and return one empty private playlist."""

    def playlist_track_uris(self, playlist_id: str) -> tuple[str, ...]:
        """Return every remote track URI in its current order."""

    def add_playlist_items(self, playlist_id: str, uris: Sequence[str]) -> str:
        """Append at most 100 track URIs and return the resulting snapshot ID."""


def _clean_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpotifyPlaylistError("Playlist name must be non-empty text")
    return value.strip()


def _clean_description(value: str) -> str:
    if not isinstance(value, str):
        raise SpotifyPlaylistError("Playlist description must be text")
    return value.strip()


def _validate_track_uris(uris: Sequence[str]) -> tuple[str, ...]:
    if isinstance(uris, (str, bytes)):
        raise SpotifyPlaylistError("Playlist items must be a sequence of Spotify track URIs")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_uri in uris:
        if not isinstance(raw_uri, str) or _SPOTIFY_TRACK_URI.fullmatch(raw_uri) is None:
            raise SpotifyPlaylistError(
                "Playlist items must use canonical spotify:track:<22-character-id> URIs"
            )
        if raw_uri in seen:
            raise SpotifyPlaylistError(f"Playlist plan contains a duplicate URI: {raw_uri}")
        seen.add(raw_uri)
        normalized.append(raw_uri)
    if not normalized:
        raise SpotifyPlaylistError("Playlist plan contains no Spotify track URIs")
    return tuple(normalized)


def read_spotify_uri_file(path: str | Path) -> tuple[str, ...]:
    """Read the ordered URI handoff emitted by candidate ranking."""

    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise SpotifyPlaylistError("Spotify URI handoff could not be read") from exc
    return _validate_track_uris(tuple(line.strip() for line in lines if line.strip()))


def _plan_identity(name: str, description: str, uris: Sequence[str]) -> str:
    canonical = json.dumps(
        {
            "schema_version": 1,
            "playlist_name": name,
            "description": description,
            "public": False,
            "spotify_uris": list(uris),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_publication_plan(
    spotify_uris: Sequence[str],
    *,
    playlist_name: str,
    description: str = "Generated by Myusic Engine.",
) -> PlaylistPublicationPlan:
    """Validate and hash the exact private-playlist mutation before execution."""

    name = _clean_name(playlist_name)
    cleaned_description = _clean_description(description)
    uris = _validate_track_uris(spotify_uris)
    return PlaylistPublicationPlan(
        plan_id=_plan_identity(name, cleaned_description, uris),
        playlist_name=name,
        description=cleaned_description,
        spotify_uris=uris,
    )


def _validate_plan(plan: PlaylistPublicationPlan) -> None:
    name = _clean_name(plan.playlist_name)
    description = _clean_description(plan.description)
    uris = _validate_track_uris(plan.spotify_uris)
    if (
        plan.schema_version != 1
        or name != plan.playlist_name
        or description != plan.description
        or uris != plan.spotify_uris
        or plan.plan_id != _plan_identity(name, description, uris)
    ):
        raise SpotifyPlaylistError("Playlist publication plan content does not match its digest")


def write_publication_plan(plan: PlaylistPublicationPlan, path: str | Path) -> Path:
    """Atomically write a human-reviewable mutation plan."""

    _validate_plan(plan)
    return atomic_write_text(
        path,
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _record(path: str | Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpotifyPlaylistError(f"{label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise SpotifyPlaylistError(f"{label} must contain one JSON object")
    return cast(Mapping[str, object], payload)


def read_publication_plan(path: str | Path) -> PlaylistPublicationPlan:
    """Read a plan and verify that its content still matches its digest."""

    record = _record(path, "Playlist publication plan")
    expected_keys = {
        "schema_version",
        "plan_id",
        "playlist_name",
        "description",
        "public",
        "item_count",
        "spotify_uris",
    }
    if set(record) != expected_keys or record.get("schema_version") != 1:
        raise SpotifyPlaylistError("Playlist publication plan does not match schema version 1")
    if record.get("public") is not False:
        raise SpotifyPlaylistError("Only private playlist publication plans are supported")
    raw_uris = record.get("spotify_uris")
    if not isinstance(raw_uris, list) or not all(isinstance(item, str) for item in raw_uris):
        raise SpotifyPlaylistError("Playlist publication plan has invalid Spotify URIs")
    plan = create_publication_plan(
        cast(list[str], raw_uris),
        playlist_name=cast(str, record.get("playlist_name")),
        description=cast(str, record.get("description")),
    )
    if record.get("item_count") != plan.item_count or record.get("plan_id") != plan.plan_id:
        raise SpotifyPlaylistError("Playlist publication plan digest or item count does not match")
    return plan


def _publication_status(confirmed: int, requested: int) -> PublicationStatus:
    if confirmed == requested:
        return "complete"
    return "created" if confirmed == 0 else "publishing"


def _validate_playlist_id(value: object) -> str:
    if not isinstance(value, str) or _SPOTIFY_ID.fullmatch(value) is None:
        raise SpotifyPlaylistError("Spotify returned an invalid playlist ID")
    return value


def _validate_playlist_url(value: object, playlist_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpotifyPlaylistError("Playlist publication has an invalid playlist URL")
    parsed_url = urlsplit(value)
    path_components = [component for component in parsed_url.path.split("/") if component]
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname not in {"open.spotify.com", "www.open.spotify.com"}
        or parsed_url.username is not None
        or parsed_url.password is not None
        or path_components != ["playlist", playlist_id]
    ):
        raise SpotifyPlaylistError("Playlist publication has an invalid playlist URL")
    return value


def write_publication_receipt(receipt: PlaylistPublicationReceipt, path: str | Path) -> Path:
    """Atomically checkpoint confirmed remote progress without storing OAuth material."""

    return atomic_write_text(
        path,
        json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_publication_receipt(path: str | Path) -> PlaylistPublicationReceipt:
    """Read and strictly validate a resumable publication checkpoint."""

    record = _record(path, "Playlist publication receipt")
    expected_keys = {
        "schema_version",
        "plan_id",
        "playlist_id",
        "playlist_uri",
        "playlist_url",
        "requested_item_count",
        "confirmed_item_count",
        "status",
        "latest_snapshot_id",
    }
    if set(record) != expected_keys or record.get("schema_version") != 1:
        raise SpotifyPlaylistError("Playlist publication receipt does not match schema version 1")
    plan_id = record.get("plan_id")
    if not isinstance(plan_id, str) or _SHA256.fullmatch(plan_id) is None:
        raise SpotifyPlaylistError("Playlist publication receipt has an invalid plan ID")
    playlist_id = _validate_playlist_id(record.get("playlist_id"))
    playlist_uri = record.get("playlist_uri")
    if playlist_uri != f"spotify:playlist:{playlist_id}":
        raise SpotifyPlaylistError("Playlist publication receipt has an inconsistent playlist URI")
    playlist_url = _validate_playlist_url(record.get("playlist_url"), playlist_id)
    requested = record.get("requested_item_count")
    confirmed = record.get("confirmed_item_count")
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < 1
        or isinstance(confirmed, bool)
        or not isinstance(confirmed, int)
        or not 0 <= confirmed <= requested
    ):
        raise SpotifyPlaylistError("Playlist publication receipt has invalid item counts")
    status = record.get("status")
    if status not in {"created", "publishing", "complete"}:
        raise SpotifyPlaylistError("Playlist publication receipt has an invalid status")
    if status != _publication_status(confirmed, requested):
        raise SpotifyPlaylistError("Playlist publication receipt status and counts disagree")
    snapshot_id = record.get("latest_snapshot_id")
    if snapshot_id is not None and (not isinstance(snapshot_id, str) or not snapshot_id):
        raise SpotifyPlaylistError("Playlist publication receipt has an invalid snapshot ID")
    return PlaylistPublicationReceipt(
        plan_id=plan_id,
        playlist_id=playlist_id,
        playlist_uri=playlist_uri,
        playlist_url=playlist_url,
        requested_item_count=requested,
        confirmed_item_count=confirmed,
        status=status,
        latest_snapshot_id=snapshot_id,
    )


def _reconciled_count(plan: PlaylistPublicationPlan, remote_uris: Sequence[str]) -> int:
    remote = tuple(remote_uris)
    if len(remote) > plan.item_count or remote != plan.spotify_uris[: len(remote)]:
        raise SpotifyPlaylistError(
            "Remote playlist items are not an exact prefix of this plan; refusing to append"
        )
    return len(remote)


def publish_playlist(
    plan: PlaylistPublicationPlan,
    gateway: SpotifyPlaylistGateway,
    receipt_path: str | Path,
) -> PlaylistPublicationReceipt:
    """Create or safely resume a private playlist from a deterministic plan."""

    _validate_plan(plan)
    destination = Path(receipt_path)
    if destination.exists():
        receipt = read_publication_receipt(destination)
        if receipt.plan_id != plan.plan_id or receipt.requested_item_count != plan.item_count:
            raise SpotifyPlaylistError(
                "Existing playlist receipt belongs to a different publication plan"
            )
    else:
        created = gateway.create_private_playlist(plan.playlist_name, plan.description)
        playlist_id = _validate_playlist_id(created.playlist_id)
        if created.playlist_uri != f"spotify:playlist:{playlist_id}":
            raise SpotifyPlaylistError("Created playlist identity is internally inconsistent")
        receipt = PlaylistPublicationReceipt(
            plan_id=plan.plan_id,
            playlist_id=playlist_id,
            playlist_uri=created.playlist_uri,
            playlist_url=created.playlist_url,
            requested_item_count=plan.item_count,
            confirmed_item_count=0,
            status="created",
        )
        write_publication_receipt(receipt, destination)

    confirmed = _reconciled_count(plan, gateway.playlist_track_uris(receipt.playlist_id))
    receipt = replace(
        receipt,
        confirmed_item_count=confirmed,
        status=_publication_status(confirmed, plan.item_count),
    )
    write_publication_receipt(receipt, destination)

    while confirmed < plan.item_count:
        batch = plan.spotify_uris[confirmed : confirmed + _MAX_ITEMS_PER_REQUEST]
        snapshot_id = gateway.add_playlist_items(receipt.playlist_id, batch)
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise SpotifyPlaylistError("Spotify returned an invalid playlist snapshot ID")
        confirmed += len(batch)
        receipt = replace(
            receipt,
            confirmed_item_count=confirmed,
            status=_publication_status(confirmed, plan.item_count),
            latest_snapshot_id=snapshot_id,
        )
        write_publication_receipt(receipt, destination)

    final_count = _reconciled_count(plan, gateway.playlist_track_uris(receipt.playlist_id))
    if final_count != plan.item_count:
        raise SpotifyPlaylistError("Spotify did not confirm every planned playlist item")
    receipt = replace(
        receipt,
        confirmed_item_count=final_count,
        status="complete",
    )
    write_publication_receipt(receipt, destination)
    return receipt


class SpotifyWebApiClient:
    """Small bearer-token client for current supported playlist operations."""

    def __init__(
        self,
        access_token: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if (
            not isinstance(access_token, str)
            or not access_token
            or any(character.isspace() for character in access_token)
        ):
            raise SpotifyPlaylistError("Spotify access token is missing or malformed")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise SpotifyPlaylistError("Spotify API timeout must be positive and finite")
        self._access_token = access_token
        self.timeout_seconds = float(timeout_seconds)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, object] | None = None,
        parameters: Mapping[str, str] | None = None,
        expected_status: int,
    ) -> Mapping[str, object]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise SpotifyPlaylistError("Spotify API request path is invalid")
        query = urlencode(sorted((parameters or {}).items()))
        url = f"{_API_BASE_URL}{path}"
        if query:
            url = f"{url}?{query}"
        encoded_body = (
            json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            if body is not None
            else None
        )
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._access_token}",
            "User-Agent": "myusic-engine/0.1.0 (private-playlist-output)",
        }
        if encoded_body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=encoded_body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            if exc.code == 401:
                detail = "authorization failed; refresh the access token"
            elif exc.code == 403:
                detail = "operation forbidden; check playlist ownership, scopes, and app access"
            elif exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                detail = "rate limited"
                if retry_after is not None and retry_after.isdecimal():
                    detail = f"rate limited; retry after {retry_after} seconds"
            elif exc.code >= 500:
                detail = "service failed before the client received confirmation"
            else:
                detail = "request was rejected"
            raise SpotifyPlaylistError(f"Spotify Web API {detail} (HTTP {exc.code})") from exc
        except (TimeoutError, URLError) as exc:
            raise SpotifyPlaylistError(
                "Spotify Web API request ended without a confirmed response"
            ) from exc
        if status != expected_status:
            raise SpotifyPlaylistError(f"Spotify Web API returned unexpected HTTP status {status}")
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise SpotifyPlaylistError("Spotify Web API response exceeded 5 MB")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpotifyPlaylistError("Spotify Web API returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise SpotifyPlaylistError("Spotify Web API returned an unexpected JSON value")
        return cast(Mapping[str, object], payload)

    def create_private_playlist(self, name: str, description: str) -> CreatedPlaylist:
        """Create an empty private playlist through ``POST /me/playlists``."""

        payload = self._request_json(
            "POST",
            "/me/playlists",
            body={
                "name": _clean_name(name),
                "description": _clean_description(description),
                "public": False,
            },
            expected_status=201,
        )
        playlist_id = _validate_playlist_id(payload.get("id"))
        if payload.get("public") is not False:
            raise SpotifyPlaylistError(
                "Spotify did not confirm that the newly created playlist is private"
            )
        playlist_url: str | None = None
        external_urls = payload.get("external_urls")
        if isinstance(external_urls, Mapping):
            playlist_url = _validate_playlist_url(external_urls.get("spotify"), playlist_id)
        return CreatedPlaylist(
            playlist_id=playlist_id,
            playlist_uri=f"spotify:playlist:{playlist_id}",
            playlist_url=playlist_url,
        )

    def playlist_track_uris(self, playlist_id: str) -> tuple[str, ...]:
        """Read all owned/collaborative playlist items through the paginated items route."""

        safe_playlist_id = _validate_playlist_id(playlist_id)
        uris: list[str] = []
        offset = 0
        while True:
            payload = self._request_json(
                "GET",
                f"/playlists/{safe_playlist_id}/items",
                parameters={"limit": str(_PLAYLIST_PAGE_SIZE), "offset": str(offset)},
                expected_status=200,
            )
            raw_items = payload.get("items")
            total = payload.get("total")
            if (
                not isinstance(raw_items, list)
                or isinstance(total, bool)
                or not isinstance(total, int)
                or total < 0
            ):
                raise SpotifyPlaylistError("Spotify returned an invalid playlist-items page")
            for raw_item in raw_items:
                if not isinstance(raw_item, Mapping):
                    raise SpotifyPlaylistError("Spotify returned an invalid playlist item")
                item = raw_item.get("item", raw_item.get("track"))
                if not isinstance(item, Mapping) or item.get("type") != "track":
                    raise SpotifyPlaylistError(
                        "Remote playlist contains an unavailable, local, or non-track item"
                    )
                uri = item.get("uri")
                if not isinstance(uri, str) or _SPOTIFY_TRACK_URI.fullmatch(uri) is None:
                    raise SpotifyPlaylistError("Spotify returned an invalid playlist track URI")
                uris.append(uri)
            if len(uris) > total:
                raise SpotifyPlaylistError("Spotify playlist pagination changed during validation")
            if len(uris) == total:
                return tuple(uris)
            if not raw_items or payload.get("next") is None:
                raise SpotifyPlaylistError("Spotify playlist pagination ended before its total")
            offset += len(raw_items)

    def add_playlist_items(self, playlist_id: str, uris: Sequence[str]) -> str:
        """Append one official-maximum batch through ``POST /playlists/{id}/items``."""

        safe_playlist_id = _validate_playlist_id(playlist_id)
        batch = _validate_track_uris(uris)
        if len(batch) > _MAX_ITEMS_PER_REQUEST:
            raise SpotifyPlaylistError("Spotify accepts at most 100 playlist items per request")
        payload = self._request_json(
            "POST",
            f"/playlists/{safe_playlist_id}/items",
            body={"uris": list(batch)},
            expected_status=201,
        )
        snapshot_id = payload.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise SpotifyPlaylistError("Spotify returned an invalid playlist snapshot ID")
        return snapshot_id
