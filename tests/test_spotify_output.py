import json
from collections.abc import Sequence
from pathlib import Path
from urllib.request import Request

import pytest

import myusic_engine.spotify_output.playlist as playlist_module
from myusic_engine.cli import main
from myusic_engine.spotify_output import (
    CreatedPlaylist,
    PlaylistPublicationReceipt,
    SpotifyPlaylistError,
    SpotifyWebApiClient,
    create_publication_plan,
    publish_playlist,
    read_publication_plan,
    read_publication_receipt,
    read_spotify_uri_file,
    write_publication_plan,
    write_publication_receipt,
)


def _uri(index: int) -> str:
    return f"spotify:track:{index:022d}"


class _FakeGateway:
    def __init__(self, remote_uris: Sequence[str] = ()) -> None:
        self.remote_uris = list(remote_uris)
        self.create_calls = 0
        self.added_batches: list[tuple[str, ...]] = []

    def create_private_playlist(self, name: str, description: str) -> CreatedPlaylist:
        assert name == "Private recommendations"
        assert description == "A deterministic test"
        self.create_calls += 1
        return CreatedPlaylist(
            playlist_id="P" * 22,
            playlist_uri=f"spotify:playlist:{'P' * 22}",
            playlist_url=f"https://open.spotify.com/playlist/{'P' * 22}",
        )

    def playlist_track_uris(self, playlist_id: str) -> tuple[str, ...]:
        assert playlist_id == "P" * 22
        return tuple(self.remote_uris)

    def add_playlist_items(self, playlist_id: str, uris: Sequence[str]) -> str:
        assert playlist_id == "P" * 22
        batch = tuple(uris)
        assert 1 <= len(batch) <= 100
        self.added_batches.append(batch)
        self.remote_uris.extend(batch)
        return f"snapshot-{len(self.added_batches)}"


class _Response:
    def __init__(self, payload: object, status: int) -> None:
        self.status = status
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int = -1) -> bytes:
        return self._raw


def test_publication_plan_round_trips_and_uri_handoff_is_strict(tmp_path: Path) -> None:
    uri_path = tmp_path / "spotify_playlist_uris.txt"
    uri_path.write_text(f"{_uri(1)}\n{_uri(2)}\n", encoding="utf-8")

    plan = create_publication_plan(
        read_spotify_uri_file(uri_path),
        playlist_name=" Private recommendations ",
        description=" A deterministic test ",
    )
    plan_path = write_publication_plan(plan, tmp_path / "plan.json")

    assert plan.playlist_name == "Private recommendations"
    assert plan.item_count == 2
    assert read_publication_plan(plan_path) == plan
    assert "access_token" not in plan_path.read_text(encoding="utf-8")

    uri_path.write_text(f"{_uri(1)}\n{_uri(1)}\n", encoding="utf-8")
    with pytest.raises(SpotifyPlaylistError, match="duplicate URI"):
        read_spotify_uri_file(uri_path)


def test_publication_batches_at_official_limit_and_confirms_remote_state(tmp_path: Path) -> None:
    uris = tuple(_uri(index) for index in range(205))
    plan = create_publication_plan(
        uris,
        playlist_name="Private recommendations",
        description="A deterministic test",
    )
    gateway = _FakeGateway()
    receipt_path = tmp_path / "receipt.json"

    receipt = publish_playlist(plan, gateway, receipt_path)

    assert gateway.create_calls == 1
    assert [len(batch) for batch in gateway.added_batches] == [100, 100, 5]
    assert tuple(gateway.remote_uris) == uris
    assert receipt.status == "complete"
    assert receipt.confirmed_item_count == 205
    assert read_publication_receipt(receipt_path) == receipt


def test_publication_reconciles_remote_prefix_before_resuming(tmp_path: Path) -> None:
    uris = tuple(_uri(index) for index in range(205))
    plan = create_publication_plan(
        uris,
        playlist_name="Private recommendations",
        description="A deterministic test",
    )
    receipt_path = tmp_path / "receipt.json"
    write_publication_receipt(
        PlaylistPublicationReceipt(
            plan_id=plan.plan_id,
            playlist_id="P" * 22,
            playlist_uri=f"spotify:playlist:{'P' * 22}",
            playlist_url=None,
            requested_item_count=205,
            confirmed_item_count=100,
            status="publishing",
            latest_snapshot_id="old-snapshot",
        ),
        receipt_path,
    )
    gateway = _FakeGateway(uris[:150])

    receipt = publish_playlist(plan, gateway, receipt_path)

    assert gateway.create_calls == 0
    assert gateway.added_batches == [uris[150:]]
    assert receipt.confirmed_item_count == 205


def test_publication_refuses_remote_drift(tmp_path: Path) -> None:
    uris = tuple(_uri(index) for index in range(3))
    plan = create_publication_plan(
        uris,
        playlist_name="Private recommendations",
        description="A deterministic test",
    )
    receipt_path = tmp_path / "receipt.json"
    write_publication_receipt(
        PlaylistPublicationReceipt(
            plan_id=plan.plan_id,
            playlist_id="P" * 22,
            playlist_uri=f"spotify:playlist:{'P' * 22}",
            playlist_url=None,
            requested_item_count=3,
            confirmed_item_count=1,
            status="publishing",
        ),
        receipt_path,
    )
    gateway = _FakeGateway((_uri(999),))

    with pytest.raises(SpotifyPlaylistError, match="not an exact prefix"):
        publish_playlist(plan, gateway, receipt_path)

    assert gateway.added_batches == []


def test_web_api_client_uses_current_private_playlist_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    playlist_id = "P" * 22
    responses = iter(
        (
            _Response(
                {
                    "id": playlist_id,
                    "public": False,
                    "external_urls": {
                        "spotify": f"https://open.spotify.com/playlist/{playlist_id}"
                    },
                },
                201,
            ),
            _Response(
                {
                    "items": [
                        {"item": {"type": "track", "uri": _uri(1)}},
                        {"item": {"type": "track", "uri": _uri(2)}},
                    ],
                    "next": None,
                    "total": 2,
                },
                200,
            ),
            _Response({"snapshot_id": "new-snapshot"}, 201),
        )
    )
    requests: list[Request] = []

    def fake_urlopen(request: Request, *, timeout: float) -> _Response:
        assert timeout == 12.0
        requests.append(request)
        return next(responses)

    monkeypatch.setattr(playlist_module, "urlopen", fake_urlopen)
    client = SpotifyWebApiClient(
        "test-token-do-not-store",
        timeout_seconds=12,
    )

    created = client.create_private_playlist("Private list", "Description")
    remote = client.playlist_track_uris(playlist_id)
    snapshot = client.add_playlist_items(playlist_id, (_uri(3),))

    assert created.playlist_uri == f"spotify:playlist:{playlist_id}"
    assert remote == (_uri(1), _uri(2))
    assert snapshot == "new-snapshot"
    assert [request.get_method() for request in requests] == ["POST", "GET", "POST"]
    assert requests[0].full_url == "https://api.spotify.com/v1/me/playlists"
    assert requests[1].full_url == (
        f"https://api.spotify.com/v1/playlists/{playlist_id}/items?limit=50&offset=0"
    )
    assert requests[2].full_url == f"https://api.spotify.com/v1/playlists/{playlist_id}/items"
    assert all(
        request.get_header("Authorization") == "Bearer test-token-do-not-store"
        for request in requests
    )
    assert json.loads(requests[0].data or b"{}") == {
        "name": "Private list",
        "description": "Description",
        "public": False,
    }


def test_publish_cli_is_dry_run_by_default_and_requires_environment_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    uri_path = tmp_path / "spotify_playlist_uris.txt"
    uri_path.write_text(f"{_uri(1)}\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    arguments = [
        "publish-spotify-playlist",
        str(uri_path),
        "--name",
        "Private list",
        "--output-dir",
        str(output_dir),
    ]

    assert main(arguments) == 0
    assert "no network request was made" in capsys.readouterr().out
    assert (output_dir / "spotify_playlist_plan.json").is_file()
    assert not (output_dir / "spotify_playlist_receipt.json").exists()

    monkeypatch.delenv("SPOTIFY_ACCESS_TOKEN", raising=False)
    assert main([*arguments, "--execute"]) == 2
    assert "SPOTIFY_ACCESS_TOKEN is not set" in capsys.readouterr().err
