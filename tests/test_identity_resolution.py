import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from myusic_engine.cli import main
from myusic_engine.matching import (
    IdentityInputError,
    IdentityPolicy,
    TrackQuery,
    load_account_catalog,
    load_account_playlist,
    load_identity_policy,
    read_track_queries,
    resolve_identities,
    review_sample,
    write_identity_resolution,
)
from myusic_engine.ranking import read_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRST_URI = "spotify:track:0000000000000000000001"
SECOND_URI = "spotify:track:0000000000000000000002"
THIRD_URI = "spotify:track:0000000000000000000003"
FOURTH_URI = "spotify:track:0000000000000000000004"
FIFTH_URI = "spotify:track:0000000000000000000005"


def _write_account_catalog(tmp_path: Path) -> Path:
    archive_path = tmp_path / "account-data.zip"
    library = {
        "tracks": [
            {
                "uri": FIRST_URI,
                "track": "Exact Song",
                "artist": "Test Artist",
                "album": "First Album",
            },
            {
                "uri": SECOND_URI,
                "track": "Shared Song",
                "artist": "Ambiguous Artist",
                "album": "Edition One",
            },
            {
                "uri": THIRD_URI,
                "track": "Shared Song",
                "artist": "Ambiguous Artist",
                "album": "Edition Two",
            },
            {
                "uri": FOURTH_URI,
                "track": "Fuzzy Song (Remastered 2025)",
                "artist": "Fuzzy Artist",
                "album": "Fuzzy Album",
            },
        ],
        "bannedTracks": [],
    }
    playlists = {
        "playlists": [
            {
                "name": "Synthetic Playlist",
                "items": [
                    {
                        "track": {
                            "trackUri": FIRST_URI,
                            "trackName": "Exact Song",
                            "artistName": "Test Artist",
                            "albumName": "First Album",
                        }
                    },
                    {
                        "track": {
                            "trackUri": FIFTH_URI,
                            "trackName": "Playlist Song",
                            "artistName": "Playlist Artist",
                            "albumName": "Playlist Album",
                        }
                    },
                    {"localTrack": {"uri": "spotify:local:synthetic"}},
                    {"episode": {"episodeUri": "spotify:episode:synthetic"}},
                ],
            }
        ]
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Spotify Account Data/YourLibrary.json", json.dumps(library))
        archive.writestr("Spotify Account Data/Playlist1.json", json.dumps(playlists))
        archive.writestr("Spotify Account Data/Identity.json", '{"username": "do-not-read"}')
    return archive_path


def _query(
    source_track_id: str,
    track_name: str | None,
    artist_name: str | None,
    album_name: str | None = None,
) -> TrackQuery:
    return TrackQuery(
        source_track_id=source_track_id,
        source_identity_source="metadata_hash",
        track_uri=None,
        track_name=track_name,
        artist_name=artist_name,
        album_name=album_name,
    )


def test_account_catalog_reads_library_and_playlist_tracks_only(tmp_path: Path) -> None:
    catalog = load_account_catalog(_write_account_catalog(tmp_path))

    assert catalog.source_files == ("YourLibrary.json", "Playlist1.json")
    assert catalog.records_seen == 6
    assert catalog.duplicates_removed == 1
    assert len(catalog.tracks) == catalog.unique_track_count == 5
    assert {track.source_collection for track in catalog.tracks} == {
        "saved_library",
        "playlist",
    }


def test_named_account_playlist_becomes_rankable_candidates(tmp_path: Path) -> None:
    archive_path = _write_account_catalog(tmp_path)
    playlist = load_account_playlist(archive_path, "synthetic playlist")

    assert playlist.playlist_name == "Synthetic Playlist"
    assert playlist.items_seen == 4
    assert playlist.non_track_items_skipped == 2
    assert playlist.duplicate_tracks_removed == 0
    assert [track.track_uri for track in playlist.tracks] == [FIRST_URI, FIFTH_URI]

    output_dir = tmp_path / "playlist-import"
    exit_code = main(
        [
            "import-account-playlist",
            str(archive_path),
            "--playlist-name",
            "Synthetic Playlist",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    candidates = read_candidates(output_dir / "candidates.jsonl")
    assert [candidate.track_id for candidate in candidates] == [FIRST_URI, FIFTH_URI]
    report = json.loads(
        (output_dir / "account_playlist_import_report.json").read_text(encoding="utf-8")
    )
    assert report["tracks_written"] == 2
    assert report["non_track_items_skipped"] == 2


def test_named_account_playlist_rejects_missing_name(tmp_path: Path) -> None:
    with pytest.raises(IdentityInputError, match="does not contain"):
        load_account_playlist(_write_account_catalog(tmp_path), "Not Here")


def test_resolution_keeps_exact_fuzzy_ambiguous_and_unmatched_distinct(
    tmp_path: Path,
) -> None:
    catalog = load_account_catalog(_write_account_catalog(tmp_path))
    existing_uri = TrackQuery(
        source_track_id=FIFTH_URI,
        source_identity_source="spotify_uri",
        track_uri=FIFTH_URI,
        track_name="Playlist Song",
        artist_name="Playlist Artist",
        album_name="Playlist Album",
    )
    queries = (
        existing_uri,
        _query("unresolved:exact", "Exact Song", "Test Artist"),
        _query("unresolved:album", "Shared Song", "Ambiguous Artist", "Edition One"),
        _query("unresolved:ambiguous", "Shared Song", "Ambiguous Artist"),
        _query("unresolved:fuzzy", "Fuzzy Song Remastered 2024", "Fuzzy Artist"),
        _query("unresolved:none", "Unknown Song", "Unknown Artist"),
    )

    result = resolve_identities(queries, catalog)
    by_id = {match.source_track_id: match for match in result.matches}

    assert by_id[FIFTH_URI].match_method == "existing_spotify_uri"
    assert by_id[FIFTH_URI].resolved_track_id == FIFTH_URI
    assert by_id["unresolved:exact"].resolved_track_id == FIRST_URI
    assert by_id["unresolved:exact"].match_confidence == 0.9
    assert by_id["unresolved:album"].resolved_track_id == SECOND_URI
    assert by_id["unresolved:album"].match_confidence == 1.0
    assert by_id["unresolved:ambiguous"].match_status == "ambiguous"
    assert by_id["unresolved:ambiguous"].resolved_track_id is None
    assert len(by_id["unresolved:ambiguous"].candidates) == 2
    assert by_id["unresolved:fuzzy"].match_status == "fuzzy"
    assert by_id["unresolved:fuzzy"].resolved_track_id is None
    assert by_id["unresolved:fuzzy"].review_required is True
    assert by_id["unresolved:none"].match_status == "unmatched"
    assert result.report.status_counts == {
        "exact": 3,
        "fuzzy": 1,
        "ambiguous": 1,
        "unmatched": 1,
    }
    assert result.report.resolved_count == 3
    assert result.report.review_required_count == 3


def test_album_mismatch_is_not_silently_accepted(tmp_path: Path) -> None:
    catalog = load_account_catalog(_write_account_catalog(tmp_path))
    query = _query("unresolved:album-mismatch", "Exact Song", "Test Artist", "Other Album")

    match = resolve_identities([query], catalog).matches[0]

    assert match.match_status == "fuzzy"
    assert match.resolved_track_id is None
    assert match.candidates[0].track_uri == FIRST_URI


def test_project_identity_policy_loads_as_versioned_defaults() -> None:
    loaded = load_identity_policy(PROJECT_ROOT / "configs" / "identity_resolution.yaml")

    assert loaded == IdentityPolicy()


def test_track_query_reader_validates_identity_fields_without_echoing_values(
    tmp_path: Path,
) -> None:
    affinity_path = tmp_path / "affinity.jsonl"
    affinity_path.write_text(
        json.dumps(
            {
                "track_id": "unresolved:synthetic",
                "track_identity_source": "metadata_hash",
                "track_uri": None,
                "track_name": "Synthetic Song",
                "artist_name": "Synthetic Artist",
                "album_name": None,
                "play_count": 2,
                "total_ms_played": 2000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    queries = read_track_queries(affinity_path)

    assert queries[0].source_track_id == "unresolved:synthetic"
    assert queries[0].play_count == 2
    assert queries[0].total_ms_played == 2000
    affinity_path.write_text(
        json.dumps(
            {
                "track_id": "private-invalid-value",
                "track_identity_source": "spotify_uri",
                "track_uri": "private-invalid-value",
                "play_count": 1,
                "total_ms_played": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IdentityInputError) as error:
        read_track_queries(affinity_path)
    assert "private-invalid-value" not in str(error.value)


def test_resolution_outputs_and_review_sample_are_deterministic(tmp_path: Path) -> None:
    catalog = load_account_catalog(_write_account_catalog(tmp_path))
    queries = (
        _query("unresolved:exact", "Exact Song", "Test Artist"),
        _query("unresolved:ambiguous", "Shared Song", "Ambiguous Artist"),
        _query("unresolved:fuzzy", "Fuzzy Song Remastered 2024", "Fuzzy Artist"),
        _query("unresolved:none", "Unknown Song", "Unknown Artist"),
    )
    result = resolve_identities(queries, catalog)

    first_sample = review_sample(result.matches, sample_size_per_status=1)
    second_sample = review_sample(reversed(result.matches), sample_size_per_status=1)
    assert first_sample == second_sample

    matches_path, report_path, review_path = write_identity_resolution(
        result, tmp_path / "output", review_sample_per_status=1
    )
    assert len(matches_path.read_text(encoding="utf-8").splitlines()) == 4
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["policy_version"] == "offline_spotify_account_catalog_v1"
    assert report["catalog_unique_tracks"] == 5
    assert len(review_path.read_text(encoding="utf-8").splitlines()) == 4


def test_cli_resolves_identities_from_local_account_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    affinity_path = tmp_path / "affinity.jsonl"
    rows = [
        {
            "track_id": "unresolved:exact",
            "track_identity_source": "metadata_hash",
            "track_uri": None,
            "track_name": "Exact Song",
            "artist_name": "Test Artist",
            "album_name": None,
            "play_count": 4,
            "total_ms_played": 4000,
        },
        {
            "track_id": "unresolved:none",
            "track_identity_source": "metadata_hash",
            "track_uri": None,
            "track_name": "Unknown Song",
            "artist_name": "Unknown Artist",
            "album_name": None,
            "play_count": 1,
            "total_ms_played": 1000,
        },
    ]
    affinity_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_directory = tmp_path / "identity"

    exit_code = main(
        [
            "resolve-identities",
            str(affinity_path),
            str(_write_account_catalog(tmp_path)),
            "--output-dir",
            str(output_directory),
            "--matching-config",
            str(PROJECT_ROOT / "configs" / "identity_resolution.yaml"),
        ]
    )

    assert exit_code == 0
    assert (output_directory / "identity_matches.jsonl").exists()
    assert (output_directory / "identity_resolution_report.json").exists()
    assert (output_directory / "identity_review_sample.jsonl").exists()
    report = json.loads(
        (output_directory / "identity_resolution_report.json").read_text(encoding="utf-8")
    )
    assert report["resolved_play_rate"] == 0.8
    assert report["resolved_ms_played_rate"] == 0.8
    output = capsys.readouterr().out
    assert "Resolved 1 of 2 tracks" in output
