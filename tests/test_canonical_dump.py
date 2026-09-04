import csv
import tarfile
from pathlib import Path

from myusic_engine.matching import (
    ExternalIdentityPolicy,
    TrackQuery,
    build_canonical_dump_mapper,
    resolve_external_identities,
)

ARTIST_MBID = "00000000-0000-4000-8000-000000000010"
RELEASE_MBID = "00000000-0000-4000-8000-000000000020"
FIRST_MBID = "00000000-0000-4000-8000-000000000030"
SECOND_MBID = "00000000-0000-4000-8000-000000000040"
THIRD_MBID = "00000000-0000-4000-8000-000000000050"
FOURTH_MBID = "00000000-0000-4000-8000-000000000060"

HEADER = (
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


def _write_dump(path: Path, *, separator_in_release: bool = False) -> Path:
    rows = (
        (
            1,
            1,
            ARTIST_MBID,
            "Beyoncé",
            RELEASE_MBID,
            "Album A",
            FIRST_MBID,
            "Song A",
            "beyoncesonga",
            1,
        ),
        (
            2,
            2,
            ARTIST_MBID,
            "Artist B",
            RELEASE_MBID,
            "Album B",
            SECOND_MBID,
            "Song B",
            "artistbsongb",
            2,
        ),
        (
            3,
            2,
            ARTIST_MBID,
            "Artist B",
            RELEASE_MBID,
            "Album B",
            THIRD_MBID,
            "Song B",
            "artistbsongb",
            3,
        ),
        (
            4,
            3,
            ARTIST_MBID,
            "Artist C",
            RELEASE_MBID,
            "Release One\u2028Subtitle" if separator_in_release else "Release One",
            FOURTH_MBID,
            "Song C",
            "artistcsongc",
            4,
        ),
        (
            5,
            3,
            ARTIST_MBID,
            "Artist C",
            RELEASE_MBID,
            "Release Two",
            FOURTH_MBID,
            "Song C",
            "artistcsongc",
            5,
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def _query(
    track_id: str,
    artist_name: str,
    track_name: str,
    album_name: str | None,
    play_count: int,
) -> TrackQuery:
    return TrackQuery(
        source_track_id=track_id,
        source_identity_source="metadata_hash",
        track_uri=None,
        track_name=track_name,
        artist_name=artist_name,
        album_name=album_name,
        play_count=play_count,
    )


def test_canonical_csv_scan_keeps_unique_exact_matches_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    queries = (
        _query("a", "Beyoncé", "Song A", "Album A", 3),
        _query("b", "Artist B", "Song B", "Album B", 2),
        _query("c", "Artist C", "Song C", None, 1),
    )
    progress: list[int] = []

    mapper, scan = build_canonical_dump_mapper(
        queries,
        _write_dump(tmp_path / "canonical_musicbrainz_data.csv"),
        progress=progress.append,
        progress_interval=2,
    )
    result = resolve_external_identities(
        queries,
        mapper,
        policy=ExternalIdentityPolicy(
            policy_version="musicbrainz_canonical_dump_exact_metadata_v1"
        ),
        provider_name=scan.provider_name,
    )
    by_id = {match.source_track_id: match for match in result.matches}

    assert by_id["a"].recording_mbid == FIRST_MBID
    assert by_id["c"].recording_mbid == FOURTH_MBID
    assert by_id["b"].match_status == "unmatched"
    # Song C is the same recording under two releases (scores 4 and 5); the higher score
    # ("Release Two") is the documented canonical/preferred row and must win the tie-break.
    resolved = mapper.lookup(artist_name="Artist C", recording_name="Song C", release_name=None)
    assert resolved is not None
    assert resolved.release_name == "Release Two"
    assert {match.provider for match in result.matches} == {"musicbrainz_canonical_dump_local"}
    assert scan.rows_scanned == 5
    assert scan.exact_query_keys == 2
    assert scan.ambiguous_query_keys == 1
    assert scan.candidate_rows_retained == 5
    assert progress == [2, 4, 5]


def test_canonical_scan_limit_targets_highest_play_count_only(tmp_path: Path) -> None:
    mapper, scan = build_canonical_dump_mapper(
        (
            _query("low", "Artist C", "Song C", None, 1),
            _query("high", "Beyoncé", "Song A", "Album A", 10),
        ),
        _write_dump(tmp_path / "canonical_musicbrainz_data.csv"),
        maximum_tracks=1,
    )

    assert scan.query_tracks_selected == 1
    assert scan.query_keys_with_metadata == 1
    assert (
        mapper.lookup(artist_name="Beyoncé", recording_name="Song A", release_name="Album A")
        is not None
    )
    assert mapper.lookup(artist_name="Artist C", recording_name="Song C", release_name=None) is None


def test_canonical_tar_scan_reads_a_non_seekable_member(tmp_path: Path) -> None:
    csv_path = _write_dump(tmp_path / "canonical_musicbrainz_data.csv", separator_in_release=True)
    archive_path = tmp_path / "canonical.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        archive.add(csv_path, arcname="nested/canonical_musicbrainz_data.csv")

    mapper, scan = build_canonical_dump_mapper(
        (_query("c", "Artist C", "Song C", "Release One\u2028Subtitle", 1),),
        archive_path,
    )

    assert scan.rows_scanned == 5
    assert (
        mapper.lookup(
            artist_name="Artist C",
            recording_name="Song C",
            release_name="Release One\u2028Subtitle",
        )
        is not None
    )
