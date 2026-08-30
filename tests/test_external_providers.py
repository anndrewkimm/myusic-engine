import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from myusic_engine.features import (
    fetch_acousticbrainz_features,
    read_feature_observations,
    write_acousticbrainz_result,
)
from myusic_engine.matching import (
    ExternalIdentityMatch,
    ExternalIdentityPolicy,
    TrackQuery,
    read_external_identity_matches,
    resolve_external_identities,
    write_external_identity_resolution,
)
from myusic_engine.providers import AcousticBrainzDocument, ListenBrainzMapping

FIRST_MBID = "00000000-0000-4000-8000-000000000001"
SECOND_MBID = "00000000-0000-4000-8000-000000000002"
RELEASE_MBID = "00000000-0000-4000-8000-000000000003"
ARTIST_MBID = "00000000-0000-4000-8000-000000000004"


class FakeMapper:
    def lookup(
        self,
        *,
        artist_name: str,
        recording_name: str,
        release_name: str | None,
    ) -> ListenBrainzMapping | None:
        if recording_name == "Missing Song":
            return None
        mapped_release = "Mapped Album" if recording_name == "Album Mismatch" else release_name
        return ListenBrainzMapping(
            recording_mbid=FIRST_MBID if recording_name != "Second Song" else SECOND_MBID,
            recording_name=recording_name,
            artist_credit_name=artist_name,
            release_mbid=RELEASE_MBID,
            release_name=mapped_release,
            artist_mbids=(ARTIST_MBID,),
            confidence=0.98,
        )


def _query(
    track_id: str,
    track_name: str,
    *,
    album_name: str | None,
    play_count: int,
) -> TrackQuery:
    return TrackQuery(
        source_track_id=track_id,
        source_identity_source="spotify_uri",
        track_uri=track_id,
        track_name=track_name,
        artist_name="Synthetic Artist",
        album_name=album_name,
        play_count=play_count,
        total_ms_played=play_count * 1000,
    )


def test_external_mapping_accepts_only_exact_metadata_and_reports_weighted_coverage(
    tmp_path: Path,
) -> None:
    queries = (
        _query("spotify:track:1", "Exact Song", album_name="Exact Album", play_count=8),
        _query("spotify:track:2", "Second Song", album_name=None, play_count=4),
        _query("spotify:track:3", "Album Mismatch", album_name="Other Album", play_count=2),
        _query("spotify:track:4", "Missing Song", album_name="Missing Album", play_count=1),
    )

    result = resolve_external_identities(queries, FakeMapper())
    by_id = {match.source_track_id: match for match in result.matches}

    assert by_id["spotify:track:1"].match_status == "exact"
    assert by_id["spotify:track:1"].recording_mbid == FIRST_MBID
    assert by_id["spotify:track:2"].match_method == "exact_title_artist"
    assert by_id["spotify:track:3"].match_status == "fuzzy"
    assert by_id["spotify:track:3"].recording_mbid is None
    assert by_id["spotify:track:4"].match_status == "unmatched"
    assert result.report.status_counts == {
        "exact": 2,
        "fuzzy": 1,
        "ambiguous": 0,
        "unmatched": 1,
    }
    assert result.report.exact_play_rate == 0.8

    paths = write_external_identity_resolution(
        result,
        tmp_path / "identity",
        review_sample_per_status=1,
    )
    assert all(path.is_file() for path in paths)
    assert read_external_identity_matches(paths[0]) == result.matches
    review_rows = paths[2].read_text(encoding="utf-8").splitlines()
    assert len(review_rows) == 3


def test_external_mapping_limit_prioritizes_play_count() -> None:
    result = resolve_external_identities(
        (
            _query("spotify:track:low", "Exact Song", album_name="Album", play_count=1),
            _query("spotify:track:high", "Second Song", album_name="Album", play_count=20),
        ),
        FakeMapper(),
        maximum_tracks=1,
    )

    assert [match.source_track_id for match in result.matches] == ["spotify:track:high"]
    assert result.report.queries_available == 2
    assert result.report.queries_processed == 1


def _exact_match(track_id: str, mbid: str, play_count: int) -> ExternalIdentityMatch:
    return ExternalIdentityMatch(
        policy_version=ExternalIdentityPolicy().policy_version,
        source_track_id=track_id,
        source_track_uri=track_id,
        track_name="Synthetic Song",
        artist_name="Synthetic Artist",
        album_name="Synthetic Album",
        play_count=play_count,
        total_ms_played=play_count * 10_000,
        match_status="exact",
        match_method="exact_title_artist_release",
        recording_mbid=mbid,
        release_mbid=RELEASE_MBID,
        artist_mbids=(ARTIST_MBID,),
        provider="synthetic_provider",
        provider_confidence=1.0,
        match_confidence=1.0,
        mapped_recording_name="Synthetic Song",
        mapped_artist_name="Synthetic Artist",
        mapped_release_name="Synthetic Album",
        review_required=False,
    )


def _low_level() -> Mapping[str, object]:
    return {
        "metadata": {
            "audio_properties": {"length": 180.0, "replay_gain": -7.0},
        },
        "lowlevel": {"average_loudness": 0.9, "dynamic_complexity": 2.5},
        "rhythm": {
            "beats_count": 360,
            "beats_loudness": {"mean": 0.4},
            "bpm": 120.0,
            "danceability": 0.7,
            "onset_rate": 4.0,
        },
        "tonal": {
            "chords_changes_rate": 0.1,
            "key_key": "C",
            "key_scale": "major",
            "key_strength": 0.8,
            "tuning_equal_tempered_deviation": 0.02,
            "tuning_frequency": 440.0,
        },
    }


def _high_level() -> Mapping[str, object]:
    tasks = {
        "danceability": ("danceable", 0.8),
        "mood_acoustic": ("acoustic", 0.2),
        "mood_aggressive": ("aggressive", 0.4),
        "mood_happy": ("happy", 0.7),
        "mood_relaxed": ("relaxed", 0.3),
        "voice_instrumental": ("instrumental", 0.6),
    }
    return {
        "metadata": {"audio_properties": {"length": 180.0}},
        "highlevel": {
            task: {"all": {positive_class: score}}
            for task, (positive_class, score) in tasks.items()
        },
    }


class FakeAcousticProvider:
    def fetch(self, recording_mbids: Sequence[str]) -> Mapping[str, AcousticBrainzDocument]:
        return {
            mbid: AcousticBrainzDocument(
                recording_mbid=mbid,
                low_level=_low_level(),
                high_level=_high_level(),
            )
            for mbid in recording_mbids
            if mbid == FIRST_MBID
        }


def test_acousticbrainz_conversion_is_source_tagged_and_reports_missing_coverage(
    tmp_path: Path,
) -> None:
    matches = (
        _exact_match("spotify:track:1", FIRST_MBID, 9),
        _exact_match("spotify:track:2", SECOND_MBID, 1),
    )

    result = fetch_acousticbrainz_features(matches, FakeAcousticProvider())

    assert result.report.exact_tracks_considered == 2
    assert result.report.low_level_tracks_covered == 1
    assert result.report.low_level_play_rate == 0.9
    assert result.report.high_level_tracks_covered == 1
    assert {observation.track_id for observation in result.observations} == {
        "spotify:track:1"
    }
    feature_names = {observation.feature_name for observation in result.observations}
    assert "tempo_bpm_estimate_v1" in feature_names
    assert "acousticbrainz_descriptor_vector_v1" in feature_names
    assert "acousticbrainz_learned_audio_vector_v1" in feature_names
    assert all(
        observation.feature_source == "acousticbrainz_cc0"
        for observation in result.observations
    )

    features_path, report_path = write_acousticbrainz_result(result, tmp_path / "features")
    assert read_feature_observations(features_path) == result.observations
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["low_level_play_rate"] == pytest.approx(0.9)
