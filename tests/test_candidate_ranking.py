from __future__ import annotations

from myusic_engine.clustering import TasteMapAssignment
from myusic_engine.features import FeatureObservation, FeatureSelector
from myusic_engine.modeling import AudioFeatureProfile, AudioInputSpec
from myusic_engine.ranking import (
    CandidateTrack,
    RecommendationConfig,
    rank_candidates,
    read_candidates,
    write_recommendations,
)


def _profile() -> AudioFeatureProfile:
    return AudioFeatureProfile(
        profile_version="synthetic_profile_v1",
        descriptor_inputs=(),
        embedding_input=AudioInputSpec(
            FeatureSelector("synthetic_embedding_v1", "synthetic", "v1"), 3
        ),
    )


def _embedding(track_id: str, vector: tuple[float, ...]) -> FeatureObservation:
    return FeatureObservation(
        track_id=track_id,
        feature_name="synthetic_embedding_v1",
        value=vector,
        feature_source="synthetic",
        source_version="v1",
        coverage_seconds=30.0,
        feature_confidence=1.0,
    )


def test_candidate_intake_accepts_spotify_urls_and_uris(tmp_path) -> None:
    source = tmp_path / "candidates.txt"
    source.write_text(
        "https://open.spotify.com/track/0123456789ABCDEFGHIJKL?si=test\n"
        "spotify:track:ZYXWVUTSRQPONMLKJIHGFE\n",
        encoding="utf-8",
    )
    candidates = read_candidates(source)
    assert candidates[0].track_id == "spotify:track:0123456789ABCDEFGHIJKL"
    assert candidates[0].spotify_uri == candidates[0].track_id
    assert candidates[1].spotify_uri == candidates[1].track_id


def test_candidate_ranking_separates_audio_and_metadata_only_tiers(tmp_path) -> None:
    seed = "spotify:track:0000000000000000000001"
    close = "spotify:track:0000000000000000000002"
    far = "spotify:track:0000000000000000000003"
    missing = "spotify:track:0000000000000000000004"
    observations = (
        _embedding(seed, (1.0, 0.0, 0.0)),
        _embedding(close, (0.9, 0.1, 0.0)),
        _embedding(far, (-1.0, 0.0, 0.0)),
    )
    candidates = (
        CandidateTrack(seed, seed, "Seed", "Artist A"),
        CandidateTrack(close, close, "Close", "Artist B"),
        CandidateTrack(far, far, "Far", "Artist C"),
        CandidateTrack(missing, missing, "Missing", "Artist D"),
    )
    cluster_assignments = tuple(
        TasteMapAssignment(
            track_id=track_id,
            cluster_id=index % 2,
            is_noise=False,
            cluster_confidence=1.0,
            distance_to_cluster_centroid=0.1,
            projection_x=float(index),
            projection_y=0.0,
            model_id="a" * 64,
            profile_name="synthetic",
            profile_version="synthetic_profile_v1",
        )
        for index, track_id in enumerate((seed, close, far))
    )
    result = rank_candidates(
        candidates,
        observations,
        profile=_profile(),
        profile_name="synthetic",
        seed_weights={seed: 1.0},
        cluster_assignments=cluster_assignments,
        config=RecommendationConfig(maximum_per_artist=2),
        top_k=10,
    )

    ranked = [item for item in result.recommendations if item.rank is not None]
    assert [item.candidate.track_id for item in ranked] == [close, far]
    missing_row = next(
        item for item in result.recommendations if item.candidate.track_id == missing
    )
    seed_row = next(item for item in result.recommendations if item.candidate.track_id == seed)
    assert missing_row.tier == "metadata_only"
    assert missing_row.exclusion_reason == "metadata_only_no_selected_audio_or_model_coverage"
    assert seed_row.exclusion_reason == "seed_track"
    assert ranked[0].cluster_model_id == "a" * 64
    assert result.report.seed_cluster_counts == {0: 1}
    assert result.report.tier_counts == {"audio_ranked": 3, "metadata_only": 1}
    paths = write_recommendations(result, tmp_path)
    assert all(path.exists() for path in paths)
    assert paths[2].read_text(encoding="utf-8").splitlines() == [close, far]
