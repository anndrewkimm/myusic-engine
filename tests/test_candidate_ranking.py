from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from myusic_engine.cli import main
from myusic_engine.clustering import TasteMapAssignment
from myusic_engine.features import FeatureObservation, FeatureSelector
from myusic_engine.modeling import (
    BEHAVIOR_FEATURE_NAMES,
    AudioFeatureProfile,
    AudioInputSpec,
    BehaviorSnapshot,
    LinearTasteModel,
)
from myusic_engine.ranking import (
    CandidateTrack,
    RecommendationConfig,
    RecommendationError,
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


def _behavior_model() -> LinearTasteModel:
    provisional = LinearTasteModel(
        model_id="pending",
        model_name="behavior_test",
        model_version="taste_logistic_test_v1",
        dataset_version="synthetic_temporal_v1",
        feature_names=("behavior:prior_seen",),
        means=(0.0,),
        scales=(1.0,),
        coefficients=(5.0,),
        intercept=0.0,
        includes_behavior=True,
        includes_descriptors=False,
        includes_embedding=False,
        profile_name=None,
        profile_version=None,
        training_rows=10,
        training_positive_rate=0.5,
        training_period_end="2024-01-01T00:00:00.000Z",
    )
    canonical = json.dumps(
        provisional.to_dict(include_model_id=False),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return replace(
        provisional,
        model_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _snapshot(
    track_id: str,
    *,
    dataset_version: str = "synthetic_temporal_v1",
    artist_key: str | None = None,
    track_play_count: float = 0.0,
    artist_play_count: float = 0.0,
) -> BehaviorSnapshot:
    behavior = [0.0] * len(BEHAVIOR_FEATURE_NAMES)
    behavior[BEHAVIOR_FEATURE_NAMES.index("prior_seen")] = 1.0
    behavior[BEHAVIOR_FEATURE_NAMES.index("prior_log_play_count")] = track_play_count
    behavior[BEHAVIOR_FEATURE_NAMES.index("prior_artist_log_play_count")] = artist_play_count
    return BehaviorSnapshot(
        track_id=track_id,
        artist_key=artist_key,
        as_of="2024-02-01T00:00:00.000Z",
        behavior_features=tuple(behavior),
        dataset_version=dataset_version,
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


def test_behavior_only_ranking_needs_no_audio_and_hashes_all_inputs() -> None:
    seen = "spotify:track:0000000000000000000001"
    cold = "spotify:track:0000000000000000000002"
    candidates = (
        CandidateTrack(seen, seen, "Seen", "Artist A"),
        CandidateTrack(cold, cold, "Cold", "Artist B"),
    )
    result = rank_candidates(
        candidates,
        (),
        model=_behavior_model(),
        behavior_snapshots=(_snapshot(seen),),
        top_k=2,
    )

    ranked = [item for item in result.recommendations if item.rank is not None]
    assert [item.candidate.track_id for item in ranked] == [seen, cold]
    assert {item.tier for item in ranked} == {"preference_ranked"}
    assert result.report.profile_name is None
    assert result.report.behavior_dataset_version == "synthetic_temporal_v1"
    assert all(len(value) == 64 for value in result.report.to_dict()["input_digests"].values())
    assert all(item.final_score == item.predicted_preference for item in ranked)

    changed_metadata = rank_candidates(
        (candidates[0], replace(candidates[1], artist_name="Different Artist")),
        (),
        model=_behavior_model(),
        behavior_snapshots=(_snapshot(seen),),
        top_k=2,
    )
    changed_snapshot = rank_candidates(
        candidates,
        (),
        model=_behavior_model(),
        behavior_snapshots=(_snapshot(cold),),
        top_k=2,
    )
    assert changed_metadata.report.run_id != result.report.run_id
    assert changed_snapshot.report.run_id != result.report.run_id


def test_behavior_only_ranking_rejects_dataset_mismatch() -> None:
    track_id = "spotify:track:0000000000000000000001"
    with pytest.raises(RecommendationError, match="different datasets"):
        rank_candidates(
            (CandidateTrack(track_id, track_id),),
            (),
            model=_behavior_model(),
            behavior_snapshots=(_snapshot(track_id, dataset_version="other_dataset"),),
        )


def test_behavior_ranking_accepts_distinct_track_history_for_one_artist() -> None:
    first = "spotify:track:0000000000000000000001"
    second = "spotify:track:0000000000000000000002"
    shared_artist = "artist:000000000000000000000000"

    result = rank_candidates(
        (
            CandidateTrack(first, first, artist_name="Artist A"),
            CandidateTrack(second, second, artist_name="Artist A"),
        ),
        (),
        model=_behavior_model(),
        behavior_snapshots=(
            _snapshot(first, artist_key=shared_artist, track_play_count=1.0, artist_play_count=3.0),
            _snapshot(
                second, artist_key=shared_artist, track_play_count=2.0, artist_play_count=3.0
            ),
        ),
    )

    assert result.report.ranked_count == 2


def test_behavior_ranking_rejects_inconsistent_artist_history() -> None:
    first = "spotify:track:0000000000000000000001"
    second = "spotify:track:0000000000000000000002"
    shared_artist = "artist:000000000000000000000000"

    with pytest.raises(RecommendationError, match="shared artist"):
        rank_candidates(
            (CandidateTrack(first, first), CandidateTrack(second, second)),
            (),
            behavior_snapshots=(
                _snapshot(first, artist_key=shared_artist, artist_play_count=3.0),
                _snapshot(second, artist_key=shared_artist, artist_play_count=4.0),
            ),
        )


def test_cli_ranks_behavior_only_candidates_without_feature_files(tmp_path) -> None:
    seen = "spotify:track:0000000000000000000001"
    cold = "spotify:track:0000000000000000000002"
    candidates_path = tmp_path / "candidates.txt"
    candidates_path.write_text(f"{seen}\n{cold}\n", encoding="utf-8")
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(_behavior_model().to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshots_path = tmp_path / "snapshots.jsonl"
    snapshots_path.write_text(json.dumps(_snapshot(seen).to_dict()) + "\n", encoding="utf-8")
    output_dir = tmp_path / "recommendations"

    exit_code = main(
        [
            "rank-candidates",
            str(candidates_path),
            "--model",
            str(model_path),
            "--behavior-snapshots",
            str(snapshots_path),
            "--top-k",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "spotify_playlist_uris.txt").read_text(encoding="utf-8").splitlines() == [
        seen,
        cold,
    ]
