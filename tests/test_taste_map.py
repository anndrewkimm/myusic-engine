from __future__ import annotations

from myusic_engine.clustering import (
    TasteMapConfig,
    build_taste_map,
    read_taste_map_assignments,
    write_taste_map,
)
from myusic_engine.features import FeatureObservation, FeatureSelector
from myusic_engine.modeling import AudioFeatureProfile, AudioInputSpec


def test_taste_map_compares_cluster_families_and_writes_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    descriptor = FeatureSelector("synthetic_descriptor_v1", "synthetic", "v1")
    embedding = FeatureSelector("synthetic_embedding_v1", "synthetic", "v1")
    profile = AudioFeatureProfile(
        profile_version="synthetic_profile_v1",
        descriptor_inputs=(AudioInputSpec(descriptor, 2),),
        embedding_input=AudioInputSpec(embedding, 3),
    )
    observations = []
    centers = ((-5.0, -5.0), (0.0, 5.0), (5.0, -5.0))
    for cluster, center in enumerate(centers):
        for offset in range(10):
            track_id = f"track-{cluster}-{offset}"
            jitter = offset / 100.0
            observations.extend(
                (
                    FeatureObservation(
                        track_id=track_id,
                        feature_name="synthetic_descriptor_v1",
                        value=(center[0] + jitter, center[1] - jitter),
                        feature_source="synthetic",
                        source_version="v1",
                        coverage_seconds=30.0,
                        feature_confidence=1.0,
                    ),
                    FeatureObservation(
                        track_id=track_id,
                        feature_name="synthetic_embedding_v1",
                        value=(center[0], center[1], jitter + 1.0),
                        feature_source="synthetic",
                        source_version="v1",
                        coverage_seconds=30.0,
                        feature_confidence=1.0,
                    ),
                )
            )

    result = build_taste_map(
        observations,
        profile=profile,
        profile_name="synthetic",
        config=TasteMapConfig(
            representation="combined",
            maximum_k=5,
            random_seeds=(11, 17),
            hdbscan_min_cluster_sizes=(5,),
        ),
    )

    assert len(result.assignments) == 30
    assert result.report.input_dimensions == 5
    assert result.report.selected_cluster_count >= 2
    assert {experiment.algorithm for experiment in result.report.experiments} == {
        "kmeans",
        "hdbscan",
    }
    assert result.model.model_id == result.report.model_id
    paths = write_taste_map(result, tmp_path)
    assert all(path.stat().st_size > 0 for path in paths)
    assert read_taste_map_assignments(paths[0]) == result.assignments
