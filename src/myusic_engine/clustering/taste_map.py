"""Reproducible taste-map experiments over exact-provenance audio representations."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from myusic_engine.features import FeatureObservation
from myusic_engine.io import atomic_write_text
from myusic_engine.modeling import AudioFeatureProfile, ProfiledFeatureCatalog
from myusic_engine.privacy import assert_privacy_safe

RepresentationKind: TypeAlias = Literal["descriptors", "embedding", "combined"]


class TasteMapError(ValueError):
    """Raised when a representation cannot support trustworthy clustering."""


@dataclass(frozen=True, slots=True)
class TasteMapConfig:
    """Small deterministic search space for clustering and stability checks."""

    representation: RepresentationKind = "embedding"
    minimum_k: int = 2
    maximum_k: int = 12
    random_seeds: tuple[int, ...] = (1729, 2718, 3141)
    hdbscan_min_cluster_sizes: tuple[int, ...] = (5, 10, 20)
    maximum_hdbscan_noise_rate: float = 0.50
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise TasteMapError("Taste-map config schema_version must be 1")
        if self.minimum_k < 2 or self.maximum_k < self.minimum_k:
            raise TasteMapError("Taste-map K range is invalid")
        if not self.random_seeds or any(seed < 0 for seed in self.random_seeds):
            raise TasteMapError("Taste-map random seeds must be non-negative")
        if len(set(self.random_seeds)) != len(self.random_seeds):
            raise TasteMapError("Taste-map random seeds must be unique")
        if not self.hdbscan_min_cluster_sizes or any(
            size < 2 for size in self.hdbscan_min_cluster_sizes
        ):
            raise TasteMapError("HDBSCAN minimum cluster sizes must be at least two")
        if not 0 <= self.maximum_hdbscan_noise_rate < 1:
            raise TasteMapError("maximum_hdbscan_noise_rate must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ClusteringExperiment:
    """Aggregate quality result for one algorithm/parameter setting."""

    algorithm: str
    parameters: dict[str, object]
    cluster_count: int
    noise_rate: float
    silhouette: float | None
    davies_bouldin: float | None
    stability_ari: float | None
    eligible_for_selection: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "parameters": dict(sorted(self.parameters.items())),
            "cluster_count": self.cluster_count,
            "noise_rate": self.noise_rate,
            "silhouette": self.silhouette,
            "davies_bouldin": self.davies_bouldin,
            "stability_ari": self.stability_ari,
            "eligible_for_selection": self.eligible_for_selection,
        }


@dataclass(frozen=True, slots=True)
class TasteMapAssignment:
    """Private per-track cluster and two-dimensional projection."""

    track_id: str
    cluster_id: int
    is_noise: bool
    cluster_confidence: float
    distance_to_cluster_centroid: float | None
    projection_x: float
    projection_y: float
    model_id: str
    profile_name: str
    profile_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "track_id": self.track_id,
            "cluster_id": self.cluster_id,
            "is_noise": self.is_noise,
            "cluster_confidence": self.cluster_confidence,
            "distance_to_cluster_centroid": self.distance_to_cluster_centroid,
            "projection_x": self.projection_x,
            "projection_y": self.projection_y,
            "model_id": self.model_id,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
        }


@dataclass(frozen=True, slots=True)
class TasteMapModel:
    """JSON-safe preprocessing and centroid artifact for cluster context."""

    model_id: str
    algorithm: str
    representation: RepresentationKind
    profile_name: str
    profile_version: str
    feature_names: tuple[str, ...]
    scaler_means: tuple[float, ...]
    scaler_scales: tuple[float, ...]
    pca_mean: tuple[float, ...]
    pca_components: tuple[tuple[float, ...], ...]
    cluster_centroids: dict[int, tuple[float, ...]]
    parameters: dict[str, object]
    schema_version: int = 1

    def to_dict(self, *, include_model_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "representation": self.representation,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "feature_names": list(self.feature_names),
            "scaler": {
                "kind": "standard_scaler",
                "means": list(self.scaler_means),
                "scales": list(self.scaler_scales),
            },
            "projection": {
                "kind": "pca_2d",
                "mean": list(self.pca_mean),
                "components": [list(row) for row in self.pca_components],
            },
            "cluster_centroids": {
                str(cluster_id): list(values)
                for cluster_id, values in sorted(self.cluster_centroids.items())
            },
            "parameters": dict(sorted(self.parameters.items())),
            "new_track_assignment": "nearest_standardized_centroid_proxy",
        }
        if include_model_id:
            record["model_id"] = self.model_id
        return record


@dataclass(frozen=True, slots=True)
class TasteMapReport:
    """Aggregate representation coverage, comparison, and selection evidence."""

    profile_name: str
    profile_version: str
    representation: RepresentationKind
    tracks_clustered: int
    input_dimensions: int
    pca_explained_variance_ratio: tuple[float, float]
    selected_algorithm: str
    selected_parameters: dict[str, object]
    selected_cluster_count: int
    selected_noise_rate: float
    cluster_sizes: dict[int, int]
    model_id: str
    experiments: tuple[ClusteringExperiment, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "representation": self.representation,
            "tracks_clustered": self.tracks_clustered,
            "input_dimensions": self.input_dimensions,
            "pca_explained_variance_ratio": list(self.pca_explained_variance_ratio),
            "selection_rule": (
                "highest silhouette among eligible K-Means/HDBSCAN settings; "
                "HDBSCAN noise must remain below configured maximum"
            ),
            "selected_algorithm": self.selected_algorithm,
            "selected_parameters": dict(sorted(self.selected_parameters.items())),
            "selected_cluster_count": self.selected_cluster_count,
            "selected_noise_rate": self.selected_noise_rate,
            "cluster_sizes": {
                str(cluster_id): count
                for cluster_id, count in sorted(self.cluster_sizes.items())
            },
            "model_id": self.model_id,
            "experiments": [experiment.to_dict() for experiment in self.experiments],
        }


@dataclass(frozen=True, slots=True)
class TasteMapResult:
    assignments: tuple[TasteMapAssignment, ...]
    model: TasteMapModel
    report: TasteMapReport


@dataclass(frozen=True, slots=True)
class _CandidateResult:
    experiment: ClusteringExperiment
    labels: np.ndarray
    confidences: np.ndarray


def _representation_flags(kind: RepresentationKind) -> tuple[bool, bool]:
    return kind in {"descriptors", "combined"}, kind in {"embedding", "combined"}


def _mean_pairwise_ari(label_runs: Sequence[np.ndarray]) -> float | None:
    if len(label_runs) < 2:
        return None
    scores = [
        adjusted_rand_score(label_runs[left], label_runs[right])
        for left in range(len(label_runs))
        for right in range(left + 1, len(label_runs))
    ]
    return round(float(np.mean(scores)), 8)


def _cluster_metrics(
    matrix: np.ndarray, labels: np.ndarray
) -> tuple[int, float, float | None, float | None]:
    non_noise = labels >= 0
    retained = matrix[non_noise]
    retained_labels = labels[non_noise]
    cluster_count = len(set(int(value) for value in retained_labels))
    noise_rate = round(1.0 - float(np.mean(non_noise)), 8)
    if cluster_count < 2 or len(retained) <= cluster_count:
        return cluster_count, noise_rate, None, None
    return (
        cluster_count,
        noise_rate,
        round(float(silhouette_score(retained, retained_labels)), 8),
        round(float(davies_bouldin_score(retained, retained_labels)), 8),
    )


def _model_identifier(model: TasteMapModel) -> str:
    canonical = json.dumps(
        model.to_dict(include_model_id=False),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_taste_map(
    observations: Iterable[FeatureObservation],
    *,
    profile: AudioFeatureProfile,
    profile_name: str,
    config: TasteMapConfig | None = None,
) -> TasteMapResult:
    """Compare cluster families, select by stated evidence, and project track vectors."""

    active = config or TasteMapConfig()
    catalog = ProfiledFeatureCatalog(observations, profile)
    include_descriptors, include_embedding = _representation_flags(active.representation)
    if include_descriptors and not profile.descriptor_inputs:
        raise TasteMapError("Selected profile has no descriptor representation")
    if include_embedding and profile.embedding_input is None:
        raise TasteMapError("Selected profile has no embedding representation")
    if include_descriptors and include_embedding:
        track_ids = sorted(catalog.fair_cohort_tracks)
    elif include_descriptors:
        track_ids = sorted(catalog.descriptor_tracks)
    else:
        track_ids = sorted(catalog.embedding_tracks)
    if len(track_ids) < 4:
        raise TasteMapError("Taste-map clustering needs at least four covered tracks")
    vectors = [
        catalog.vector(
            track_id,
            include_descriptors=include_descriptors,
            include_embedding=include_embedding,
        )
        for track_id in track_ids
    ]
    if any(vector is None for vector in vectors):
        raise TasteMapError("Taste-map cohort contains an incomplete representation")
    matrix = np.asarray(cast(list[tuple[float, ...]], vectors), dtype=np.float64)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    pca = PCA(n_components=2, random_state=active.random_seeds[0])
    projection = pca.fit_transform(standardized)

    candidates: list[_CandidateResult] = []
    maximum_k = min(active.maximum_k, len(track_ids) - 1)
    for cluster_count in range(active.minimum_k, maximum_k + 1):
        label_runs: list[np.ndarray] = []
        inertia_runs: list[float] = []
        silhouette_runs: list[float] = []
        davies_runs: list[float] = []
        for seed in active.random_seeds:
            estimator = KMeans(
                n_clusters=cluster_count,
                n_init=10,
                random_state=seed,
            )
            labels = estimator.fit_predict(standardized)
            label_runs.append(labels)
            inertia_runs.append(float(estimator.inertia_))
            _, _, silhouette, davies = _cluster_metrics(standardized, labels)
            assert silhouette is not None and davies is not None
            silhouette_runs.append(silhouette)
            davies_runs.append(davies)
        representative_index = int(np.argmin(inertia_runs))
        experiment = ClusteringExperiment(
            algorithm="kmeans",
            parameters={
                "n_clusters": cluster_count,
                "seeds": list(active.random_seeds),
                "n_init": 10,
            },
            cluster_count=cluster_count,
            noise_rate=0.0,
            silhouette=round(float(np.mean(silhouette_runs)), 8),
            davies_bouldin=round(float(np.mean(davies_runs)), 8),
            stability_ari=_mean_pairwise_ari(label_runs),
            eligible_for_selection=True,
        )
        candidates.append(
            _CandidateResult(
                experiment=experiment,
                labels=label_runs[representative_index],
                confidences=np.ones(len(track_ids), dtype=np.float64),
            )
        )

    for minimum_cluster_size in active.hdbscan_min_cluster_sizes:
        if minimum_cluster_size >= len(track_ids):
            continue
        estimator = HDBSCAN(min_cluster_size=minimum_cluster_size, copy=False)
        labels = estimator.fit_predict(standardized)
        cluster_count, noise_rate, silhouette, davies = _cluster_metrics(
            standardized, labels
        )
        eligible_for_selection = (
            silhouette is not None
            and noise_rate <= active.maximum_hdbscan_noise_rate
            and cluster_count >= 2
        )
        experiment = ClusteringExperiment(
            algorithm="hdbscan",
            parameters={"min_cluster_size": minimum_cluster_size},
            cluster_count=cluster_count,
            noise_rate=noise_rate,
            silhouette=silhouette,
            davies_bouldin=davies,
            stability_ari=None,
            eligible_for_selection=eligible_for_selection,
        )
        candidates.append(
            _CandidateResult(
                experiment=experiment,
                labels=labels,
                confidences=np.asarray(estimator.probabilities_, dtype=np.float64),
            )
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.experiment.eligible_for_selection
        and candidate.experiment.silhouette is not None
    ]
    if not eligible:
        raise TasteMapError("No clustering experiment produced two evaluable clusters")
    selected = max(
        eligible,
        key=lambda candidate: (
            cast(float, candidate.experiment.silhouette),
            candidate.experiment.stability_ari or -1.0,
            -candidate.experiment.noise_rate,
            candidate.experiment.algorithm,
        ),
    )
    labels = selected.labels
    centroids: dict[int, tuple[float, ...]] = {}
    for cluster_id in sorted(set(int(value) for value in labels if value >= 0)):
        centroid = np.mean(standardized[labels == cluster_id], axis=0)
        centroids[cluster_id] = tuple(float(value) for value in centroid)
    feature_names: tuple[str, ...] = ()
    if include_descriptors:
        feature_names += catalog.descriptor_feature_names
    if include_embedding:
        feature_names += catalog.embedding_feature_names
    provisional_model = TasteMapModel(
        model_id="pending",
        algorithm=selected.experiment.algorithm,
        representation=active.representation,
        profile_name=profile_name,
        profile_version=profile.profile_version,
        feature_names=feature_names,
        scaler_means=tuple(float(value) for value in scaler.mean_),
        scaler_scales=tuple(float(value) for value in scaler.scale_),
        pca_mean=tuple(float(value) for value in pca.mean_),
        pca_components=tuple(
            tuple(float(value) for value in component) for component in pca.components_
        ),
        cluster_centroids=centroids,
        parameters=selected.experiment.parameters,
    )
    model = replace(provisional_model, model_id=_model_identifier(provisional_model))
    assignments: list[TasteMapAssignment] = []
    for index, track_id in enumerate(track_ids):
        cluster_id = int(labels[index])
        is_noise = cluster_id < 0
        distance = None
        if not is_noise:
            centroid_array = np.asarray(centroids[cluster_id], dtype=np.float64)
            distance = round(float(np.linalg.norm(standardized[index] - centroid_array)), 8)
        assignments.append(
            TasteMapAssignment(
                track_id=track_id,
                cluster_id=cluster_id,
                is_noise=is_noise,
                cluster_confidence=round(float(selected.confidences[index]), 8),
                distance_to_cluster_centroid=distance,
                projection_x=round(float(projection[index, 0]), 8),
                projection_y=round(float(projection[index, 1]), 8),
                model_id=model.model_id,
                profile_name=profile_name,
                profile_version=profile.profile_version,
            )
        )
    cluster_sizes = Counter(int(value) for value in labels)
    explained = tuple(round(float(value), 8) for value in pca.explained_variance_ratio_)
    report = TasteMapReport(
        profile_name=profile_name,
        profile_version=profile.profile_version,
        representation=active.representation,
        tracks_clustered=len(track_ids),
        input_dimensions=matrix.shape[1],
        pca_explained_variance_ratio=(explained[0], explained[1]),
        selected_algorithm=selected.experiment.algorithm,
        selected_parameters=selected.experiment.parameters,
        selected_cluster_count=selected.experiment.cluster_count,
        selected_noise_rate=selected.experiment.noise_rate,
        cluster_sizes=dict(cluster_sizes),
        model_id=model.model_id,
        experiments=tuple(candidate.experiment for candidate in candidates),
    )
    assert_privacy_safe(report.to_dict())
    return TasteMapResult(assignments=tuple(assignments), model=model, report=report)


def write_taste_map(
    result: TasteMapResult, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Write private assignments plus reproducible model and aggregate report."""

    destination = Path(output_dir)
    assignment_lines = []
    for assignment in result.assignments:
        record = assignment.to_dict()
        assert_privacy_safe(record)
        assignment_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    assignments_path = atomic_write_text(
        destination / "taste_map_assignments.jsonl",
        "\n".join(assignment_lines) + "\n",
    )
    model_record = result.model.to_dict()
    assert_privacy_safe(model_record)
    model_path = atomic_write_text(
        destination / "taste_map_model.json",
        json.dumps(model_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    report_record = result.report.to_dict()
    assert_privacy_safe(report_record)
    report_path = atomic_write_text(
        destination / "taste_map_report.json",
        json.dumps(report_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return assignments_path, model_path, report_path


def _assignment_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TasteMapError(f"Taste-map assignment field {key} must be text")
    return value


def _assignment_number(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TasteMapError(f"Taste-map assignment field {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise TasteMapError(f"Taste-map assignment field {key} must be finite")
    return number


def read_taste_map_assignments(path: str | Path) -> tuple[TasteMapAssignment, ...]:
    """Read private taste-map assignments for recommendation cluster context."""

    assignments: list[TasteMapAssignment] = []
    seen_tracks: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TasteMapError(
                    f"Taste-map assignment line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise TasteMapError(
                    f"Taste-map assignment line {line_number} must be an object"
                )
            record = cast(Mapping[str, object], payload)
            if record.get("schema_version") != 1:
                raise TasteMapError("Taste-map assignment schema_version must be 1")
            raw_cluster = record.get("cluster_id")
            raw_noise = record.get("is_noise")
            raw_distance = record.get("distance_to_cluster_centroid")
            if isinstance(raw_cluster, bool) or not isinstance(raw_cluster, int):
                raise TasteMapError("Taste-map assignment cluster_id must be an integer")
            if not isinstance(raw_noise, bool):
                raise TasteMapError("Taste-map assignment is_noise must be boolean")
            if raw_distance is not None and (
                isinstance(raw_distance, bool) or not isinstance(raw_distance, (int, float))
            ):
                raise TasteMapError("Taste-map assignment distance must be null or numeric")
            assignment = TasteMapAssignment(
                track_id=_assignment_text(record, "track_id"),
                cluster_id=raw_cluster,
                is_noise=raw_noise,
                cluster_confidence=_assignment_number(record, "cluster_confidence"),
                distance_to_cluster_centroid=(
                    float(raw_distance) if raw_distance is not None else None
                ),
                projection_x=_assignment_number(record, "projection_x"),
                projection_y=_assignment_number(record, "projection_y"),
                model_id=_assignment_text(record, "model_id"),
                profile_name=_assignment_text(record, "profile_name"),
                profile_version=_assignment_text(record, "profile_version"),
            )
            if assignment.track_id in seen_tracks:
                raise TasteMapError("Taste-map assignments contain a duplicate track_id")
            seen_tracks.add(assignment.track_id)
            assignments.append(assignment)
    return tuple(assignments)
