"""Provenance-aware cosine retrieval with weighted multi-seed queries."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from myusic_engine.features import (
    FeatureCatalog,
    FeatureObservation,
    FeatureRecordError,
    FeatureSelector,
)


class SimilarityError(ValueError):
    """Raised when an embedding index or query is invalid."""


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimilarityError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SimilarityError(f"{field_name} must be finite")
    return normalized


def _normalized_vector(vector: Sequence[float]) -> tuple[float, ...]:
    if not vector:
        raise SimilarityError("Embedding vectors must not be empty")
    values = tuple(_finite_number(value, "embedding component") for value in vector)
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0:
        raise SimilarityError("Embedding vectors must have non-zero norm")
    return tuple(value / norm for value in values)


def weighted_query_embedding(
    weighted_vectors: Iterable[tuple[Sequence[float], float]],
) -> tuple[float, ...]:
    """Normalize each seed, take its positive weighted sum, then normalize again."""

    prepared: list[tuple[tuple[float, ...], float]] = []
    dimensions: int | None = None
    for vector, raw_weight in weighted_vectors:
        weight = _finite_number(raw_weight, "seed weight")
        if weight <= 0:
            raise SimilarityError("Seed weights must be positive")
        normalized = _normalized_vector(vector)
        if dimensions is None:
            dimensions = len(normalized)
        elif len(normalized) != dimensions:
            raise SimilarityError("All seed embeddings must have the same dimensions")
        prepared.append((normalized, weight))
    if not prepared or dimensions is None:
        raise SimilarityError("At least one seed embedding is required")
    centroid = tuple(
        math.fsum(weight * vector[index] for vector, weight in prepared)
        for index in range(dimensions)
    )
    return _normalized_vector(centroid)


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """The exact model output selected for one similarity index."""

    selector: FeatureSelector
    dimensions: int

    def __post_init__(self) -> None:
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int):
            raise SimilarityError("Embedding dimensions must be an integer")
        if self.dimensions < 1:
            raise SimilarityError("Embedding dimensions must be positive")


@dataclass(frozen=True, slots=True)
class NumericRangeFilter:
    """Inclusive hard range over one exact numeric feature definition."""

    selector: FeatureSelector
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise SimilarityError("Numeric range filter needs a minimum or maximum")
        if self.minimum is not None:
            object.__setattr__(self, "minimum", _finite_number(self.minimum, "filter minimum"))
        if self.maximum is not None:
            object.__setattr__(self, "maximum", _finite_number(self.maximum, "filter maximum"))
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise SimilarityError("Numeric filter minimum must not exceed maximum")
        confidence = _finite_number(self.minimum_confidence, "filter minimum_confidence")
        if not 0 <= confidence <= 1:
            raise SimilarityError("Filter minimum_confidence must be in [0, 1]")
        object.__setattr__(self, "minimum_confidence", confidence)


@dataclass(frozen=True, slots=True)
class CategoricalFilter:
    """Hard allowlist over one exact text feature definition."""

    selector: FeatureSelector
    allowed_values: frozenset[str]
    required: bool = True
    minimum_confidence: float = 0.0
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_values or any(
            not isinstance(value, str) or not value.strip() for value in self.allowed_values
        ):
            raise SimilarityError("Categorical filter values must be non-empty text")
        normalized_values = frozenset(
            value.strip() if self.case_sensitive else value.strip().casefold()
            for value in self.allowed_values
        )
        object.__setattr__(self, "allowed_values", normalized_values)
        confidence = _finite_number(self.minimum_confidence, "filter minimum_confidence")
        if not 0 <= confidence <= 1:
            raise SimilarityError("Filter minimum_confidence must be in [0, 1]")
        object.__setattr__(self, "minimum_confidence", confidence)


SimilarityFilter: TypeAlias = NumericRangeFilter | CategoricalFilter


@dataclass(frozen=True, slots=True)
class FilterEvidence:
    """Feature value that allowed a candidate through one hard filter."""

    selector: str
    value: float | str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "selector": self.selector,
            "value": self.value,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    """One ranked cosine match with retrieval evidence."""

    rank: int
    track_id: str
    cosine_similarity: float
    embedding_confidence: float
    embedding_coverage_seconds: float
    filter_evidence: tuple[FilterEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "track_id": self.track_id,
            "cosine_similarity": round(self.cosine_similarity, 8),
            "embedding_confidence": self.embedding_confidence,
            "embedding_coverage_seconds": self.embedding_coverage_seconds,
            "filter_evidence": [evidence.to_dict() for evidence in self.filter_evidence],
        }


class SimilarityIndex:
    """Small exact index used before a production vector store is warranted."""

    def __init__(
        self,
        embedding_observations: Iterable[FeatureObservation],
        *,
        embedding_spec: EmbeddingSpec,
        feature_observations: Iterable[FeatureObservation] = (),
    ) -> None:
        self.embedding_spec = embedding_spec
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._embedding_records: dict[str, FeatureObservation] = {}
        for observation in embedding_observations:
            if not embedding_spec.selector.matches(observation):
                continue
            if not isinstance(observation.value, tuple):
                raise SimilarityError("Selected embedding observations must contain vectors")
            if len(observation.value) != embedding_spec.dimensions:
                raise SimilarityError("Selected embedding has unexpected dimensions")
            if observation.track_id in self._vectors:
                raise SimilarityError("Duplicate selected embedding for one track")
            self._vectors[observation.track_id] = _normalized_vector(observation.value)
            self._embedding_records[observation.track_id] = observation
        if not self._vectors:
            raise SimilarityError("No observations match the selected embedding provenance")
        try:
            self._features = FeatureCatalog(feature_observations)
        except FeatureRecordError as exc:
            raise SimilarityError(str(exc)) from exc

    @property
    def track_count(self) -> int:
        return len(self._vectors)

    def query(
        self,
        seed_weights: Mapping[str, float],
        *,
        top_k: int = 10,
        filters: Sequence[SimilarityFilter] = (),
        minimum_embedding_confidence: float = 0.0,
        exclude_seeds: bool = True,
    ) -> tuple[SimilarityMatch, ...]:
        """Retrieve the top cosine matches for a weighted set of seed track IDs."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise SimilarityError("top_k must be a positive integer")
        confidence_floor = _finite_number(
            minimum_embedding_confidence, "minimum_embedding_confidence"
        )
        if not 0 <= confidence_floor <= 1:
            raise SimilarityError("minimum_embedding_confidence must be in [0, 1]")
        missing_seeds = sorted(set(seed_weights) - set(self._vectors))
        if missing_seeds:
            raise SimilarityError(f"Seed tracks are missing selected embeddings: {missing_seeds}")
        query_vector = weighted_query_embedding(
            (self._vectors[track_id], weight) for track_id, weight in seed_weights.items()
        )

        candidates: list[tuple[float, str, FeatureObservation, tuple[FilterEvidence, ...]]] = []
        seed_ids = set(seed_weights)
        for track_id, vector in self._vectors.items():
            if exclude_seeds and track_id in seed_ids:
                continue
            embedding_record = self._embedding_records[track_id]
            if embedding_record.feature_confidence < confidence_floor:
                continue
            passed, evidence = self._passes_filters(track_id, filters)
            if not passed:
                continue
            similarity = math.fsum(
                query_component * candidate_component
                for query_component, candidate_component in zip(query_vector, vector, strict=True)
            )
            similarity = max(-1.0, min(1.0, similarity))
            candidates.append((similarity, track_id, embedding_record, evidence))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            SimilarityMatch(
                rank=rank,
                track_id=track_id,
                cosine_similarity=similarity,
                embedding_confidence=embedding_record.feature_confidence,
                embedding_coverage_seconds=embedding_record.coverage_seconds,
                filter_evidence=evidence,
            )
            for rank, (similarity, track_id, embedding_record, evidence) in enumerate(
                candidates[:top_k], start=1
            )
        )

    def _passes_filters(
        self, track_id: str, filters: Sequence[SimilarityFilter]
    ) -> tuple[bool, tuple[FilterEvidence, ...]]:
        evidence: list[FilterEvidence] = []
        for active_filter in filters:
            observation = self._features.get(track_id, active_filter.selector)
            if (
                observation is None
                or observation.feature_confidence < active_filter.minimum_confidence
            ):
                if active_filter.required:
                    return False, ()
                continue
            if isinstance(active_filter, NumericRangeFilter):
                if not isinstance(observation.value, float):
                    raise SimilarityError("Numeric filter selected a non-numeric feature")
                if active_filter.minimum is not None and observation.value < active_filter.minimum:
                    return False, ()
                if active_filter.maximum is not None and observation.value > active_filter.maximum:
                    return False, ()
                matched_value: float | str = observation.value
            else:
                if not isinstance(observation.value, str):
                    raise SimilarityError("Categorical filter selected a non-text feature")
                comparison_value = (
                    observation.value
                    if active_filter.case_sensitive
                    else observation.value.casefold()
                )
                if comparison_value not in active_filter.allowed_values:
                    return False, ()
                matched_value = observation.value
            evidence.append(
                FilterEvidence(
                    selector=active_filter.selector.label,
                    value=matched_value,
                    confidence=observation.feature_confidence,
                )
            )
        return True, tuple(evidence)
