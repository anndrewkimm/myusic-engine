"""Validated track-level aggregation for window-level audio embeddings."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from myusic_engine.features.records import FeatureObservation


class EmbeddingExtractionError(ValueError):
    """Raised when model output cannot form a trustworthy track embedding."""


@dataclass(frozen=True, slots=True)
class PooledEmbedding:
    """A normalized track vector and its aggregation evidence."""

    vector: tuple[float, ...]
    window_count: int
    coverage_seconds: float
    confidence: float


def _finite_vector(vector: Sequence[float], dimensions: int) -> tuple[float, ...]:
    if len(vector) != dimensions:
        raise EmbeddingExtractionError("Window embedding has unexpected dimensions")
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingExtractionError("Window embeddings must contain only finite values")
    return values


def mean_pool_l2_normalize(
    window_vectors: Iterable[Sequence[float]],
    *,
    dimensions: int,
    coverage_seconds: float,
    reliable_coverage_seconds: float = 30.0,
) -> PooledEmbedding:
    """Arithmetic-mean pool model windows and L2-normalize the resulting track vector."""

    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise EmbeddingExtractionError("dimensions must be a positive integer")
    coverage = float(coverage_seconds)
    if not math.isfinite(coverage) or coverage <= 0:
        raise EmbeddingExtractionError("coverage_seconds must be positive and finite")
    if not math.isfinite(reliable_coverage_seconds) or reliable_coverage_seconds <= 0:
        raise EmbeddingExtractionError("reliable_coverage_seconds must be positive and finite")
    vectors = [_finite_vector(vector, dimensions) for vector in window_vectors]
    if not vectors:
        raise EmbeddingExtractionError("At least one window embedding is required")
    pooled = tuple(
        math.fsum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)
    )
    norm = math.sqrt(math.fsum(value * value for value in pooled))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise EmbeddingExtractionError("Mean window embedding has zero norm")
    normalized = tuple(value / norm for value in pooled)
    confidence = 0.95 * min(1.0, coverage / reliable_coverage_seconds)
    return PooledEmbedding(
        vector=normalized,
        window_count=len(vectors),
        coverage_seconds=coverage,
        confidence=confidence,
    )


def embedding_observation(
    track_id: str,
    pooled: PooledEmbedding,
    *,
    feature_name: str,
    feature_source: str,
    source_version: str,
) -> FeatureObservation:
    """Convert a pooled model result to the shared provenance contract."""

    return FeatureObservation(
        track_id=track_id,
        feature_name=feature_name,
        value=pooled.vector,
        feature_source=feature_source,
        source_version=source_version,
        coverage_seconds=pooled.coverage_seconds,
        feature_confidence=pooled.confidence,
    )
