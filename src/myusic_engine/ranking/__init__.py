"""Behavior aggregation, retrieval, and recommendation ranking."""

from myusic_engine.ranking.behavior import (
    AffinityConfig,
    BehaviorAggregationError,
    TrackAffinity,
    aggregate_track_behavior,
    load_affinity_config,
    load_duration_map,
    score_affinity,
    write_track_affinities,
)
from myusic_engine.ranking.similarity import (
    CategoricalFilter,
    EmbeddingSpec,
    FilterEvidence,
    NumericRangeFilter,
    SimilarityError,
    SimilarityFilter,
    SimilarityIndex,
    SimilarityMatch,
    weighted_query_embedding,
)

__all__ = [
    "AffinityConfig",
    "BehaviorAggregationError",
    "CategoricalFilter",
    "EmbeddingSpec",
    "FilterEvidence",
    "NumericRangeFilter",
    "SimilarityError",
    "SimilarityFilter",
    "SimilarityIndex",
    "SimilarityMatch",
    "TrackAffinity",
    "aggregate_track_behavior",
    "load_affinity_config",
    "load_duration_map",
    "score_affinity",
    "write_track_affinities",
    "weighted_query_embedding",
]
