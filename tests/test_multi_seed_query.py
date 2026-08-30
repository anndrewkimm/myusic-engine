import math

import pytest

from myusic_engine.features import FeatureObservation, FeatureSelector
from myusic_engine.ranking import (
    CategoricalFilter,
    EmbeddingSpec,
    NumericRangeFilter,
    SimilarityError,
    SimilarityIndex,
    weighted_query_embedding,
)

EMBEDDING_SELECTOR = FeatureSelector(
    feature_name="discogs_effnet_embedding_v1",
    feature_source="essentia",
    source_version="discogs-effnet-test-v1",
)
TEMPO_SELECTOR = FeatureSelector(
    feature_name="tempo_bpm_estimate_v1",
    feature_source="clean_room_audio",
    source_version="extractor-test-v1",
)
MODE_SELECTOR = FeatureSelector(
    feature_name="mode_estimate_v1",
    feature_source="clean_room_audio",
    source_version="extractor-test-v1",
)


def _embedding(
    track_id: str, vector: tuple[float, ...], confidence: float = 1.0
) -> FeatureObservation:
    return FeatureObservation(
        track_id=track_id,
        feature_name=EMBEDDING_SELECTOR.feature_name,
        value=vector,
        feature_source=EMBEDDING_SELECTOR.feature_source,
        source_version=EMBEDDING_SELECTOR.source_version,
        coverage_seconds=30.0,
        feature_confidence=confidence,
    )


def _feature(
    track_id: str,
    selector: FeatureSelector,
    value: float | str,
    confidence: float = 1.0,
) -> FeatureObservation:
    return FeatureObservation(
        track_id=track_id,
        feature_name=selector.feature_name,
        value=value,
        feature_source=selector.feature_source,
        source_version=selector.source_version,
        coverage_seconds=30.0,
        feature_confidence=confidence,
    )


def _index(*, features: tuple[FeatureObservation, ...] = ()) -> SimilarityIndex:
    embeddings = [
        _embedding("seed-a", (10.0, 0.0)),
        _embedding("seed-b", (0.0, 1.0)),
        _embedding("balanced", (1.0, 1.0), confidence=0.95),
        _embedding("mostly-a", (1.0, 0.2), confidence=0.90),
        _embedding("opposite", (-1.0, -1.0), confidence=0.99),
    ]
    return SimilarityIndex(
        embeddings,
        embedding_spec=EmbeddingSpec(selector=EMBEDDING_SELECTOR, dimensions=2),
        feature_observations=features,
    )


def test_weighted_query_normalizes_each_seed_before_averaging() -> None:
    query = weighted_query_embedding([((10.0, 0.0), 1.0), ((0.0, 1.0), 1.0)])

    assert query == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))


def test_multi_seed_query_ranks_cosine_matches_and_excludes_seeds() -> None:
    matches = _index().query({"seed-a": 1.0, "seed-b": 1.0}, top_k=3)

    assert [match.track_id for match in matches] == ["balanced", "mostly-a", "opposite"]
    assert matches[0].cosine_similarity == pytest.approx(1.0)
    assert matches[0].embedding_confidence == 0.95
    assert [match.rank for match in matches] == [1, 2, 3]


def test_seed_weights_shift_the_query_direction() -> None:
    equal = _index().query({"seed-a": 1.0, "seed-b": 1.0}, top_k=2)
    weighted = _index().query({"seed-a": 5.0, "seed-b": 1.0}, top_k=2)

    assert equal[0].track_id == "balanced"
    assert weighted[0].track_id == "mostly-a"


def test_numeric_and_categorical_filters_use_exact_provenance() -> None:
    features = (
        _feature("balanced", TEMPO_SELECTOR, 122.0),
        _feature("balanced", MODE_SELECTOR, "major"),
        _feature("mostly-a", TEMPO_SELECTOR, 168.0),
        _feature("mostly-a", MODE_SELECTOR, "minor"),
        FeatureObservation(
            track_id="opposite",
            feature_name=TEMPO_SELECTOR.feature_name,
            value=120.0,
            feature_source="acousticbrainz",
            source_version="legacy-test-v1",
            coverage_seconds=200.0,
            feature_confidence=1.0,
        ),
    )
    filters = (
        NumericRangeFilter(TEMPO_SELECTOR, minimum=115, maximum=130),
        CategoricalFilter(MODE_SELECTOR, frozenset({"MAJOR"})),
    )

    matches = _index(features=features).query({"seed-a": 1.0, "seed-b": 1.0}, filters=filters)

    assert [match.track_id for match in matches] == ["balanced"]
    assert [item.value for item in matches[0].filter_evidence] == [122.0, "major"]
    assert all("clean_room_audio" in item.selector for item in matches[0].filter_evidence)


def test_optional_filter_does_not_drop_missing_features() -> None:
    optional_tempo = NumericRangeFilter(
        TEMPO_SELECTOR,
        minimum=100,
        maximum=140,
        required=False,
    )

    matches = _index().query({"seed-a": 1.0}, filters=(optional_tempo,))

    assert len(matches) == 4
    assert all(match.filter_evidence == () for match in matches)


def test_embedding_and_filter_confidence_floors_are_enforced() -> None:
    features = (
        _feature("balanced", TEMPO_SELECTOR, 122.0, confidence=0.4),
        _feature("mostly-a", TEMPO_SELECTOR, 121.0, confidence=0.9),
    )
    tempo_filter = NumericRangeFilter(
        TEMPO_SELECTOR,
        minimum=100,
        maximum=130,
        minimum_confidence=0.5,
    )

    matches = _index(features=features).query(
        {"seed-a": 1.0},
        filters=(tempo_filter,),
        minimum_embedding_confidence=0.9,
    )

    assert [match.track_id for match in matches] == ["mostly-a"]


@pytest.mark.parametrize(
    ("seed_weights", "message"),
    [
        ({}, "At least one"),
        ({"missing": 1.0}, "missing selected embeddings"),
        ({"seed-a": 0.0}, "positive"),
    ],
)
def test_invalid_seed_queries_are_rejected(seed_weights: dict[str, float], message: str) -> None:
    with pytest.raises(SimilarityError, match=message):
        _index().query(seed_weights)


def test_embedding_dimensions_are_checked_when_index_is_built() -> None:
    with pytest.raises(SimilarityError, match="dimensions"):
        SimilarityIndex(
            [_embedding("track-a", (1.0, 2.0, 3.0))],
            embedding_spec=EmbeddingSpec(selector=EMBEDDING_SELECTOR, dimensions=2),
        )
