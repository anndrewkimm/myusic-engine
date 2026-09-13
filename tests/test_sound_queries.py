import json
from dataclasses import replace

import pytest

from myusic_engine.features import FeatureObservation, FeatureSelector
from myusic_engine.modeling import AudioFeatureProfile, AudioInputSpec
from myusic_engine.ranking import CandidateTrack, RecommendationError, rank_candidates
from myusic_engine.ranking.constraints import RankingConstraints, load_ranking_constraints
from myusic_engine.ranking.query import SoundQuery
from myusic_engine.ranking.similarity import NumericRangeFilter, SimilarityError

SELECTOR = FeatureSelector("synthetic_embedding_v1", "synthetic", "v1")
TEMPO = FeatureSelector("tempo_bpm_estimate_v1", "synthetic", "v1")
PROFILE = AudioFeatureProfile("synthetic_v1", (), AudioInputSpec(SELECTOR, 2))
CANDIDATES = (
    CandidateTrack("a", artist_name="Artist A"),
    CandidateTrack("b", artist_name="Artist B"),
)


def observation(track, vector, selector=SELECTOR):
    return FeatureObservation(
        track,
        selector.feature_name,
        vector,
        selector.feature_source,
        selector.source_version,
        30,
        1,
    )


def rank(query=None, **kwargs):
    return rank_candidates(
        CANDIDATES,
        (
            observation("a", (1.0, 0.0)),
            observation("b", (0.0, 1.0)),
            observation("a", 120.0, TEMPO),
        ),
        profile=PROFILE,
        profile_name="synthetic",
        sound_query=query,
        **kwargs,
    )


def test_sound_query_ranks_and_combines_with_weighted_seeds():
    query = SoundQuery("piano", (0.0, 1.0), SELECTOR)
    result = rank(query)
    assert result.recommendations[0].candidate.track_id == "b"
    assert result.recommendations[0].cosine_similarity == 1
    assert result.report.sound_query == query
    mixed = rank(query, seed_weights={"a": 2.0})
    assert mixed.recommendations[0].candidate.track_id == "b"
    assert mixed.recommendations[0].cosine_similarity == pytest.approx(1 / 5**0.5)
    assert mixed.report.run_id != result.report.run_id


def test_query_space_is_checked_even_if_dimensions_match():
    query = SoundQuery(
        "piano", (1.0, 0.0), FeatureSelector("other_embedding_v1", "synthetic", "v1")
    )
    with pytest.raises(RecommendationError, match="space"):
        rank(query)


def test_filter_missing_values_are_excluded_with_a_reason():
    query = SoundQuery("piano", (0.0, 1.0), SELECTOR)
    constraints = RankingConstraints((NumericRangeFilter(TEMPO, 100, 140),))
    result = rank(query, constraints=constraints)
    assert result.recommendations[0].candidate.track_id == "a"
    assert result.recommendations[1].exclusion_reason == "missing_filter_feature"
    assert rank(query).report.run_id != result.report.run_id
    excluded = rank(query, constraints=RankingConstraints(excluded_artists=frozenset({"ARTIST B"})))
    assert excluded.recommendations[1].exclusion_reason == "excluded_artist"


def test_run_hash_captures_top_k_and_query_weight():
    query = SoundQuery("piano", (0.0, 1.0), SELECTOR)
    identifiers = {
        rank(query).report.run_id,
        rank(query, top_k=1).report.run_id,
        rank(replace(query, weight=2)).report.run_id,
    }
    assert len(identifiers) == 3


@pytest.mark.parametrize("weight", [0, -1, float("inf"), float("nan")])
def test_nonfinite_or_nonpositive_query_weight_rejected(weight):
    with pytest.raises(SimilarityError):
        SoundQuery("piano", (1.0, 0.0), SELECTOR, weight)


def test_constraints_roundtrip_and_strict_types(tmp_path):
    path = tmp_path / "filters.json"
    constraints = RankingConstraints((NumericRangeFilter(TEMPO, 100, 140),))
    document = constraints.to_dict()
    path.write_text(json.dumps(document))
    assert load_ranking_constraints(path) == constraints
    document["filters"][0]["required"] = "false"
    path.write_text(json.dumps(document))
    with pytest.raises(SimilarityError, match="boolean"):
        load_ranking_constraints(path)
