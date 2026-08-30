"""Explainable final ranking over candidate identities and independently sourced audio."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import yaml

from myusic_engine.clustering import TasteMapAssignment
from myusic_engine.features import FeatureObservation
from myusic_engine.io import atomic_write_text
from myusic_engine.modeling import (
    BEHAVIOR_FEATURE_NAMES,
    AudioFeatureProfile,
    BehaviorSnapshot,
    LinearTasteModel,
    ProfiledFeatureCatalog,
    artist_key,
    model_input_vector,
)
from myusic_engine.privacy import assert_privacy_safe
from myusic_engine.ranking.candidates import CandidateTrack
from myusic_engine.ranking.similarity import weighted_query_embedding

RankingTier: TypeAlias = Literal["audio_ranked", "preference_ranked", "metadata_only"]
_ARTIST_BEHAVIOR_INDICES = tuple(
    BEHAVIOR_FEATURE_NAMES.index(name)
    for name in (
        "prior_artist_log_play_count",
        "prior_artist_outcome_coverage",
        "prior_artist_positive_rate",
    )
)


class RecommendationError(ValueError):
    """Raised when candidates cannot be ranked under the selected provenance."""


@dataclass(frozen=True, slots=True)
class RecommendationConfig:
    """Transparent final score weights and diversity controls."""

    audio_similarity_weight: float = 0.50
    predicted_preference_weight: float = 0.35
    novelty_bonus_weight: float = 0.15
    preference_only_novelty_bonus_weight: float = 0.0
    artist_repetition_penalty: float = 0.10
    maximum_per_artist: int = 3
    explanation_feature_count: int = 5
    schema_version: int = 1

    def __post_init__(self) -> None:
        weights = (
            self.audio_similarity_weight,
            self.predicted_preference_weight,
            self.novelty_bonus_weight,
            self.preference_only_novelty_bonus_weight,
            self.artist_repetition_penalty,
        )
        if self.schema_version != 1 or any(
            not math.isfinite(weight) or weight < 0 for weight in weights
        ):
            raise RecommendationError("Recommendation config weights are invalid")
        if sum(weights[:3]) <= 0:
            raise RecommendationError("At least one positive ranking weight is required")
        if self.maximum_per_artist < 1 or self.explanation_feature_count < 1:
            raise RecommendationError("Recommendation diversity controls must be positive")


def load_recommendation_config(path: str | Path) -> RecommendationConfig:
    """Load the ranking section while leaving behavior configuration independent."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RecommendationError("Recommendation config is not valid YAML") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise RecommendationError("Recommendation config schema_version must be 1")
    raw = payload.get("ranking")
    if not isinstance(raw, Mapping):
        raise RecommendationError("Recommendation config needs a ranking object")
    section = cast(Mapping[str, object], raw)
    allowed = {
        "audio_similarity_weight",
        "predicted_preference_weight",
        "novelty_bonus_weight",
        "preference_only_novelty_bonus_weight",
        "artist_repetition_penalty",
        "maximum_per_artist",
        "explanation_feature_count",
    }
    unknown = set(section) - allowed
    if unknown:
        raise RecommendationError(f"Unknown ranking fields: {', '.join(sorted(unknown))}")
    defaults = RecommendationConfig()

    def number(key: str, default: float) -> float:
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RecommendationError(f"Ranking field {key} must be numeric")
        return float(value)

    def integer(key: str, default: int) -> int:
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RecommendationError(f"Ranking field {key} must be an integer")
        return value

    return RecommendationConfig(
        audio_similarity_weight=number("audio_similarity_weight", defaults.audio_similarity_weight),
        predicted_preference_weight=number(
            "predicted_preference_weight", defaults.predicted_preference_weight
        ),
        novelty_bonus_weight=number("novelty_bonus_weight", defaults.novelty_bonus_weight),
        preference_only_novelty_bonus_weight=number(
            "preference_only_novelty_bonus_weight",
            defaults.preference_only_novelty_bonus_weight,
        ),
        artist_repetition_penalty=number(
            "artist_repetition_penalty", defaults.artist_repetition_penalty
        ),
        maximum_per_artist=integer("maximum_per_artist", defaults.maximum_per_artist),
        explanation_feature_count=integer(
            "explanation_feature_count", defaults.explanation_feature_count
        ),
    )


@dataclass(frozen=True, slots=True)
class RankedRecommendation:
    """One candidate with separate acoustic, preference, novelty, and diversity evidence."""

    rank: int | None
    candidate: CandidateTrack
    tier: RankingTier
    final_score: float | None
    base_score: float | None
    cosine_similarity: float | None
    predicted_preference: float | None
    novelty_bonus: float
    artist_repetition_penalty: float
    model_explanations: tuple[tuple[str, float], ...]
    cluster_id: int | None
    cluster_is_noise: bool | None
    cluster_confidence: float | None
    cluster_model_id: str | None
    exclusion_reason: str | None
    run_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "rank": self.rank,
            **self.candidate.to_dict(),
            "ranking_tier": self.tier,
            "final_score": self.final_score,
            "score_components": {
                "base_score": self.base_score,
                "cosine_similarity": self.cosine_similarity,
                "predicted_preference": self.predicted_preference,
                "novelty_bonus": self.novelty_bonus,
                "artist_repetition_penalty": self.artist_repetition_penalty,
            },
            "model_explanations": [
                {"feature_name": name, "log_odds_contribution": round(value, 8)}
                for name, value in self.model_explanations
            ],
            "cluster_context": (
                {
                    "cluster_id": self.cluster_id,
                    "is_noise": self.cluster_is_noise,
                    "confidence": self.cluster_confidence,
                    "taste_map_model_id": self.cluster_model_id,
                }
                if self.cluster_model_id is not None
                else None
            ),
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class RecommendationReport:
    """Reproducibility and coverage record for one recommendation run."""

    run_id: str
    candidates_seen: int
    ranked_count: int
    output_count: int
    tier_counts: dict[str, int]
    seed_count: int
    model_id: str | None
    model_name: str | None
    taste_map_model_id: str | None
    seed_cluster_counts: dict[int, int]
    profile_name: str | None
    profile_version: str | None
    candidate_digest: str
    feature_digest: str
    behavior_snapshot_digest: str
    taste_map_assignment_digest: str
    behavior_dataset_version: str | None
    behavior_snapshot_as_of: str | None
    config: RecommendationConfig
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidates_seen": self.candidates_seen,
            "ranked_count": self.ranked_count,
            "output_count": self.output_count,
            "tier_counts": dict(sorted(self.tier_counts.items())),
            "seed_count": self.seed_count,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "taste_map_model_id": self.taste_map_model_id,
            "seed_cluster_counts": {
                str(cluster_id): count
                for cluster_id, count in sorted(self.seed_cluster_counts.items())
            },
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "input_digests": {
                "candidates": self.candidate_digest,
                "features": self.feature_digest,
                "behavior_snapshots": self.behavior_snapshot_digest,
                "taste_map_assignments": self.taste_map_assignment_digest,
            },
            "behavior_dataset_version": self.behavior_dataset_version,
            "behavior_snapshot_as_of": self.behavior_snapshot_as_of,
            "ranking_config": {
                "audio_similarity_weight": self.config.audio_similarity_weight,
                "predicted_preference_weight": self.config.predicted_preference_weight,
                "novelty_bonus_weight": self.config.novelty_bonus_weight,
                "preference_only_novelty_bonus_weight": (
                    self.config.preference_only_novelty_bonus_weight
                ),
                "artist_repetition_penalty": self.config.artist_repetition_penalty,
                "maximum_per_artist": self.config.maximum_per_artist,
                "explanation_feature_count": self.config.explanation_feature_count,
            },
        }


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    recommendations: tuple[RankedRecommendation, ...]
    report: RecommendationReport


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    candidate: CandidateTrack
    tier: RankingTier
    base_score: float | None
    similarity: float | None
    preference: float | None
    novelty: float
    explanations: tuple[tuple[str, float], ...]
    cluster_assignment: TasteMapAssignment | None
    exclusion_reason: str | None


def _normalized(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm <= 0:
        raise RecommendationError("Audio embedding must have non-zero norm")
    return tuple(value / norm for value in vector)


def _behavior_for_candidate(
    candidate: CandidateTrack,
    by_track: Mapping[str, BehaviorSnapshot],
    by_artist: Mapping[str, BehaviorSnapshot],
) -> tuple[float, ...]:
    snapshot = by_track.get(candidate.track_id)
    if snapshot is not None:
        return snapshot.behavior_features
    values = [0.0] * len(BEHAVIOR_FEATURE_NAMES)
    key = artist_key(candidate.artist_name)
    artist_snapshot = by_artist.get(key) if key is not None else None
    if artist_snapshot is not None:
        for index in _ARTIST_BEHAVIOR_INDICES:
            values[index] = artist_snapshot.behavior_features[index]
    return tuple(values)


def _novelty(behavior: Sequence[float]) -> float:
    log_play_count = behavior[BEHAVIOR_FEATURE_NAMES.index("prior_log_play_count")]
    play_count = max(0.0, math.expm1(log_play_count))
    return 1.0 / (1.0 + math.sqrt(play_count))


def _run_identifier(
    seeds: Mapping[str, float],
    profile_name: str | None,
    profile_version: str | None,
    model_id: str | None,
    taste_map_model_id: str | None,
    config: RecommendationConfig,
    *,
    candidate_digest: str,
    feature_digest: str,
    behavior_snapshot_digest: str,
    taste_map_assignment_digest: str,
) -> str:
    record = {
        "candidate_digest": candidate_digest,
        "feature_digest": feature_digest,
        "behavior_snapshot_digest": behavior_snapshot_digest,
        "taste_map_assignment_digest": taste_map_assignment_digest,
        "seeds": dict(sorted(seeds.items())),
        "profile_name": profile_name,
        "profile_version": profile_version,
        "model_id": model_id,
        "taste_map_model_id": taste_map_model_id,
        "config": RecommendationReport(
            run_id="pending",
            candidates_seen=0,
            ranked_count=0,
            output_count=0,
            tier_counts={},
            seed_count=0,
            model_id=None,
            model_name=None,
            taste_map_model_id=taste_map_model_id,
            seed_cluster_counts={},
            profile_name=profile_name,
            profile_version=profile_version,
            candidate_digest=candidate_digest,
            feature_digest=feature_digest,
            behavior_snapshot_digest=behavior_snapshot_digest,
            taste_map_assignment_digest=taste_map_assignment_digest,
            behavior_dataset_version=None,
            behavior_snapshot_as_of=None,
            config=config,
        ).to_dict()["ranking_config"],
    }
    canonical = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_digest(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def rank_candidates(
    candidates: Iterable[CandidateTrack],
    observations: Iterable[FeatureObservation],
    *,
    profile: AudioFeatureProfile | None = None,
    profile_name: str | None = None,
    seed_weights: Mapping[str, float] | None = None,
    model: LinearTasteModel | None = None,
    behavior_snapshots: Iterable[BehaviorSnapshot] = (),
    cluster_assignments: Iterable[TasteMapAssignment] = (),
    config: RecommendationConfig | None = None,
    top_k: int = 50,
) -> RecommendationResult:
    """Rank candidates while preserving missing-audio and metadata-only distinctions."""

    active = config or RecommendationConfig()
    candidate_rows = tuple(candidates)
    if not candidate_rows or top_k < 1:
        raise RecommendationError("Candidates and top_k must be non-empty and positive")
    candidate_ids = [candidate.track_id for candidate in candidate_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RecommendationError("Candidates contain duplicate track IDs")
    observation_rows = tuple(observations)
    if (profile is None) != (profile_name is None):
        raise RecommendationError("Audio profile and profile_name must be supplied together")
    if profile is None and observation_rows:
        raise RecommendationError("Feature observations require an audio profile")
    catalog = ProfiledFeatureCatalog(observation_rows, profile) if profile is not None else None
    seeds = dict(seed_weights or {})
    if seeds and (profile is None or profile.embedding_input is None):
        raise RecommendationError("Seed similarity requires an embedding profile")
    if (
        model is not None
        and (model.includes_descriptors or model.includes_embedding)
        and (
            profile is None
            or model.profile_name != profile_name
            or model.profile_version != profile.profile_version
        )
    ):
        raise RecommendationError("Taste model and selected audio profile do not match")
    query_vector: tuple[float, ...] | None = None
    if seeds:
        seed_vectors = []
        for track_id, weight in seeds.items():
            assert catalog is not None
            representation = catalog.get(track_id)
            if representation is None or representation.embedding is None:
                raise RecommendationError(f"Seed track lacks the selected embedding: {track_id}")
            seed_vectors.append((representation.embedding, weight))
        query_vector = weighted_query_embedding(seed_vectors)
    snapshots = tuple(behavior_snapshots)
    snapshot_by_track = {snapshot.track_id: snapshot for snapshot in snapshots}
    if len(snapshot_by_track) != len(snapshots):
        raise RecommendationError("Behavior snapshots contain duplicate track IDs")
    snapshot_versions = {snapshot.dataset_version for snapshot in snapshots}
    snapshot_times = {snapshot.as_of for snapshot in snapshots}
    if len(snapshot_versions) > 1 or len(snapshot_times) > 1:
        raise RecommendationError("Behavior snapshots must share one dataset version and as_of")
    behavior_dataset_version = next(iter(snapshot_versions), None)
    behavior_snapshot_as_of = next(iter(snapshot_times), None)
    if (
        model is not None
        and model.includes_behavior
        and behavior_dataset_version is not None
        and model.dataset_version != behavior_dataset_version
    ):
        raise RecommendationError("Taste model and behavior snapshots use different datasets")
    snapshot_by_artist: dict[str, BehaviorSnapshot] = {}
    for snapshot in snapshots:
        if snapshot.artist_key is None:
            continue
        previous = snapshot_by_artist.get(snapshot.artist_key)
        if previous is not None and any(
            previous.behavior_features[index] != snapshot.behavior_features[index]
            for index in _ARTIST_BEHAVIOR_INDICES
        ):
            raise RecommendationError("Behavior snapshots disagree on shared artist features")
        snapshot_by_artist[snapshot.artist_key] = snapshot
    assignment_rows = tuple(cluster_assignments)
    if profile is None and assignment_rows:
        raise RecommendationError("Taste-map assignments require an audio profile")
    assignment_by_track = {assignment.track_id: assignment for assignment in assignment_rows}
    if len(assignment_by_track) != len(assignment_rows):
        raise RecommendationError("Taste-map assignments contain duplicate track IDs")
    taste_map_model_ids = {assignment.model_id for assignment in assignment_rows}
    if len(taste_map_model_ids) > 1:
        raise RecommendationError("Taste-map assignments mix model IDs")
    if any(
        assignment.profile_name != profile_name
        or profile is None
        or assignment.profile_version != profile.profile_version
        for assignment in assignment_rows
    ):
        raise RecommendationError("Taste-map assignments and audio profile do not match")
    taste_map_model_id = next(iter(taste_map_model_ids), None)
    candidate_digest = _content_digest(
        candidate.to_dict() for candidate in sorted(candidate_rows, key=lambda item: item.track_id)
    )
    feature_digest = _content_digest(
        observation.to_dict()
        for observation in sorted(
            observation_rows,
            key=lambda item: (
                item.track_id,
                item.feature_name,
                item.feature_source,
                item.source_version,
            ),
        )
    )
    behavior_snapshot_digest = _content_digest(
        snapshot.to_dict() for snapshot in sorted(snapshots, key=lambda item: item.track_id)
    )
    taste_map_assignment_digest = _content_digest(
        assignment.to_dict()
        for assignment in sorted(assignment_rows, key=lambda item: item.track_id)
    )
    run_id = _run_identifier(
        seeds,
        profile_name,
        profile.profile_version if profile is not None else None,
        model.model_id if model is not None else None,
        taste_map_model_id,
        active,
        candidate_digest=candidate_digest,
        feature_digest=feature_digest,
        behavior_snapshot_digest=behavior_snapshot_digest,
        taste_map_assignment_digest=taste_map_assignment_digest,
    )
    scored: list[_ScoredCandidate] = []
    for candidate in candidate_rows:
        behavior = _behavior_for_candidate(candidate, snapshot_by_track, snapshot_by_artist)
        novelty = _novelty(behavior)
        representation = catalog.get(candidate.track_id) if catalog is not None else None
        similarity = None
        if (
            query_vector is not None
            and representation is not None
            and representation.embedding is not None
        ):
            candidate_vector = _normalized(representation.embedding)
            similarity = max(
                -1.0,
                min(
                    1.0,
                    math.fsum(
                        left * right
                        for left, right in zip(query_vector, candidate_vector, strict=True)
                    ),
                ),
            )
        preference = None
        explanations: tuple[tuple[str, float], ...] = ()
        if model is not None:
            model_values = model_input_vector(model, behavior, catalog, candidate.track_id)
            if model_values is not None:
                preference = model.predict_probability(model_values)
                explanations = tuple(
                    sorted(
                        model.contributions(model_values),
                        key=lambda item: abs(item[1]),
                        reverse=True,
                    )[: active.explanation_feature_count]
                )
        if similarity is not None:
            tier: RankingTier = "audio_ranked"
        elif preference is not None:
            tier = "preference_ranked"
        else:
            tier = "metadata_only"
        numerator = 0.0
        denominator = 0.0
        if similarity is not None:
            numerator += active.audio_similarity_weight * ((similarity + 1.0) / 2.0)
            denominator += active.audio_similarity_weight
        if preference is not None:
            numerator += active.predicted_preference_weight * preference
            denominator += active.predicted_preference_weight
        novelty_weight = (
            active.novelty_bonus_weight
            if similarity is not None
            else active.preference_only_novelty_bonus_weight
        )
        if tier != "metadata_only":
            numerator += novelty_weight * novelty
            denominator += novelty_weight
        base_score = numerator / denominator if denominator else None
        exclusion_reason = None
        if candidate.track_id in seeds:
            base_score = None
            exclusion_reason = "seed_track"
        scored.append(
            _ScoredCandidate(
                candidate=candidate,
                tier=tier,
                base_score=base_score,
                similarity=similarity,
                preference=preference,
                novelty=novelty,
                explanations=explanations,
                cluster_assignment=assignment_by_track.get(candidate.track_id),
                exclusion_reason=exclusion_reason,
            )
        )

    remaining = [item for item in scored if item.base_score is not None]
    selected: list[RankedRecommendation] = []
    artist_counts: Counter[str] = Counter()
    while remaining and len(selected) < top_k:
        eligible: list[tuple[float, _ScoredCandidate, str | None]] = []
        for item in remaining:
            key = artist_key(item.candidate.artist_name)
            if key is not None and artist_counts[key] >= active.maximum_per_artist:
                continue
            repetition = active.artist_repetition_penalty * (
                artist_counts[key] if key is not None else 0
            )
            assert item.base_score is not None
            eligible.append((item.base_score - repetition, item, key))
        if not eligible:
            break
        adjusted, chosen, chosen_artist = max(
            eligible, key=lambda item: (item[0], item[1].candidate.track_id)
        )
        repetition = chosen.base_score - adjusted if chosen.base_score is not None else 0.0
        selected.append(
            RankedRecommendation(
                rank=len(selected) + 1,
                candidate=chosen.candidate,
                tier=chosen.tier,
                final_score=round(adjusted, 8),
                base_score=round(chosen.base_score, 8) if chosen.base_score is not None else None,
                cosine_similarity=(
                    round(chosen.similarity, 8) if chosen.similarity is not None else None
                ),
                predicted_preference=(
                    round(chosen.preference, 8) if chosen.preference is not None else None
                ),
                novelty_bonus=round(chosen.novelty, 8),
                artist_repetition_penalty=round(repetition, 8),
                model_explanations=chosen.explanations,
                cluster_id=(
                    chosen.cluster_assignment.cluster_id
                    if chosen.cluster_assignment is not None
                    else None
                ),
                cluster_is_noise=(
                    chosen.cluster_assignment.is_noise
                    if chosen.cluster_assignment is not None
                    else None
                ),
                cluster_confidence=(
                    chosen.cluster_assignment.cluster_confidence
                    if chosen.cluster_assignment is not None
                    else None
                ),
                cluster_model_id=(
                    chosen.cluster_assignment.model_id
                    if chosen.cluster_assignment is not None
                    else None
                ),
                exclusion_reason=None,
                run_id=run_id,
            )
        )
        remaining.remove(chosen)
        if chosen_artist is not None:
            artist_counts[chosen_artist] += 1

    selected_ids = {item.candidate.track_id for item in selected}
    unranked = [
        RankedRecommendation(
            rank=None,
            candidate=item.candidate,
            tier=item.tier,
            final_score=None,
            base_score=item.base_score,
            cosine_similarity=item.similarity,
            predicted_preference=item.preference,
            novelty_bonus=round(item.novelty, 8),
            artist_repetition_penalty=0.0,
            model_explanations=item.explanations,
            cluster_id=(
                item.cluster_assignment.cluster_id if item.cluster_assignment is not None else None
            ),
            cluster_is_noise=(
                item.cluster_assignment.is_noise if item.cluster_assignment is not None else None
            ),
            cluster_confidence=(
                item.cluster_assignment.cluster_confidence
                if item.cluster_assignment is not None
                else None
            ),
            cluster_model_id=(
                item.cluster_assignment.model_id if item.cluster_assignment is not None else None
            ),
            exclusion_reason=(
                item.exclusion_reason
                or (
                    "metadata_only_no_selected_audio_or_model_coverage"
                    if item.base_score is None
                    else "outside_top_k_or_artist_diversity_limit"
                )
            ),
            run_id=run_id,
        )
        for item in scored
        if item.candidate.track_id not in selected_ids
    ]
    recommendations = tuple(selected + unranked)
    tier_counts = Counter(item.tier for item in recommendations)
    seed_cluster_counts = Counter(
        assignment_by_track[track_id].cluster_id
        for track_id in seeds
        if track_id in assignment_by_track
    )
    report = RecommendationReport(
        run_id=run_id,
        candidates_seen=len(candidate_rows),
        ranked_count=sum(item.rank is not None for item in recommendations),
        output_count=len(selected),
        tier_counts={str(tier): count for tier, count in tier_counts.items()},
        seed_count=len(seeds),
        model_id=model.model_id if model is not None else None,
        model_name=model.model_name if model is not None else None,
        taste_map_model_id=taste_map_model_id,
        seed_cluster_counts=dict(seed_cluster_counts),
        profile_name=profile_name,
        profile_version=profile.profile_version if profile is not None else None,
        candidate_digest=candidate_digest,
        feature_digest=feature_digest,
        behavior_snapshot_digest=behavior_snapshot_digest,
        taste_map_assignment_digest=taste_map_assignment_digest,
        behavior_dataset_version=behavior_dataset_version,
        behavior_snapshot_as_of=behavior_snapshot_as_of,
        config=active,
    )
    assert_privacy_safe(report.to_dict())
    return RecommendationResult(recommendations=recommendations, report=report)


def write_recommendations(
    result: RecommendationResult, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Write explanations, run log, and official Spotify URI handoff text."""

    destination = Path(output_dir)
    lines = []
    for recommendation in result.recommendations:
        record = recommendation.to_dict()
        assert_privacy_safe(record)
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    recommendations_path = atomic_write_text(
        destination / "recommendations.jsonl", "\n".join(lines) + "\n"
    )
    report_record = result.report.to_dict()
    assert_privacy_safe(report_record)
    report_path = atomic_write_text(
        destination / "recommendation_run.json",
        json.dumps(report_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    spotify_uris = [
        item.candidate.spotify_uri
        for item in result.recommendations
        if item.rank is not None and item.candidate.spotify_uri is not None
    ]
    spotify_path = atomic_write_text(
        destination / "spotify_playlist_uris.txt",
        "\n".join(spotify_uris) + ("\n" if spotify_uris else ""),
    )
    return recommendations_path, report_path, spotify_path
