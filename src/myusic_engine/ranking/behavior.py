"""Track-level behavioral aggregation and transparent affinity scoring."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import yaml

from myusic_engine.ingest.models import NormalizedListeningEvent
from myusic_engine.io import atomic_write_text
from myusic_engine.privacy import assert_privacy_safe


class BehaviorAggregationError(ValueError):
    """Raised when behavior inputs or scoring configuration are invalid."""


_AFFINITY_CONFIG_FIELDS = frozenset(
    {
        "scoring_version",
        "play_count_log_weight",
        "completion_rate_weight",
        "intentional_start_rate_weight",
        "recency_score_weight",
        "early_skip_rate_weight",
        "early_skip_threshold_ms",
        "recency_half_life_days",
        "completion_threshold",
        "session_gap_minutes",
        "complete_end_reasons",
        "skip_end_reasons",
        "intentional_start_reasons",
        "passive_start_reasons",
    }
)


@dataclass(frozen=True, slots=True)
class AffinityConfig:
    """Versioned controls for behavior aggregation and the initial heuristic."""

    scoring_version: str = "behavior_affinity_v1"
    play_count_log_weight: float = 1.0
    completion_rate_weight: float = 1.5
    intentional_start_rate_weight: float = 1.0
    recency_score_weight: float = 0.5
    early_skip_rate_weight: float = -2.0
    early_skip_threshold_ms: int = 30_000
    recency_half_life_days: float = 180.0
    completion_threshold: float = 0.90
    session_gap_minutes: int = 30
    complete_end_reasons: frozenset[str] = frozenset({"endplay", "trackdone"})
    skip_end_reasons: frozenset[str] = frozenset({"backbtn", "fwdbtn"})
    intentional_start_reasons: frozenset[str] = frozenset(
        {"backbtn", "clickrow", "fwdbtn", "playbtn", "remote"}
    )
    passive_start_reasons: frozenset[str] = frozenset({"appload", "autoplay", "trackdone"})

    def __post_init__(self) -> None:
        if not self.scoring_version.strip():
            raise BehaviorAggregationError("scoring_version must not be empty")
        if self.early_skip_threshold_ms < 0:
            raise BehaviorAggregationError("early_skip_threshold_ms must be non-negative")
        if self.recency_half_life_days <= 0:
            raise BehaviorAggregationError("recency_half_life_days must be positive")
        if not 0 < self.completion_threshold <= 1:
            raise BehaviorAggregationError("completion_threshold must be in (0, 1]")
        if self.session_gap_minutes <= 0:
            raise BehaviorAggregationError("session_gap_minutes must be positive")
        overlap = self.intentional_start_reasons & self.passive_start_reasons
        if overlap:
            raise BehaviorAggregationError("intentional and passive start reasons must not overlap")


@dataclass(frozen=True, slots=True)
class TrackAffinity:
    """Explainable behavior metrics and affinity score for one track identity."""

    track_id: str
    track_identity_source: str
    track_uri: str | None
    track_name: str | None
    artist_name: str | None
    album_name: str | None
    first_played_at: str
    last_played_at: str
    aggregation_as_of: str
    play_count: int
    total_ms_played: int
    median_ms_played: float
    duration_ms: int | None
    duration_coverage_rate: float
    median_completion_ratio: float | None
    completion_rate: float | None
    completion_signal_coverage: float
    early_skip_rate: float | None
    early_skip_signal_coverage: float
    intentional_start_rate: float | None
    intentional_start_signal_coverage: float
    repeat_within_session_rate: float
    recency_weighted_plays: float
    recency_score: float
    affinity_score: float
    affinity_confidence: float
    affinity_components: dict[str, float]
    scoring_version: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready record."""

        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "track_identity_source": self.track_identity_source,
            "track_uri": self.track_uri,
            "track_name": self.track_name,
            "artist_name": self.artist_name,
            "album_name": self.album_name,
            "first_played_at": self.first_played_at,
            "last_played_at": self.last_played_at,
            "aggregation_as_of": self.aggregation_as_of,
            "play_count": self.play_count,
            "total_ms_played": self.total_ms_played,
            "median_ms_played": self.median_ms_played,
            "duration_ms": self.duration_ms,
            "duration_coverage_rate": self.duration_coverage_rate,
            "median_completion_ratio": self.median_completion_ratio,
            "completion_rate": self.completion_rate,
            "completion_signal_coverage": self.completion_signal_coverage,
            "early_skip_rate": self.early_skip_rate,
            "early_skip_signal_coverage": self.early_skip_signal_coverage,
            "intentional_start_rate": self.intentional_start_rate,
            "intentional_start_signal_coverage": self.intentional_start_signal_coverage,
            "repeat_within_session_rate": self.repeat_within_session_rate,
            "recency_weighted_plays": self.recency_weighted_plays,
            "recency_score": self.recency_score,
            "affinity_score": self.affinity_score,
            "affinity_confidence": self.affinity_confidence,
            "affinity_components": dict(sorted(self.affinity_components.items())),
            "scoring_version": self.scoring_version,
        }


def _number(section: Mapping[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BehaviorAggregationError(f"Affinity config field {key!r} must be numeric")
    return float(value)


def _integer(section: Mapping[str, object], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BehaviorAggregationError(f"Affinity config field {key!r} must be an integer")
    return value


def _reason_set(
    section: Mapping[str, object], key: str, default: frozenset[str]
) -> frozenset[str]:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise BehaviorAggregationError(
            f"Affinity config field {key!r} must be a list of non-empty strings"
        )
    return frozenset(cast(str, item).strip().casefold() for item in value)


def load_affinity_config(path: str | Path) -> AffinityConfig:
    """Load and validate the ``affinity`` section of recommendation YAML."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BehaviorAggregationError("Recommendation config is not valid YAML") from exc
    if not isinstance(payload, Mapping):
        raise BehaviorAggregationError("Recommendation config must be an object")
    if payload.get("schema_version") != 1:
        raise BehaviorAggregationError("Recommendation config schema_version must be 1")
    raw_section = payload.get("affinity")
    if not isinstance(raw_section, Mapping):
        raise BehaviorAggregationError("Recommendation config must contain an affinity object")
    section = cast(Mapping[str, object], raw_section)
    unknown_fields = {str(key) for key in section if key not in _AFFINITY_CONFIG_FIELDS}
    if unknown_fields:
        unknown_list = ", ".join(sorted(unknown_fields))
        raise BehaviorAggregationError(f"Unknown affinity config fields: {unknown_list}")
    defaults = AffinityConfig()
    scoring_version = section.get("scoring_version", defaults.scoring_version)
    if not isinstance(scoring_version, str):
        raise BehaviorAggregationError("Affinity config field 'scoring_version' must be text")
    return replace(
        defaults,
        scoring_version=scoring_version,
        play_count_log_weight=_number(
            section, "play_count_log_weight", defaults.play_count_log_weight
        ),
        completion_rate_weight=_number(
            section, "completion_rate_weight", defaults.completion_rate_weight
        ),
        intentional_start_rate_weight=_number(
            section,
            "intentional_start_rate_weight",
            defaults.intentional_start_rate_weight,
        ),
        recency_score_weight=_number(
            section, "recency_score_weight", defaults.recency_score_weight
        ),
        early_skip_rate_weight=_number(
            section, "early_skip_rate_weight", defaults.early_skip_rate_weight
        ),
        early_skip_threshold_ms=_integer(
            section, "early_skip_threshold_ms", defaults.early_skip_threshold_ms
        ),
        recency_half_life_days=_number(
            section, "recency_half_life_days", defaults.recency_half_life_days
        ),
        completion_threshold=_number(
            section, "completion_threshold", defaults.completion_threshold
        ),
        session_gap_minutes=_integer(
            section, "session_gap_minutes", defaults.session_gap_minutes
        ),
        complete_end_reasons=_reason_set(
            section, "complete_end_reasons", defaults.complete_end_reasons
        ),
        skip_end_reasons=_reason_set(section, "skip_end_reasons", defaults.skip_end_reasons),
        intentional_start_reasons=_reason_set(
            section, "intentional_start_reasons", defaults.intentional_start_reasons
        ),
        passive_start_reasons=_reason_set(
            section, "passive_start_reasons", defaults.passive_start_reasons
        ),
    )


def load_duration_map(path: str | Path) -> dict[str, int]:
    """Load a JSON object mapping a track URI or track ID to duration milliseconds."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BehaviorAggregationError("Duration map is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BehaviorAggregationError("Duration map must be a JSON object")
    durations: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise BehaviorAggregationError("Duration map keys must be non-empty track IDs")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BehaviorAggregationError(
                "Duration map values must be positive integer milliseconds"
            )
        durations[key.strip()] = value
    return durations


def _parsed_timestamp(value: str) -> datetime:
    candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise BehaviorAggregationError(
            "Normalized event has an invalid played_at timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BehaviorAggregationError("Normalized event played_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _track_identity(event: NormalizedListeningEvent) -> tuple[str, str]:
    if event.track_uri:
        return event.track_uri, "spotify_uri"
    metadata = [event.track_name, event.artist_name, event.album_name]
    canonical = "\u001f".join((value or "").strip().casefold() for value in metadata)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"unresolved:track:{digest}", "metadata_hash"


def _known_completion(
    event: NormalizedListeningEvent,
    duration_ms: int | None,
    config: AffinityConfig,
) -> tuple[bool | None, float | None]:
    if duration_ms is not None:
        ratio = min(event.ms_played / duration_ms, 1.0)
        return ratio >= config.completion_threshold, ratio
    reason_end = event.reason_end.casefold() if event.reason_end else None
    if reason_end in config.complete_end_reasons:
        return True, None
    if event.skipped is True or reason_end in config.skip_end_reasons:
        return False, None
    return None, None


def _known_early_skip(
    event: NormalizedListeningEvent, config: AffinityConfig
) -> bool | None:
    reason_end = event.reason_end.casefold() if event.reason_end else None
    if event.skipped is True or reason_end in config.skip_end_reasons:
        return event.ms_played < config.early_skip_threshold_ms
    if event.skipped is False or reason_end in config.complete_end_reasons:
        return False
    return None


def _known_intentional_start(
    event: NormalizedListeningEvent, config: AffinityConfig
) -> bool | None:
    reason_start = event.reason_start.casefold() if event.reason_start else None
    if reason_start in config.intentional_start_reasons:
        return True
    if reason_start in config.passive_start_reasons:
        return False
    return None


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rounded(value: float, digits: int = 6) -> float:
    return round(value, digits)


def score_affinity(
    *,
    play_count: int,
    completion_rate: float | None,
    completion_coverage: float,
    intentional_start_rate: float | None,
    intentional_start_coverage: float,
    recency_score: float,
    early_skip_rate: float | None,
    early_skip_coverage: float,
    config: AffinityConfig,
) -> tuple[float, float, dict[str, float]]:
    """Apply the versioned heuristic and return score, confidence, and components."""

    if play_count < 1:
        raise BehaviorAggregationError("play_count must be positive")
    required_rates = {
        "completion_coverage": completion_coverage,
        "intentional_start_coverage": intentional_start_coverage,
        "recency_score": recency_score,
        "early_skip_coverage": early_skip_coverage,
    }
    optional_rates = {
        "completion_rate": completion_rate,
        "intentional_start_rate": intentional_start_rate,
        "early_skip_rate": early_skip_rate,
    }
    if any(not 0 <= value <= 1 for value in required_rates.values()):
        raise BehaviorAggregationError("Affinity coverage and recency inputs must be in [0, 1]")
    if any(value is not None and not 0 <= value <= 1 for value in optional_rates.values()):
        raise BehaviorAggregationError("Affinity rate inputs must be null or in [0, 1]")

    components = {
        "play_count": config.play_count_log_weight * math.log1p(play_count),
        "completion": config.completion_rate_weight * (completion_rate or 0.0),
        "intentional_start": config.intentional_start_rate_weight
        * (intentional_start_rate or 0.0),
        "recency": config.recency_score_weight * recency_score,
        "early_skip": config.early_skip_rate_weight * (early_skip_rate or 0.0),
    }
    score = sum(components.values())

    coefficient_total = sum(
        abs(weight)
        for weight in (
            config.play_count_log_weight,
            config.completion_rate_weight,
            config.intentional_start_rate_weight,
            config.recency_score_weight,
            config.early_skip_rate_weight,
        )
    )
    available_weight = (
        abs(config.play_count_log_weight)
        + abs(config.completion_rate_weight) * completion_coverage
        + abs(config.intentional_start_rate_weight) * intentional_start_coverage
        + abs(config.recency_score_weight)
        + abs(config.early_skip_rate_weight) * early_skip_coverage
    )
    coverage_confidence = available_weight / coefficient_total if coefficient_total else 0.0
    sample_confidence = 1.0 - math.exp(-play_count / 3.0)
    confidence = coverage_confidence * sample_confidence
    return _rounded(score), _rounded(confidence), {
        name: _rounded(value) for name, value in components.items()
    }


def aggregate_track_behavior(
    events: Iterable[NormalizedListeningEvent],
    *,
    durations_ms: Mapping[str, int] | None = None,
    config: AffinityConfig | None = None,
    as_of: datetime | None = None,
) -> tuple[TrackAffinity, ...]:
    """Aggregate normalized music events into one affinity record per track."""

    active_config = config or AffinityConfig()
    duration_lookup = durations_ms or {}
    timestamped_events = [(_parsed_timestamp(event.played_at), event) for event in events]
    if not timestamped_events:
        return ()
    timestamped_events.sort(key=lambda item: (item[0], item[1].event_id))
    latest_event_time = timestamped_events[-1][0]
    reference_time = as_of or latest_event_time
    if reference_time.tzinfo is None or reference_time.utcoffset() is None:
        raise BehaviorAggregationError("as_of must include a timezone")
    reference_time = reference_time.astimezone(timezone.utc)
    if reference_time < latest_event_time:
        raise BehaviorAggregationError("as_of cannot be earlier than the latest event")

    track_events: dict[str, list[tuple[datetime, NormalizedListeningEvent]]] = defaultdict(list)
    identity_sources: dict[str, str] = {}
    for played_at, event in timestamped_events:
        if event.media_type != "track":
            continue
        track_id, identity_source = _track_identity(event)
        track_events[track_id].append((played_at, event))
        identity_sources[track_id] = identity_source

    repeat_counts: Counter[str] = Counter()
    seen_in_session: set[str] = set()
    previous_time: datetime | None = None
    session_gap = timedelta(minutes=active_config.session_gap_minutes)
    for played_at, event in timestamped_events:
        if previous_time is None or played_at - previous_time > session_gap:
            seen_in_session.clear()
        previous_time = played_at
        if event.media_type != "track":
            continue
        track_id, _ = _track_identity(event)
        if track_id in seen_in_session:
            repeat_counts[track_id] += 1
        seen_in_session.add(track_id)

    affinities: list[TrackAffinity] = []
    for track_id, grouped in track_events.items():
        grouped.sort(key=lambda item: (item[0], item[1].event_id))
        grouped_events = [event for _, event in grouped]
        play_count = len(grouped_events)
        track_uri = next(
            (event.track_uri for event in reversed(grouped_events) if event.track_uri), None
        )
        track_name = next(
            (event.track_name for event in reversed(grouped_events) if event.track_name), None
        )
        artist_name = next(
            (event.artist_name for event in reversed(grouped_events) if event.artist_name), None
        )
        album_name = next(
            (event.album_name for event in reversed(grouped_events) if event.album_name), None
        )
        duration_ms = duration_lookup.get(track_uri or track_id)
        if duration_ms is not None and (
            isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0
        ):
            raise BehaviorAggregationError("Track durations must be positive integer milliseconds")

        completion_signals: list[bool] = []
        completion_ratios: list[float] = []
        early_skip_signals: list[bool] = []
        intentional_start_signals: list[bool] = []
        recency_weights: list[float] = []
        for played_at, event in grouped:
            completed, completion_ratio = _known_completion(event, duration_ms, active_config)
            if completed is not None:
                completion_signals.append(completed)
            if completion_ratio is not None:
                completion_ratios.append(completion_ratio)
            early_skip = _known_early_skip(event, active_config)
            if early_skip is not None:
                early_skip_signals.append(early_skip)
            intentional_start = _known_intentional_start(event, active_config)
            if intentional_start is not None:
                intentional_start_signals.append(intentional_start)
            age_days = (reference_time - played_at).total_seconds() / 86_400
            recency_weights.append(0.5 ** (age_days / active_config.recency_half_life_days))

        completion_rate = _rate(completion_signals)
        early_skip_rate = _rate(early_skip_signals)
        intentional_start_rate = _rate(intentional_start_signals)
        completion_coverage = len(completion_signals) / play_count
        early_skip_coverage = len(early_skip_signals) / play_count
        intentional_start_coverage = len(intentional_start_signals) / play_count
        recency_weighted_plays = sum(recency_weights)
        recency_score = recency_weighted_plays / play_count
        affinity_score, affinity_confidence, affinity_components = score_affinity(
            play_count=play_count,
            completion_rate=completion_rate,
            completion_coverage=completion_coverage,
            intentional_start_rate=intentional_start_rate,
            intentional_start_coverage=intentional_start_coverage,
            recency_score=recency_score,
            early_skip_rate=early_skip_rate,
            early_skip_coverage=early_skip_coverage,
            config=active_config,
        )
        affinity = TrackAffinity(
            track_id=track_id,
            track_identity_source=identity_sources[track_id],
            track_uri=track_uri,
            track_name=track_name,
            artist_name=artist_name,
            album_name=album_name,
            first_played_at=_timestamp_text(grouped[0][0]),
            last_played_at=_timestamp_text(grouped[-1][0]),
            aggregation_as_of=_timestamp_text(reference_time),
            play_count=play_count,
            total_ms_played=sum(event.ms_played for event in grouped_events),
            median_ms_played=_rounded(
                float(statistics.median(event.ms_played for event in grouped_events))
            ),
            duration_ms=duration_ms,
            duration_coverage_rate=1.0 if duration_ms is not None else 0.0,
            median_completion_ratio=(
                _rounded(float(statistics.median(completion_ratios)))
                if completion_ratios
                else None
            ),
            completion_rate=_rounded(completion_rate) if completion_rate is not None else None,
            completion_signal_coverage=_rounded(completion_coverage),
            early_skip_rate=_rounded(early_skip_rate) if early_skip_rate is not None else None,
            early_skip_signal_coverage=_rounded(early_skip_coverage),
            intentional_start_rate=(
                _rounded(intentional_start_rate) if intentional_start_rate is not None else None
            ),
            intentional_start_signal_coverage=_rounded(intentional_start_coverage),
            repeat_within_session_rate=_rounded(repeat_counts[track_id] / play_count),
            recency_weighted_plays=_rounded(recency_weighted_plays),
            recency_score=_rounded(recency_score),
            affinity_score=affinity_score,
            affinity_confidence=affinity_confidence,
            affinity_components=affinity_components,
            scoring_version=active_config.scoring_version,
        )
        assert_privacy_safe(affinity.to_dict())
        affinities.append(affinity)

    return tuple(sorted(affinities, key=lambda item: (-item.affinity_score, item.track_id)))


def write_track_affinities(
    affinities: Iterable[TrackAffinity], output_path: str | Path
) -> Path:
    """Atomically write one source-safe track-affinity JSON record per line."""

    serialized: list[str] = []
    for affinity in affinities:
        record = affinity.to_dict()
        assert_privacy_safe(record)
        serialized.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    content = "\n".join(serialized)
    if content:
        content += "\n"
    return atomic_write_text(output_path, content)
