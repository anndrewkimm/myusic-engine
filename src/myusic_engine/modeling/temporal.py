"""Chronological implicit-feedback examples with strict point-in-time behavior features."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TypeAlias, cast

from myusic_engine.ingest import NormalizedListeningEvent
from myusic_engine.io import atomic_write_text
from myusic_engine.modeling.config import TemporalConfig
from myusic_engine.privacy import assert_privacy_safe

DatasetSplit: TypeAlias = Literal["train", "validation", "test"]
_SPLITS: tuple[DatasetSplit, ...] = ("train", "validation", "test")

BEHAVIOR_FEATURE_NAMES: tuple[str, ...] = (
    "prior_seen",
    "prior_log_play_count",
    "prior_log_total_ms",
    "prior_outcome_coverage",
    "prior_positive_rate",
    "prior_early_skip_rate",
    "prior_intentional_coverage",
    "prior_intentional_rate",
    "prior_recency_score",
    "prior_track_age_log_days",
    "prior_artist_log_play_count",
    "prior_artist_outcome_coverage",
    "prior_artist_positive_rate",
)


class TemporalDatasetError(ValueError):
    """Raised when chronological examples cannot be built without leakage."""


@dataclass(slots=True)
class _BehaviorState:
    play_count: int = 0
    total_ms: int = 0
    positive_count: int = 0
    negative_count: int = 0
    intentional_count: int = 0
    intentional_known: int = 0
    first_played_at: datetime | None = None
    last_played_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TemporalTasteSample:
    """One target-period label and only the behavior known before that period."""

    sample_id: str
    track_id: str
    artist_key: str | None
    period_index: int
    period_start: str
    period_end: str
    split: DatasetSplit
    label: int
    positive_events: int
    negative_events: int
    outcome_positive_rate: float
    sample_weight: float
    behavior_features: tuple[float, ...]
    dataset_version: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.label not in {0, 1}:
            raise TemporalDatasetError("Temporal sample schema or label is invalid")
        if len(self.behavior_features) != len(BEHAVIOR_FEATURE_NAMES):
            raise TemporalDatasetError("Temporal sample behavior vector has wrong dimensions")
        if self.positive_events + self.negative_events < 1:
            raise TemporalDatasetError("Temporal sample needs at least one known outcome")
        if not 0 <= self.outcome_positive_rate <= 1 or self.sample_weight <= 0:
            raise TemporalDatasetError("Temporal sample rate or weight is invalid")
        if any(not math.isfinite(value) for value in self.behavior_features):
            raise TemporalDatasetError("Temporal sample behavior values must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "track_id": self.track_id,
            "artist_key": self.artist_key,
            "period_index": self.period_index,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "split": self.split,
            "label": self.label,
            "positive_events": self.positive_events,
            "negative_events": self.negative_events,
            "outcome_positive_rate": self.outcome_positive_rate,
            "sample_weight": self.sample_weight,
            "behavior_features": {
                name: value
                for name, value in zip(BEHAVIOR_FEATURE_NAMES, self.behavior_features, strict=True)
            },
            "dataset_version": self.dataset_version,
        }


@dataclass(frozen=True, slots=True)
class BehaviorSnapshot:
    """Latest point-in-time behavior vector used to rank known or cold-start tracks."""

    track_id: str
    artist_key: str | None
    as_of: str
    behavior_features: tuple[float, ...]
    dataset_version: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "artist_key": self.artist_key,
            "as_of": self.as_of,
            "behavior_features": {
                name: value
                for name, value in zip(BEHAVIOR_FEATURE_NAMES, self.behavior_features, strict=True)
            },
            "dataset_version": self.dataset_version,
        }


@dataclass(frozen=True, slots=True)
class TemporalDatasetReport:
    """Aggregate label, split, and leakage-boundary diagnostics."""

    dataset_version: str
    period_days: int
    first_event_at: str
    last_event_at: str
    events_seen: int
    track_events_seen: int
    known_positive_events: int
    known_negative_events: int
    unknown_outcome_events: int
    periods_with_samples: int
    split_period_counts: dict[str, int]
    sample_counts: dict[str, int]
    positive_sample_counts: dict[str, int]
    abstained_track_periods: int
    unique_sample_tracks: int
    snapshot_tracks: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "period_days": self.period_days,
            "first_event_at": self.first_event_at,
            "last_event_at": self.last_event_at,
            "events_seen": self.events_seen,
            "track_events_seen": self.track_events_seen,
            "known_positive_events": self.known_positive_events,
            "known_negative_events": self.known_negative_events,
            "unknown_outcome_events": self.unknown_outcome_events,
            "periods_with_samples": self.periods_with_samples,
            "split_period_counts": dict(sorted(self.split_period_counts.items())),
            "sample_counts": dict(sorted(self.sample_counts.items())),
            "positive_sample_counts": dict(sorted(self.positive_sample_counts.items())),
            "abstained_track_periods": self.abstained_track_periods,
            "unique_sample_tracks": self.unique_sample_tracks,
            "snapshot_tracks": self.snapshot_tracks,
        }


@dataclass(frozen=True, slots=True)
class TemporalDatasetResult:
    samples: tuple[TemporalTasteSample, ...]
    snapshots: tuple[BehaviorSnapshot, ...]
    report: TemporalDatasetReport


def artist_key(artist_name: str | None) -> str | None:
    """Create a stable private grouping key without retaining an artist name."""

    if artist_name is None or not artist_name.strip():
        return None
    canonical = " ".join(artist_name.casefold().split())
    return f"artist:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


def _track_id(event: NormalizedListeningEvent) -> str:
    if event.track_uri:
        return event.track_uri
    canonical = "\u001f".join(
        (value or "").strip().casefold()
        for value in (event.track_name, event.artist_name, event.album_name)
    )
    return f"unresolved:track:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


def _parse_timestamp(value: str) -> datetime:
    candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise TemporalDatasetError("Normalized history contains an invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TemporalDatasetError("Normalized timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _outcome(event: NormalizedListeningEvent, config: TemporalConfig) -> bool | None:
    reason_end = event.reason_end.casefold() if event.reason_end else None
    if reason_end in config.complete_end_reasons:
        return True
    explicit_skip = event.skipped is True or reason_end in config.skip_end_reasons
    if explicit_skip and event.ms_played < config.early_skip_maximum_ms:
        return False
    if event.skipped is False and event.ms_played >= config.positive_minimum_ms:
        return True
    return None


def _intentional(event: NormalizedListeningEvent, config: TemporalConfig) -> bool | None:
    reason_start = event.reason_start.casefold() if event.reason_start else None
    if reason_start in config.intentional_start_reasons:
        return True
    if reason_start in config.passive_start_reasons:
        return False
    return None


def _behavior_vector(
    track_state: _BehaviorState | None,
    artist_state: _BehaviorState | None,
    as_of: datetime,
    config: TemporalConfig,
) -> tuple[float, ...]:
    track = track_state or _BehaviorState()
    artist = artist_state or _BehaviorState()
    track_known = track.positive_count + track.negative_count
    artist_known = artist.positive_count + artist.negative_count
    days_since = (
        max(0.0, (as_of - track.last_played_at).total_seconds() / 86_400.0)
        if track.last_played_at is not None
        else math.inf
    )
    age_days = (
        max(0.0, (as_of - track.first_played_at).total_seconds() / 86_400.0)
        if track.first_played_at is not None
        else 0.0
    )
    values = (
        1.0 if track.play_count else 0.0,
        math.log1p(track.play_count),
        math.log1p(track.total_ms),
        track_known / track.play_count if track.play_count else 0.0,
        track.positive_count / track_known if track_known else 0.0,
        track.negative_count / track_known if track_known else 0.0,
        track.intentional_known / track.play_count if track.play_count else 0.0,
        track.intentional_count / track.intentional_known if track.intentional_known else 0.0,
        (
            math.exp(-math.log(2.0) * days_since / config.recency_half_life_days)
            if math.isfinite(days_since)
            else 0.0
        ),
        math.log1p(age_days),
        math.log1p(artist.play_count),
        artist_known / artist.play_count if artist.play_count else 0.0,
        artist.positive_count / artist_known if artist_known else 0.0,
    )
    return tuple(round(value, 8) for value in values)


def _update_state(
    state: _BehaviorState,
    event: NormalizedListeningEvent,
    played_at: datetime,
    config: TemporalConfig,
) -> None:
    state.play_count += 1
    state.total_ms += event.ms_played
    if state.first_played_at is None:
        state.first_played_at = played_at
    state.last_played_at = played_at
    outcome = _outcome(event, config)
    if outcome is True:
        state.positive_count += 1
    elif outcome is False:
        state.negative_count += 1
    intentional = _intentional(event, config)
    if intentional is not None:
        state.intentional_known += 1
        state.intentional_count += int(intentional)


def _split_periods(period_indices: list[int], config: TemporalConfig) -> dict[int, DatasetSplit]:
    ordered = sorted(set(period_indices))
    required = config.minimum_train_periods + 2
    if len(ordered) < required:
        raise TemporalDatasetError(
            f"Need at least {required} periods with labels for chronological train/validation/test"
        )
    validation_count = max(1, round(len(ordered) * config.validation_fraction))
    test_count = max(1, round(len(ordered) * config.test_fraction))
    while len(ordered) - validation_count - test_count < config.minimum_train_periods:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise TemporalDatasetError("Not enough whole periods remain for training")
    train_end = len(ordered) - validation_count - test_count
    validation_end = len(ordered) - test_count
    assignments: dict[int, DatasetSplit] = {}
    for index, period_index in enumerate(ordered):
        if index < train_end:
            assignments[period_index] = "train"
        elif index < validation_end:
            assignments[period_index] = "validation"
        else:
            assignments[period_index] = "test"
    return assignments


@dataclass(frozen=True, slots=True)
class _PendingSample:
    track_id: str
    artist_key: str | None
    period_index: int
    period_start: datetime
    period_end: datetime
    label: int
    positive_events: int
    negative_events: int
    outcome_positive_rate: float
    sample_weight: float
    behavior_features: tuple[float, ...]


def build_temporal_dataset(
    events: Iterable[NormalizedListeningEvent],
    *,
    config: TemporalConfig | None = None,
) -> TemporalDatasetResult:
    """Build point-in-time examples; update history state only after each target period."""

    active = config or TemporalConfig()
    track_states: dict[str, _BehaviorState] = {}
    artist_states: dict[str, _BehaviorState] = {}
    track_artists: dict[str, str | None] = {}
    pending_samples: list[_PendingSample] = []
    current_events: list[tuple[NormalizedListeningEvent, datetime]] = []
    anchor: datetime | None = None
    current_period_index: int | None = None
    previous_time: datetime | None = None
    first_event_at: datetime | None = None
    last_event_at: datetime | None = None
    events_seen = 0
    track_events_seen = 0
    known_positive_events = 0
    known_negative_events = 0
    unknown_outcome_events = 0
    abstained_track_periods = 0

    def flush_period(period_index: int) -> None:
        nonlocal abstained_track_periods
        assert anchor is not None
        period_start = anchor + timedelta(days=active.period_days * period_index)
        period_end = period_start + timedelta(days=active.period_days)
        grouped: dict[str, list[tuple[NormalizedListeningEvent, datetime]]] = defaultdict(list)
        for event, played_at in current_events:
            grouped[_track_id(event)].append((event, played_at))
        for track_id, grouped_events in sorted(grouped.items()):
            current_artist = next(
                (
                    key
                    for event, _ in reversed(grouped_events)
                    if (key := artist_key(event.artist_name)) is not None
                ),
                track_artists.get(track_id),
            )
            outcomes = [_outcome(event, active) for event, _ in grouped_events]
            positives = sum(outcome is True for outcome in outcomes)
            negatives = sum(outcome is False for outcome in outcomes)
            known = positives + negatives
            if known >= active.minimum_labeled_events:
                positive_rate = positives / known
                if positive_rate >= active.positive_fraction_threshold:
                    label = 1
                elif positive_rate <= active.negative_fraction_threshold:
                    label = 0
                else:
                    label = -1
                if label >= 0:
                    pending_samples.append(
                        _PendingSample(
                            track_id=track_id,
                            artist_key=current_artist,
                            period_index=period_index,
                            period_start=period_start,
                            period_end=period_end,
                            label=label,
                            positive_events=positives,
                            negative_events=negatives,
                            outcome_positive_rate=round(positive_rate, 8),
                            sample_weight=round(
                                min(active.maximum_sample_weight, math.sqrt(known)), 8
                            ),
                            behavior_features=_behavior_vector(
                                track_states.get(track_id),
                                artist_states.get(current_artist) if current_artist else None,
                                period_start,
                                active,
                            ),
                        )
                    )
                else:
                    abstained_track_periods += 1
            for event, played_at in grouped_events:
                track_state = track_states.setdefault(track_id, _BehaviorState())
                _update_state(track_state, event, played_at, active)
                event_artist = artist_key(event.artist_name) or current_artist
                if event_artist is not None:
                    _update_state(
                        artist_states.setdefault(event_artist, _BehaviorState()),
                        event,
                        played_at,
                        active,
                    )
                    track_artists[track_id] = event_artist
                elif track_id not in track_artists:
                    track_artists[track_id] = None

    for event in events:
        events_seen += 1
        played_at = _parse_timestamp(event.played_at)
        if previous_time is not None and played_at < previous_time:
            raise TemporalDatasetError("Cleaned history must be sorted chronologically")
        previous_time = played_at
        first_event_at = first_event_at or played_at
        last_event_at = played_at
        if anchor is None:
            anchor = played_at.replace(hour=0, minute=0, second=0, microsecond=0)
        if event.media_type != "track":
            continue
        track_events_seen += 1
        outcome = _outcome(event, active)
        if outcome is True:
            known_positive_events += 1
        elif outcome is False:
            known_negative_events += 1
        else:
            unknown_outcome_events += 1
        period_index = (played_at - anchor).days // active.period_days
        if current_period_index is None:
            current_period_index = period_index
        elif period_index != current_period_index:
            flush_period(current_period_index)
            current_events.clear()
            current_period_index = period_index
        current_events.append((event, played_at))

    if first_event_at is None or last_event_at is None or anchor is None:
        raise TemporalDatasetError("Normalized history contains no events")
    if current_period_index is not None:
        flush_period(current_period_index)
    if not pending_samples:
        raise TemporalDatasetError("History produced no decisive track-period labels")

    split_by_period = _split_periods([sample.period_index for sample in pending_samples], active)
    samples: list[TemporalTasteSample] = []
    for pending in pending_samples:
        split = split_by_period[pending.period_index]
        digest_input = (
            f"{pending.track_id}\u001f{pending.period_index}\u001f{active.dataset_version}"
        )
        samples.append(
            TemporalTasteSample(
                sample_id=hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
                track_id=pending.track_id,
                artist_key=pending.artist_key,
                period_index=pending.period_index,
                period_start=_timestamp(pending.period_start),
                period_end=_timestamp(pending.period_end),
                split=split,
                label=pending.label,
                positive_events=pending.positive_events,
                negative_events=pending.negative_events,
                outcome_positive_rate=pending.outcome_positive_rate,
                sample_weight=pending.sample_weight,
                behavior_features=pending.behavior_features,
                dataset_version=active.dataset_version,
            )
        )
    samples.sort(key=lambda item: (item.period_index, item.track_id))

    snapshot_as_of = last_event_at + timedelta(milliseconds=1)
    snapshots_list: list[BehaviorSnapshot] = []
    for track_id, state in sorted(track_states.items()):
        track_artist = track_artists.get(track_id)
        snapshots_list.append(
            BehaviorSnapshot(
                track_id=track_id,
                artist_key=track_artist,
                as_of=_timestamp(snapshot_as_of),
                behavior_features=_behavior_vector(
                    state,
                    artist_states.get(track_artist) if track_artist is not None else None,
                    snapshot_as_of,
                    active,
                ),
                dataset_version=active.dataset_version,
            )
        )
    snapshots = tuple(snapshots_list)
    sample_counts = Counter(sample.split for sample in samples)
    positive_counts = Counter(sample.split for sample in samples if sample.label == 1)
    period_counts = Counter(split_by_period.values())
    report = TemporalDatasetReport(
        dataset_version=active.dataset_version,
        period_days=active.period_days,
        first_event_at=_timestamp(first_event_at),
        last_event_at=_timestamp(last_event_at),
        events_seen=events_seen,
        track_events_seen=track_events_seen,
        known_positive_events=known_positive_events,
        known_negative_events=known_negative_events,
        unknown_outcome_events=unknown_outcome_events,
        periods_with_samples=len(split_by_period),
        split_period_counts={split: period_counts[split] for split in _SPLITS},
        sample_counts={split: sample_counts[split] for split in _SPLITS},
        positive_sample_counts={split: positive_counts[split] for split in _SPLITS},
        abstained_track_periods=abstained_track_periods,
        unique_sample_tracks=len({sample.track_id for sample in samples}),
        snapshot_tracks=len(snapshots),
    )
    assert_privacy_safe(report.to_dict())
    return TemporalDatasetResult(samples=tuple(samples), snapshots=snapshots, report=report)


def _feature_vector(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise TemporalDatasetError(f"{label} must be an object")
    section = cast(Mapping[str, object], value)
    if set(section) != set(BEHAVIOR_FEATURE_NAMES):
        raise TemporalDatasetError(f"{label} does not match the behavior feature contract")
    result: list[float] = []
    for name in BEHAVIOR_FEATURE_NAMES:
        raw = section[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TemporalDatasetError(f"{label}.{name} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise TemporalDatasetError(f"{label}.{name} must be finite")
        result.append(number)
    return tuple(result)


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TemporalDatasetError(f"Temporal record field {key} must be text")
    return value


def _required_int(record: Mapping[str, object], key: str, minimum: int = 0) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TemporalDatasetError(f"Temporal record field {key} must be an integer")
    return value


def _required_number(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalDatasetError(f"Temporal record field {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise TemporalDatasetError(f"Temporal record field {key} must be finite")
    return number


def read_temporal_samples(path: str | Path) -> tuple[TemporalTasteSample, ...]:
    """Read and validate private temporal examples from JSON Lines."""

    samples: list[TemporalTasteSample] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TemporalDatasetError(
                    f"Temporal sample line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise TemporalDatasetError(f"Temporal sample line {line_number} must be an object")
            record = cast(Mapping[str, object], payload)
            split_value = record.get("split")
            if split_value not in {"train", "validation", "test"}:
                raise TemporalDatasetError("Temporal sample has invalid split")
            label = _required_int(record, "label")
            if label not in {0, 1}:
                raise TemporalDatasetError("Temporal sample label must be zero or one")
            raw_artist = record.get("artist_key")
            if raw_artist is not None and not isinstance(raw_artist, str):
                raise TemporalDatasetError("Temporal sample artist_key must be null or text")
            sample = TemporalTasteSample(
                schema_version=_required_int(record, "schema_version"),
                sample_id=_required_text(record, "sample_id"),
                track_id=_required_text(record, "track_id"),
                artist_key=raw_artist,
                period_index=_required_int(record, "period_index"),
                period_start=_required_text(record, "period_start"),
                period_end=_required_text(record, "period_end"),
                split=cast(DatasetSplit, split_value),
                label=label,
                positive_events=_required_int(record, "positive_events"),
                negative_events=_required_int(record, "negative_events"),
                outcome_positive_rate=_required_number(record, "outcome_positive_rate"),
                sample_weight=_required_number(record, "sample_weight"),
                behavior_features=_feature_vector(
                    record.get("behavior_features"), "behavior_features"
                ),
                dataset_version=_required_text(record, "dataset_version"),
            )
            if sample.sample_id in seen_ids:
                raise TemporalDatasetError("Temporal sample input contains duplicate sample_id")
            seen_ids.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise TemporalDatasetError("Temporal sample input contains no records")
    return tuple(samples)


def read_behavior_snapshots(path: str | Path) -> tuple[BehaviorSnapshot, ...]:
    """Read latest history-state vectors used by recommendation scoring."""

    snapshots: list[BehaviorSnapshot] = []
    seen_tracks: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TemporalDatasetError(
                    f"Behavior snapshot line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise TemporalDatasetError(
                    f"Behavior snapshot line {line_number} must be an object"
                )
            record = cast(Mapping[str, object], payload)
            raw_artist = record.get("artist_key")
            if raw_artist is not None and not isinstance(raw_artist, str):
                raise TemporalDatasetError("Behavior snapshot artist_key must be null or text")
            snapshot = BehaviorSnapshot(
                schema_version=_required_int(record, "schema_version"),
                track_id=_required_text(record, "track_id"),
                artist_key=raw_artist,
                as_of=_required_text(record, "as_of"),
                behavior_features=_feature_vector(
                    record.get("behavior_features"), "behavior_features"
                ),
                dataset_version=_required_text(record, "dataset_version"),
            )
            if snapshot.track_id in seen_tracks:
                raise TemporalDatasetError("Behavior snapshot input contains duplicate track_id")
            seen_tracks.add(snapshot.track_id)
            snapshots.append(snapshot)
    return tuple(snapshots)


def write_temporal_dataset(
    result: TemporalDatasetResult, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Atomically write private samples/snapshots and an aggregate report."""

    destination = Path(output_dir)
    sample_lines = []
    for sample in result.samples:
        record = sample.to_dict()
        assert_privacy_safe(record)
        sample_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    snapshot_lines = []
    for snapshot in result.snapshots:
        record = snapshot.to_dict()
        assert_privacy_safe(record)
        snapshot_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    samples_path = atomic_write_text(
        destination / "temporal_taste_samples.jsonl",
        "\n".join(sample_lines) + "\n",
    )
    snapshots_path = atomic_write_text(
        destination / "behavior_snapshots.jsonl",
        "\n".join(snapshot_lines) + ("\n" if snapshot_lines else ""),
    )
    report_record = result.report.to_dict()
    assert_privacy_safe(report_record)
    report_path = atomic_write_text(
        destination / "temporal_dataset_report.json",
        json.dumps(report_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return samples_path, snapshots_path, report_path
