"""Convert exact MusicBrainz matches into source-tagged CC0 AcousticBrainz features."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from myusic_engine.features.records import FeatureObservation, write_feature_observations
from myusic_engine.io import atomic_write_text
from myusic_engine.matching.external import ExternalIdentityMatch
from myusic_engine.privacy import assert_privacy_safe
from myusic_engine.providers import AcousticBrainzDocument, AcousticBrainzProvider, ProviderError

LOW_LEVEL_SOURCE = "acousticbrainz_cc0"
LOW_LEVEL_VERSION = "frozen-2022-lowlevel-converter-v1"
HIGH_LEVEL_SOURCE = "acousticbrainz_cc0"
HIGH_LEVEL_VERSION = "frozen-2022-highlevel-converter-v1"

_LOW_NUMBER_FEATURES = (
    ("lowlevel.average_loudness", "average_loudness_v1"),
    ("lowlevel.dynamic_complexity", "dynamic_complexity_v1"),
    ("metadata.audio_properties.replay_gain", "replay_gain_db_v1"),
    ("rhythm.beats_count", "beat_count_v1"),
    ("rhythm.beats_loudness.mean", "beats_loudness_mean_v1"),
    ("rhythm.bpm", "tempo_bpm_estimate_v1"),
    ("rhythm.danceability", "danceability_estimate_v1"),
    ("rhythm.onset_rate", "onset_rate_hz_v1"),
    ("tonal.chords_changes_rate", "chord_change_rate_v1"),
    ("tonal.key_strength", "key_strength_v1"),
    ("tonal.tuning_equal_tempered_deviation", "tuning_deviation_v1"),
    ("tonal.tuning_frequency", "tuning_frequency_hz_v1"),
)
_LOW_TEXT_FEATURES = (
    ("tonal.key_key", "key_estimate_v1"),
    ("tonal.key_scale", "mode_estimate_v1"),
)
_HIGH_LEVEL_TASKS = (
    ("danceability", "danceable", "danceable_score_v1"),
    ("mood_acoustic", "acoustic", "acoustic_score_v1"),
    ("mood_aggressive", "aggressive", "aggressive_score_v1"),
    ("mood_happy", "happy", "happy_score_v1"),
    ("mood_relaxed", "relaxed", "relaxed_score_v1"),
    ("voice_instrumental", "instrumental", "instrumental_score_v1"),
)


@dataclass(frozen=True, slots=True)
class AcousticFeatureCoverageReport:
    """Private aggregate coverage without track names or provider response bodies."""

    exact_tracks_considered: int
    unique_recording_mbids: int
    low_level_tracks_covered: int
    high_level_tracks_covered: int
    low_level_track_rate: float
    high_level_track_rate: float
    low_level_play_rate: float
    high_level_play_rate: float
    low_level_ms_played_rate: float
    high_level_ms_played_rate: float
    observations_written: int
    feature_counts: dict[str, int]
    schema_version: int = 1
    conversion_version: str = "acousticbrainz-feature-conversion-v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "conversion_version": self.conversion_version,
            "exact_tracks_considered": self.exact_tracks_considered,
            "unique_recording_mbids": self.unique_recording_mbids,
            "low_level_tracks_covered": self.low_level_tracks_covered,
            "high_level_tracks_covered": self.high_level_tracks_covered,
            "low_level_track_rate": self.low_level_track_rate,
            "high_level_track_rate": self.high_level_track_rate,
            "low_level_play_rate": self.low_level_play_rate,
            "high_level_play_rate": self.high_level_play_rate,
            "low_level_ms_played_rate": self.low_level_ms_played_rate,
            "high_level_ms_played_rate": self.high_level_ms_played_rate,
            "observations_written": self.observations_written,
            "feature_counts": dict(sorted(self.feature_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class AcousticFeatureResult:
    observations: tuple[FeatureObservation, ...]
    report: AcousticFeatureCoverageReport


def _nested(record: Mapping[str, object], path: str) -> object | None:
    current: object = record
    for component in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(component)
        if current is None:
            return None
    return current


def _finite_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError(f"AcousticBrainz field {label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProviderError(f"AcousticBrainz field {label} must be finite")
    return number


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(f"AcousticBrainz field {label} must be text")
    return value.strip()


def _coverage_seconds(document: Mapping[str, object] | None) -> float:
    if document is None:
        return 0.0
    value = _finite_number(_nested(document, "metadata.audio_properties.length"), "length")
    return max(0.0, value or 0.0)


def _measurement_confidence(coverage_seconds: float, *, ceiling: float) -> float:
    coverage_factor = min(1.0, coverage_seconds / 30.0) if coverage_seconds > 0 else 0.75
    return round(ceiling * coverage_factor, 6)


def _positive_score(
    document: Mapping[str, object], task: str, positive_class: str
) -> float | None:
    value = _finite_number(
        _nested(document, f"highlevel.{task}.all.{positive_class}"),
        f"highlevel.{task}.all.{positive_class}",
    )
    if value is not None and not 0 <= value <= 1:
        raise ProviderError("AcousticBrainz high-level score must be in [0, 1]")
    return value


def _observations_for_document(
    track_id: str, document: AcousticBrainzDocument
) -> tuple[FeatureObservation, ...]:
    observations: list[FeatureObservation] = []
    low = document.low_level
    low_coverage = _coverage_seconds(low)
    if low is not None:
        low_confidence = _measurement_confidence(low_coverage, ceiling=0.85)
        descriptor_values: list[float] = []
        descriptor_complete = True
        for path, feature_name in _LOW_NUMBER_FEATURES:
            value = _finite_number(_nested(low, path), path)
            if value is None:
                descriptor_complete = False
                continue
            observations.append(
                FeatureObservation(
                    track_id=track_id,
                    feature_name=feature_name,
                    value=value,
                    feature_source=LOW_LEVEL_SOURCE,
                    source_version=LOW_LEVEL_VERSION,
                    coverage_seconds=low_coverage,
                    feature_confidence=low_confidence,
                )
            )
            descriptor_values.append(value)
        for path, feature_name in _LOW_TEXT_FEATURES:
            text_value = _optional_text(_nested(low, path), path)
            if text_value is None:
                continue
            observations.append(
                FeatureObservation(
                    track_id=track_id,
                    feature_name=feature_name,
                    value=text_value,
                    feature_source=LOW_LEVEL_SOURCE,
                    source_version=LOW_LEVEL_VERSION,
                    coverage_seconds=low_coverage,
                    feature_confidence=low_confidence,
                )
            )
        if descriptor_complete:
            observations.append(
                FeatureObservation(
                    track_id=track_id,
                    feature_name="acousticbrainz_descriptor_vector_v1",
                    value=tuple(descriptor_values),
                    feature_source=LOW_LEVEL_SOURCE,
                    source_version=LOW_LEVEL_VERSION,
                    coverage_seconds=low_coverage,
                    feature_confidence=low_confidence,
                )
            )

    high = document.high_level
    high_coverage = _coverage_seconds(high) or low_coverage
    if high is not None:
        high_confidence = _measurement_confidence(high_coverage, ceiling=0.70)
        learned_vector: list[float] = []
        learned_complete = True
        for task, positive_class, feature_name in _HIGH_LEVEL_TASKS:
            value = _positive_score(high, task, positive_class)
            if value is None:
                learned_complete = False
                continue
            observations.append(
                FeatureObservation(
                    track_id=track_id,
                    feature_name=feature_name,
                    value=value,
                    feature_source=HIGH_LEVEL_SOURCE,
                    source_version=HIGH_LEVEL_VERSION,
                    coverage_seconds=high_coverage,
                    feature_confidence=high_confidence,
                )
            )
            learned_vector.append(value)
        if learned_complete:
            observations.append(
                FeatureObservation(
                    track_id=track_id,
                    feature_name="acousticbrainz_learned_audio_vector_v1",
                    value=tuple(learned_vector),
                    feature_source=HIGH_LEVEL_SOURCE,
                    source_version=HIGH_LEVEL_VERSION,
                    coverage_seconds=high_coverage,
                    feature_confidence=high_confidence,
                )
            )
    return tuple(observations)


def fetch_acousticbrainz_features(
    matches: Iterable[ExternalIdentityMatch],
    provider: AcousticBrainzProvider,
    *,
    batch_size: int = 25,
    progress: Callable[[int, int], None] | None = None,
) -> AcousticFeatureResult:
    """Fetch frozen features for exact MBID matches and preserve Spotify IDs as track IDs."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 25:
        raise ProviderError("AcousticBrainz batch_size must be between 1 and 25")
    exact_matches = tuple(
        match
        for match in matches
        if match.match_status == "exact" and match.recording_mbid is not None
    )
    tracks_by_mbid: dict[str, list[ExternalIdentityMatch]] = defaultdict(list)
    for match in exact_matches:
        assert match.recording_mbid is not None
        tracks_by_mbid[match.recording_mbid.casefold()].append(match)
    mbids = sorted(tracks_by_mbid)
    documents: dict[str, AcousticBrainzDocument] = {}
    batch_count = math.ceil(len(mbids) / batch_size) if mbids else 0
    for batch_index, start in enumerate(range(0, len(mbids), batch_size), start=1):
        documents.update(provider.fetch(mbids[start : start + batch_size]))
        if progress is not None:
            progress(batch_index, batch_count)

    observations: list[FeatureObservation] = []
    low_covered: set[str] = set()
    high_covered: set[str] = set()
    for mbid, matched_tracks in tracks_by_mbid.items():
        document = documents.get(mbid)
        if document is None:
            continue
        for match in matched_tracks:
            if document.low_level is not None:
                low_covered.add(match.source_track_id)
            if document.high_level is not None:
                high_covered.add(match.source_track_id)
            observations.extend(_observations_for_document(match.source_track_id, document))
    ordered_observations = tuple(
        sorted(
            observations,
            key=lambda item: (
                item.track_id,
                item.feature_name,
                item.feature_source,
                item.source_version,
            ),
        )
    )
    match_by_track = {match.source_track_id: match for match in exact_matches}
    total_plays = sum(match.play_count for match in exact_matches)
    total_ms = sum(match.total_ms_played for match in exact_matches)

    def weighted_rate(covered: set[str], field_name: str) -> float:
        denominator = total_plays if field_name == "play_count" else total_ms
        numerator = sum(getattr(match_by_track[track_id], field_name) for track_id in covered)
        return round(numerator / denominator, 6) if denominator else 0.0

    feature_counts = Counter(item.feature_name for item in ordered_observations)
    considered = len(exact_matches)
    report = AcousticFeatureCoverageReport(
        exact_tracks_considered=considered,
        unique_recording_mbids=len(mbids),
        low_level_tracks_covered=len(low_covered),
        high_level_tracks_covered=len(high_covered),
        low_level_track_rate=round(len(low_covered) / considered, 6) if considered else 0.0,
        high_level_track_rate=round(len(high_covered) / considered, 6) if considered else 0.0,
        low_level_play_rate=weighted_rate(low_covered, "play_count"),
        high_level_play_rate=weighted_rate(high_covered, "play_count"),
        low_level_ms_played_rate=weighted_rate(low_covered, "total_ms_played"),
        high_level_ms_played_rate=weighted_rate(high_covered, "total_ms_played"),
        observations_written=len(ordered_observations),
        feature_counts=dict(feature_counts),
    )
    assert_privacy_safe(report.to_dict())
    return AcousticFeatureResult(observations=ordered_observations, report=report)


def write_acousticbrainz_result(
    result: AcousticFeatureResult, output_dir: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    features_path = write_feature_observations(
        result.observations, destination / "acousticbrainz_features.jsonl"
    )
    report_path = atomic_write_text(
        destination / "acousticbrainz_coverage_report.json",
        json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return features_path, report_path
