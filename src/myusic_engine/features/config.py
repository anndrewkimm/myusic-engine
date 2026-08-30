"""Versioned configuration for the clean-room objective audio extractor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


class FeatureConfigError(ValueError):
    """Raised when a phase-3 feature configuration is malformed."""


@dataclass(frozen=True, slots=True)
class ObjectiveFeatureConfig:
    """Parameters that materially define objective feature values."""

    target_sample_rate_hz: int = 44_100
    minimum_coverage_seconds: float = 20.0
    source_version: str = "objective-dsp-0.3.0"
    frame_length: int = 4096
    hop_length: int = 1024
    silence_trim_db: float = 60.0
    rhythm_frame_length: int = 2048
    rhythm_hop_length: int = 512
    rhythm_onset_max_size: int = 5
    rhythm_onset_absolute_floor: float = 0.3
    rhythm_onset_mad_multiplier: float = 3.0
    rhythm_onset_wait_frames: int = 3
    minimum_tempo_beats: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_sample_rate_hz, bool)
            or not isinstance(self.target_sample_rate_hz, int)
            or not 8_000 <= self.target_sample_rate_hz <= 96_000
        ):
            raise FeatureConfigError("target_sample_rate_hz must be between 8000 and 96000")
        if (
            isinstance(self.minimum_coverage_seconds, bool)
            or not isinstance(self.minimum_coverage_seconds, (int, float))
            or not math.isfinite(self.minimum_coverage_seconds)
            or self.minimum_coverage_seconds <= 0
        ):
            raise FeatureConfigError("minimum_coverage_seconds must be positive and finite")
        if not isinstance(self.source_version, str) or not self.source_version.strip():
            raise FeatureConfigError("source_version must be non-empty")
        for name in (
            "frame_length",
            "hop_length",
            "rhythm_frame_length",
            "rhythm_hop_length",
            "rhythm_onset_max_size",
            "rhythm_onset_wait_frames",
            "minimum_tempo_beats",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise FeatureConfigError(f"{name} must be a positive integer")
        for name in ("rhythm_onset_absolute_floor", "rhythm_onset_mad_multiplier"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise FeatureConfigError(f"{name} must be a positive finite number")
        if not math.isfinite(self.silence_trim_db) or not 20 <= self.silence_trim_db <= 120:
            raise FeatureConfigError("silence_trim_db must be between 20 and 120")
        if self.hop_length > self.frame_length:
            raise FeatureConfigError("hop_length must not exceed frame_length")
        if self.rhythm_hop_length > self.rhythm_frame_length:
            raise FeatureConfigError("rhythm_hop_length must not exceed rhythm_frame_length")


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FeatureConfigError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], value)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeatureConfigError(f"{field_name} must be an integer")
    return value


def _number(record: Mapping[str, object], field_name: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureConfigError(f"{field_name} must be numeric")
    return float(value)


def _text(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise FeatureConfigError(f"{field_name} must be text")
    return value


def load_objective_feature_config(path: str | Path) -> ObjectiveFeatureConfig:
    """Load the objective extractor section from the versioned feature YAML."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FeatureConfigError(f"Could not read feature configuration: {exc}") from exc
    root = _mapping(payload, "feature configuration")
    allowed_root = {
        "schema_version",
        "feature_set",
        "provenance",
        "audio",
        "extractor",
        "features",
        "embedding",
        "learned_scores",
    }
    unknown_root = set(root) - allowed_root
    if unknown_root:
        raise FeatureConfigError(
            f"Unknown feature configuration fields: {', '.join(sorted(unknown_root))}"
        )
    if root.get("schema_version") != 1:
        raise FeatureConfigError("feature configuration schema_version must be 1")
    audio = _mapping(root.get("audio"), "audio")
    extractor = _mapping(root.get("extractor"), "extractor")
    unknown_audio = set(audio) - {
        "mono",
        "target_sample_rate_hz",
        "minimum_coverage_seconds",
    }
    if unknown_audio:
        raise FeatureConfigError(f"Unknown audio fields: {', '.join(sorted(unknown_audio))}")
    allowed_extractor = {
        "source_version",
        "frame_length",
        "hop_length",
        "silence_trim_db",
        "rhythm_frame_length",
        "rhythm_hop_length",
        "rhythm_onset_max_size",
        "rhythm_onset_absolute_floor",
        "rhythm_onset_mad_multiplier",
        "rhythm_onset_wait_frames",
        "minimum_tempo_beats",
    }
    unknown_extractor = set(extractor) - allowed_extractor
    if unknown_extractor:
        raise FeatureConfigError(
            f"Unknown extractor fields: {', '.join(sorted(unknown_extractor))}"
        )
    return ObjectiveFeatureConfig(
        target_sample_rate_hz=_integer(audio, "target_sample_rate_hz"),
        minimum_coverage_seconds=_number(audio, "minimum_coverage_seconds"),
        source_version=_text(extractor, "source_version"),
        frame_length=_integer(extractor, "frame_length"),
        hop_length=_integer(extractor, "hop_length"),
        silence_trim_db=_number(extractor, "silence_trim_db"),
        rhythm_frame_length=_integer(extractor, "rhythm_frame_length"),
        rhythm_hop_length=_integer(extractor, "rhythm_hop_length"),
        rhythm_onset_max_size=_integer(extractor, "rhythm_onset_max_size"),
        rhythm_onset_absolute_floor=_number(extractor, "rhythm_onset_absolute_floor"),
        rhythm_onset_mad_multiplier=_number(extractor, "rhythm_onset_mad_multiplier"),
        rhythm_onset_wait_frames=_integer(extractor, "rhythm_onset_wait_frames"),
        minimum_tempo_beats=_integer(extractor, "minimum_tempo_beats"),
    )
