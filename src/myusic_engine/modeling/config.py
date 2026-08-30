"""Versioned configuration for temporal taste datasets and model ablations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

from myusic_engine.features import FeatureSelector


class ModelingConfigError(ValueError):
    """Raised when modeling configuration is unsafe or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Controls period construction and conservative implicit-feedback labels."""

    dataset_version: str = "temporal_taste_v1"
    period_days: int = 90
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    minimum_train_periods: int = 4
    minimum_labeled_events: int = 1
    positive_fraction_threshold: float = 0.60
    negative_fraction_threshold: float = 0.40
    positive_minimum_ms: int = 30_000
    early_skip_maximum_ms: int = 30_000
    recency_half_life_days: float = 180.0
    maximum_sample_weight: float = 5.0
    complete_end_reasons: frozenset[str] = frozenset({"endplay", "trackdone"})
    skip_end_reasons: frozenset[str] = frozenset({"backbtn", "fwdbtn"})
    intentional_start_reasons: frozenset[str] = frozenset(
        {"backbtn", "clickrow", "fwdbtn", "playbtn", "remote"}
    )
    passive_start_reasons: frozenset[str] = frozenset(
        {"appload", "autoplay", "trackdone"}
    )

    def __post_init__(self) -> None:
        if not self.dataset_version.strip():
            raise ModelingConfigError("dataset_version must not be empty")
        if self.period_days < 1:
            raise ModelingConfigError("period_days must be positive")
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ModelingConfigError("validation_fraction and test_fraction must be in (0, 1)")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ModelingConfigError("validation and test fractions must leave training periods")
        if self.minimum_train_periods < 1:
            raise ModelingConfigError("minimum_train_periods must be positive")
        if self.minimum_labeled_events < 1:
            raise ModelingConfigError("minimum_labeled_events must be positive")
        if not 0.5 <= self.positive_fraction_threshold <= 1:
            raise ModelingConfigError("positive_fraction_threshold must be in [0.5, 1]")
        if not 0 <= self.negative_fraction_threshold <= 0.5:
            raise ModelingConfigError("negative_fraction_threshold must be in [0, 0.5]")
        if self.negative_fraction_threshold >= self.positive_fraction_threshold:
            raise ModelingConfigError("negative threshold must be below positive threshold")
        if self.positive_minimum_ms < 0 or self.early_skip_maximum_ms < 0:
            raise ModelingConfigError("listening-time thresholds must be non-negative")
        if self.recency_half_life_days <= 0 or self.maximum_sample_weight < 1:
            raise ModelingConfigError(
                "recency half-life and maximum sample weight must be positive"
            )
        if self.intentional_start_reasons & self.passive_start_reasons:
            raise ModelingConfigError("intentional and passive start reasons must not overlap")


@dataclass(frozen=True, slots=True)
class AudioInputSpec:
    """One exact scalar/vector observation flattened into a model input."""

    selector: FeatureSelector
    dimensions: int = 1

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ModelingConfigError("audio input dimensions must be positive")

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.dimensions == 1:
            return (self.selector.label,)
        return tuple(f"{self.selector.label}[{index}]" for index in range(self.dimensions))


@dataclass(frozen=True, slots=True)
class AudioFeatureProfile:
    """A provenance-locked descriptor and optional embedding representation."""

    profile_version: str
    descriptor_inputs: tuple[AudioInputSpec, ...]
    embedding_input: AudioInputSpec | None = None
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.profile_version.strip():
            raise ModelingConfigError("profile_version must not be empty")
        if not self.descriptor_inputs and self.embedding_input is None:
            raise ModelingConfigError("audio profile must select descriptors or an embedding")
        if not 0 <= self.minimum_confidence <= 1:
            raise ModelingConfigError("profile minimum_confidence must be in [0, 1]")
        selectors = [item.selector.label for item in self.descriptor_inputs]
        if self.embedding_input is not None:
            selectors.append(self.embedding_input.selector.label)
        if len(selectors) != len(set(selectors)):
            raise ModelingConfigError("audio profile contains duplicate exact selectors")


@dataclass(frozen=True, slots=True)
class TasteModelConfig:
    """Deterministic logistic-regression and ranking evaluation controls."""

    model_version: str = "taste_logistic_v1"
    random_seed: int = 1729
    regularization_c: float = 1.0
    maximum_iterations: int = 2_000
    ranking_k: int = 20

    def __post_init__(self) -> None:
        if not self.model_version.strip():
            raise ModelingConfigError("model_version must not be empty")
        if self.random_seed < 0 or self.regularization_c <= 0:
            raise ModelingConfigError("random_seed and regularization_c are invalid")
        if self.maximum_iterations < 1 or self.ranking_k < 1:
            raise ModelingConfigError("maximum_iterations and ranking_k must be positive")


@dataclass(frozen=True, slots=True)
class ModelingConfig:
    """Complete configuration boundary for dataset and model stages."""

    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    model: TasteModelConfig = field(default_factory=TasteModelConfig)
    profiles: Mapping[str, AudioFeatureProfile] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ModelingConfigError("modeling config schema_version must be 1")
        if any(not name.strip() for name in self.profiles):
            raise ModelingConfigError("profile names must not be empty")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelingConfigError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _integer(section: Mapping[str, object], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelingConfigError(f"{key} must be an integer")
    return value


def _number(section: Mapping[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingConfigError(f"{key} must be numeric")
    return float(value)


def _text(section: Mapping[str, object], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ModelingConfigError(f"{key} must be non-empty text")
    return value.strip()


def _reason_set(
    section: Mapping[str, object], key: str, default: frozenset[str]
) -> frozenset[str]:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ModelingConfigError(f"{key} must be a list of non-empty strings")
    return frozenset(cast(str, item).strip().casefold() for item in value)


def _selector(value: object, label: str) -> AudioInputSpec:
    section = _mapping(value, label)
    allowed = {"feature_name", "feature_source", "source_version", "dimensions"}
    unknown = set(section) - allowed
    if unknown:
        raise ModelingConfigError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    try:
        selector = FeatureSelector(
            feature_name=_text(section, "feature_name", ""),
            feature_source=_text(section, "feature_source", ""),
            source_version=_text(section, "source_version", ""),
        )
    except ValueError as exc:
        raise ModelingConfigError(str(exc)) from exc
    return AudioInputSpec(
        selector=selector,
        dimensions=_integer(section, "dimensions", 1),
    )


def _profile(value: object, label: str) -> AudioFeatureProfile:
    section = _mapping(value, label)
    allowed = {
        "profile_version",
        "minimum_confidence",
        "descriptor_inputs",
        "embedding_input",
    }
    unknown = set(section) - allowed
    if unknown:
        raise ModelingConfigError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    raw_descriptors = section.get("descriptor_inputs", [])
    if not isinstance(raw_descriptors, list):
        raise ModelingConfigError(f"{label}.descriptor_inputs must be an array")
    raw_embedding = section.get("embedding_input")
    return AudioFeatureProfile(
        profile_version=_text(section, "profile_version", ""),
        minimum_confidence=_number(section, "minimum_confidence", 0.0),
        descriptor_inputs=tuple(
            _selector(item, f"{label}.descriptor_inputs[{index}]")
            for index, item in enumerate(raw_descriptors)
        ),
        embedding_input=(
            None if raw_embedding is None else _selector(raw_embedding, f"{label}.embedding_input")
        ),
    )


def load_modeling_config(path: str | Path) -> ModelingConfig:
    """Load a strict version-1 modeling YAML file."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ModelingConfigError("Modeling config is not valid YAML") from exc
    root = _mapping(payload, "modeling config")
    allowed_root = {"schema_version", "temporal", "model", "profiles"}
    unknown_root = set(root) - allowed_root
    if unknown_root:
        raise ModelingConfigError(
            f"Unknown modeling config fields: {', '.join(sorted(unknown_root))}"
        )
    if root.get("schema_version") != 1:
        raise ModelingConfigError("modeling config schema_version must be 1")

    temporal_section = _mapping(root.get("temporal", {}), "temporal")
    allowed_temporal = {
        "dataset_version",
        "period_days",
        "validation_fraction",
        "test_fraction",
        "minimum_train_periods",
        "minimum_labeled_events",
        "positive_fraction_threshold",
        "negative_fraction_threshold",
        "positive_minimum_ms",
        "early_skip_maximum_ms",
        "recency_half_life_days",
        "maximum_sample_weight",
        "complete_end_reasons",
        "skip_end_reasons",
        "intentional_start_reasons",
        "passive_start_reasons",
    }
    unknown_temporal = set(temporal_section) - allowed_temporal
    if unknown_temporal:
        raise ModelingConfigError(
            f"Unknown temporal fields: {', '.join(sorted(unknown_temporal))}"
        )
    defaults = TemporalConfig()
    temporal = TemporalConfig(
        dataset_version=_text(
            temporal_section, "dataset_version", defaults.dataset_version
        ),
        period_days=_integer(temporal_section, "period_days", defaults.period_days),
        validation_fraction=_number(
            temporal_section, "validation_fraction", defaults.validation_fraction
        ),
        test_fraction=_number(temporal_section, "test_fraction", defaults.test_fraction),
        minimum_train_periods=_integer(
            temporal_section, "minimum_train_periods", defaults.minimum_train_periods
        ),
        minimum_labeled_events=_integer(
            temporal_section, "minimum_labeled_events", defaults.minimum_labeled_events
        ),
        positive_fraction_threshold=_number(
            temporal_section,
            "positive_fraction_threshold",
            defaults.positive_fraction_threshold,
        ),
        negative_fraction_threshold=_number(
            temporal_section,
            "negative_fraction_threshold",
            defaults.negative_fraction_threshold,
        ),
        positive_minimum_ms=_integer(
            temporal_section, "positive_minimum_ms", defaults.positive_minimum_ms
        ),
        early_skip_maximum_ms=_integer(
            temporal_section, "early_skip_maximum_ms", defaults.early_skip_maximum_ms
        ),
        recency_half_life_days=_number(
            temporal_section,
            "recency_half_life_days",
            defaults.recency_half_life_days,
        ),
        maximum_sample_weight=_number(
            temporal_section, "maximum_sample_weight", defaults.maximum_sample_weight
        ),
        complete_end_reasons=_reason_set(
            temporal_section, "complete_end_reasons", defaults.complete_end_reasons
        ),
        skip_end_reasons=_reason_set(
            temporal_section, "skip_end_reasons", defaults.skip_end_reasons
        ),
        intentional_start_reasons=_reason_set(
            temporal_section,
            "intentional_start_reasons",
            defaults.intentional_start_reasons,
        ),
        passive_start_reasons=_reason_set(
            temporal_section, "passive_start_reasons", defaults.passive_start_reasons
        ),
    )

    model_section = _mapping(root.get("model", {}), "model")
    allowed_model = {
        "model_version",
        "random_seed",
        "regularization_c",
        "maximum_iterations",
        "ranking_k",
    }
    unknown_model = set(model_section) - allowed_model
    if unknown_model:
        raise ModelingConfigError(f"Unknown model fields: {', '.join(sorted(unknown_model))}")
    model_defaults = TasteModelConfig()
    model = TasteModelConfig(
        model_version=_text(model_section, "model_version", model_defaults.model_version),
        random_seed=_integer(model_section, "random_seed", model_defaults.random_seed),
        regularization_c=_number(
            model_section, "regularization_c", model_defaults.regularization_c
        ),
        maximum_iterations=_integer(
            model_section, "maximum_iterations", model_defaults.maximum_iterations
        ),
        ranking_k=_integer(model_section, "ranking_k", model_defaults.ranking_k),
    )

    raw_profiles = _mapping(root.get("profiles", {}), "profiles")
    profiles = {
        name: _profile(value, f"profiles.{name}")
        for raw_name, value in raw_profiles.items()
        if (name := str(raw_name).strip())
    }
    if len(profiles) != len(raw_profiles):
        raise ModelingConfigError("profile names must not be empty")
    return ModelingConfig(temporal=temporal, model=model, profiles=profiles)
