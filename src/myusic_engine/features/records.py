"""Versioned, source-tagged feature records and deterministic local serialization."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from myusic_engine.io import atomic_write_text
from myusic_engine.privacy import assert_privacy_safe


FeatureValue: TypeAlias = float | str | tuple[float, ...]
_FEATURE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_v[0-9]+$")
_VALUE_FIELDS = frozenset({"value_number", "value_text", "value_vector"})
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "track_id",
        "feature_name",
        "value_number",
        "value_text",
        "value_vector",
        "feature_source",
        "source_version",
        "coverage_seconds",
        "feature_confidence",
    }
)


class FeatureRecordError(ValueError):
    """Raised when a feature record is malformed or ambiguous."""


@dataclass(frozen=True, slots=True)
class FeatureSelector:
    """An exact feature definition, including measurement provenance."""

    feature_name: str
    feature_source: str
    source_version: str

    def __post_init__(self) -> None:
        _validate_identity_fields(
            track_id="selector",
            feature_name=self.feature_name,
            feature_source=self.feature_source,
            source_version=self.source_version,
        )

    @property
    def label(self) -> str:
        return f"{self.feature_name}@{self.feature_source}:{self.source_version}"

    def matches(self, observation: FeatureObservation) -> bool:
        return (
            observation.feature_name == self.feature_name
            and observation.feature_source == self.feature_source
            and observation.source_version == self.source_version
        )


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """One independently measured track feature with mandatory provenance."""

    track_id: str
    feature_name: str
    value: FeatureValue
    feature_source: str
    source_version: str
    coverage_seconds: float
    feature_confidence: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise FeatureRecordError("Feature observation schema_version must be 1")
        _validate_identity_fields(
            track_id=self.track_id,
            feature_name=self.feature_name,
            feature_source=self.feature_source,
            source_version=self.source_version,
        )
        coverage = _finite_number(self.coverage_seconds, "coverage_seconds")
        confidence = _finite_number(self.feature_confidence, "feature_confidence")
        if coverage < 0:
            raise FeatureRecordError("coverage_seconds must be non-negative")
        if not 0 <= confidence <= 1:
            raise FeatureRecordError("feature_confidence must be in [0, 1]")
        object.__setattr__(self, "coverage_seconds", coverage)
        object.__setattr__(self, "feature_confidence", confidence)
        object.__setattr__(self, "value", _validated_value(self.value))

    @property
    def selector(self) -> FeatureSelector:
        return FeatureSelector(
            feature_name=self.feature_name,
            feature_source=self.feature_source,
            source_version=self.source_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to the feature-observation v1 data contract."""

        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "feature_name": self.feature_name,
            "feature_source": self.feature_source,
            "source_version": self.source_version,
            "coverage_seconds": self.coverage_seconds,
            "feature_confidence": self.feature_confidence,
        }
        if isinstance(self.value, float):
            record["value_number"] = self.value
        elif isinstance(self.value, str):
            record["value_text"] = self.value
        else:
            record["value_vector"] = list(self.value)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> FeatureObservation:
        """Validate and construct an observation from contract-shaped data."""

        unknown_fields = set(record) - _RECORD_FIELDS
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise FeatureRecordError(f"Unknown feature observation fields: {names}")
        populated_value_fields = [field for field in _VALUE_FIELDS if field in record]
        if len(populated_value_fields) != 1:
            raise FeatureRecordError("Feature observation must contain exactly one value field")
        value_field = populated_value_fields[0]
        raw_value = record[value_field]
        if value_field == "value_vector":
            if not isinstance(raw_value, list):
                raise FeatureRecordError("value_vector must be an array")
            value: object = tuple(raw_value)
        else:
            value = raw_value
        return cls(
            schema_version=_required_integer(record, "schema_version"),
            track_id=_required_string(record, "track_id"),
            feature_name=_required_string(record, "feature_name"),
            value=cast(FeatureValue, value),
            feature_source=_required_string(record, "feature_source"),
            source_version=_required_string(record, "source_version"),
            coverage_seconds=_required_number(record, "coverage_seconds"),
            feature_confidence=_required_number(record, "feature_confidence"),
        )


def _validate_identity_fields(
    *, track_id: str, feature_name: str, feature_source: str, source_version: str
) -> None:
    if not isinstance(track_id, str) or not track_id.strip():
        raise FeatureRecordError("track_id must be non-empty text")
    if track_id != track_id.strip():
        raise FeatureRecordError("track_id must not contain surrounding whitespace")
    if not isinstance(feature_name, str) or _FEATURE_NAME_PATTERN.fullmatch(feature_name) is None:
        raise FeatureRecordError("feature_name must be lowercase, versioned, and end in _vN")
    if not isinstance(feature_source, str) or not feature_source.strip():
        raise FeatureRecordError("feature_source must be non-empty text")
    if feature_source != feature_source.strip():
        raise FeatureRecordError("feature_source must not contain surrounding whitespace")
    if not isinstance(source_version, str) or not source_version.strip():
        raise FeatureRecordError("source_version must be non-empty text")
    if source_version != source_version.strip():
        raise FeatureRecordError("source_version must not contain surrounding whitespace")


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureRecordError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise FeatureRecordError(f"{field_name} must be finite")
    return normalized


def _validated_value(value: object) -> FeatureValue:
    if isinstance(value, bool):
        raise FeatureRecordError("Feature value must not be boolean")
    if isinstance(value, (int, float)):
        return _finite_number(value, "value_number")
    if isinstance(value, str):
        if not value.strip():
            raise FeatureRecordError("value_text must not be empty")
        return value.strip()
    if isinstance(value, tuple):
        if not value:
            raise FeatureRecordError("value_vector must not be empty")
        return tuple(_finite_number(item, "value_vector item") for item in value)
    raise FeatureRecordError("Feature value must be a number, text, or numeric tuple")


def _required_string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise FeatureRecordError(f"{field_name} must be text")
    return value


def _required_integer(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeatureRecordError(f"{field_name} must be an integer")
    return value


def _required_number(record: Mapping[str, object], field_name: str) -> float:
    if field_name not in record:
        raise FeatureRecordError(f"Feature observation is missing {field_name}")
    return _finite_number(record[field_name], field_name)


class FeatureCatalog:
    """Exact-key lookup that rejects duplicate or silently competing measurements."""

    def __init__(self, observations: Iterable[FeatureObservation]) -> None:
        self._records: dict[tuple[str, str, str, str], FeatureObservation] = {}
        for observation in observations:
            key = (
                observation.track_id,
                observation.feature_name,
                observation.feature_source,
                observation.source_version,
            )
            if key in self._records:
                raise FeatureRecordError(
                    "Duplicate observation for the same track, feature, source, and version"
                )
            self._records[key] = observation

    def get(self, track_id: str, selector: FeatureSelector) -> FeatureObservation | None:
        return self._records.get(
            (
                track_id,
                selector.feature_name,
                selector.feature_source,
                selector.source_version,
            )
        )


def read_feature_observations(path: str | Path) -> tuple[FeatureObservation, ...]:
    """Read contract-shaped feature observations from JSON Lines."""

    observations: list[FeatureObservation] = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeatureRecordError(
                    f"Feature observation line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise FeatureRecordError(
                    f"Feature observation line {line_number} must be an object"
                )
            observations.append(FeatureObservation.from_dict(cast(Mapping[str, object], payload)))
    FeatureCatalog(observations)
    return tuple(observations)


def write_feature_observations(
    observations: Iterable[FeatureObservation], path: str | Path
) -> Path:
    """Sort and atomically write feature observations as JSON Lines."""

    ordered = sorted(
        observations,
        key=lambda item: (
            item.track_id,
            item.feature_name,
            item.feature_source,
            item.source_version,
        ),
    )
    FeatureCatalog(ordered)
    serialized: list[str] = []
    for observation in ordered:
        record = observation.to_dict()
        assert_privacy_safe(record)
        serialized.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    content = "\n".join(serialized)
    if content:
        content += "\n"
    return atomic_write_text(path, content)
