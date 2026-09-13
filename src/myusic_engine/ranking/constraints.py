"""Exact-provenance hard filters shared by candidate ranking and local workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from myusic_engine.features import FeatureCatalog, FeatureSelector
from myusic_engine.ranking.similarity import (
    CategoricalFilter,
    NumericRangeFilter,
    SimilarityError,
    SimilarityFilter,
)


@dataclass(frozen=True, slots=True)
class RankingConstraints:
    filters: tuple[SimilarityFilter, ...] = ()
    excluded_artists: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in self.excluded_artists):
            raise SimilarityError("Excluded artist names must be nonempty strings")
        object.__setattr__(
            self,
            "excluded_artists",
            frozenset(value.strip().casefold() for value in self.excluded_artists),
        )

    def exclusion(self, track_id: str, artist: str | None, catalog: FeatureCatalog) -> str | None:
        if artist and artist.strip().casefold() in self.excluded_artists:
            return "excluded_artist"
        for rule in self.filters:
            observation = catalog.get(track_id, rule.selector)
            if observation is None or observation.feature_confidence < rule.minimum_confidence:
                if rule.required:
                    return "missing_filter_feature"
                continue
            value = observation.value
            if isinstance(rule, NumericRangeFilter):
                if not isinstance(value, float):
                    raise SimilarityError("Numeric filter selected a non-numeric feature")
                if rule.minimum is not None and value < rule.minimum:
                    return "feature_filter"
                if rule.maximum is not None and value > rule.maximum:
                    return "feature_filter"
            else:
                if not isinstance(value, str):
                    raise SimilarityError("Categorical filter selected a non-text feature")
                if (value if rule.case_sensitive else value.casefold()) not in rule.allowed_values:
                    return "feature_filter"
        return None

    def to_dict(self) -> dict[str, object]:
        rules: list[dict[str, object]] = []
        for rule in self.filters:
            row: dict[str, object] = {
                "feature_name": rule.selector.feature_name,
                "feature_source": rule.selector.feature_source,
                "source_version": rule.selector.source_version,
                "required": rule.required,
                "minimum_confidence": rule.minimum_confidence,
            }
            if isinstance(rule, NumericRangeFilter):
                row.update(minimum=rule.minimum, maximum=rule.maximum)
            else:
                row.update(
                    allowed_values=sorted(rule.allowed_values), case_sensitive=rule.case_sensitive
                )
            rules.append(row)
        return {
            "schema_version": 1,
            "filters": rules,
            "excluded_artists": sorted(self.excluded_artists),
        }


def load_ranking_constraints(path: str | Path) -> RankingConstraints:
    """Load a strict JSON filter document; omitted measurements never pass a required rule."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SimilarityError("Filter configuration is not valid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise SimilarityError("Filter configuration needs schema_version 1")
    if set(data) - {"schema_version", "filters", "excluded_artists"}:
        raise SimilarityError("Unknown filter configuration field")
    artists = data.get("excluded_artists", [])
    if not isinstance(artists, list) or any(not isinstance(value, str) for value in artists):
        raise SimilarityError("excluded_artists must be a list of strings")
    raw_rules = data.get("filters", [])
    if not isinstance(raw_rules, list):
        raise SimilarityError("filters must be a list")
    rules: list[SimilarityFilter] = []
    common = {"feature_name", "feature_source", "source_version", "required", "minimum_confidence"}
    for raw in raw_rules:
        if not isinstance(raw, Mapping):
            raise SimilarityError("Each filter must be an object")
        categorical = "allowed_values" in raw
        allowed = common | (
            {"allowed_values", "case_sensitive"} if categorical else {"minimum", "maximum"}
        )
        if set(raw) - allowed:
            raise SimilarityError("Unknown or mixed numeric/categorical filter fields")
        if any(
            not isinstance(raw.get(key), str)
            for key in ("feature_name", "feature_source", "source_version")
        ):
            raise SimilarityError("Each filter needs an exact feature name, source, and version")
        selector = FeatureSelector(
            raw["feature_name"], raw["feature_source"], raw["source_version"]
        )
        required = raw.get("required", True)
        if not isinstance(required, bool):
            raise SimilarityError("Filter required must be boolean")
        confidence = raw.get("minimum_confidence", 0.0)
        if categorical:
            values = raw["allowed_values"]
            sensitive = raw.get("case_sensitive", False)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise SimilarityError("allowed_values must be a list of strings")
            if not isinstance(sensitive, bool):
                raise SimilarityError("case_sensitive must be boolean")
            rules.append(
                CategoricalFilter(selector, frozenset(values), required, confidence, sensitive)
            )
        else:
            rules.append(
                NumericRangeFilter(
                    selector, raw.get("minimum"), raw.get("maximum"), required, confidence
                )
            )
    return RankingConstraints(tuple(rules), frozenset(artists))
