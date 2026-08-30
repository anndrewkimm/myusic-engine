"""Exact-provenance audio representation joins shared by modeling and clustering."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from myusic_engine.features import FeatureCatalog, FeatureObservation, FeatureRecordError
from myusic_engine.modeling.config import AudioFeatureProfile, AudioInputSpec


class RepresentationError(ValueError):
    """Raised when selected feature observations do not satisfy their profile."""


@dataclass(frozen=True, slots=True)
class TrackAudioRepresentation:
    """Complete descriptor/embedding vectors available for one stable track ID."""

    track_id: str
    descriptors: tuple[float, ...] | None
    embedding: tuple[float, ...] | None


def _selected_values(
    catalog: FeatureCatalog,
    track_id: str,
    specs: tuple[AudioInputSpec, ...],
    minimum_confidence: float,
) -> tuple[float, ...] | None:
    values: list[float] = []
    for spec in specs:
        observation = catalog.get(track_id, spec.selector)
        if observation is None or observation.feature_confidence < minimum_confidence:
            return None
        if spec.dimensions == 1:
            if not isinstance(observation.value, float):
                raise RepresentationError(
                    f"Selected scalar {spec.selector.label} is not numeric"
                )
            selected: tuple[float, ...] = (observation.value,)
        else:
            if not isinstance(observation.value, tuple):
                raise RepresentationError(
                    f"Selected vector {spec.selector.label} is not a vector"
                )
            if len(observation.value) != spec.dimensions:
                raise RepresentationError(
                    f"Selected vector {spec.selector.label} has unexpected dimensions"
                )
            selected = observation.value
        if any(not math.isfinite(value) for value in selected):
            raise RepresentationError("Selected audio representation contains non-finite values")
        values.extend(selected)
    return tuple(values)


class ProfiledFeatureCatalog:
    """Join exact source/version observations without silently mixing providers."""

    def __init__(
        self,
        observations: Iterable[FeatureObservation],
        profile: AudioFeatureProfile,
    ) -> None:
        records = tuple(observations)
        try:
            self._catalog = FeatureCatalog(records)
        except FeatureRecordError as exc:
            raise RepresentationError(str(exc)) from exc
        self.profile = profile
        self._track_ids = frozenset(record.track_id for record in records)
        self._representations: dict[str, TrackAudioRepresentation] = {}
        embedding_specs = (
            (profile.embedding_input,) if profile.embedding_input is not None else ()
        )
        for track_id in sorted(self._track_ids):
            descriptors = (
                _selected_values(
                    self._catalog,
                    track_id,
                    profile.descriptor_inputs,
                    profile.minimum_confidence,
                )
                if profile.descriptor_inputs
                else None
            )
            embedding = (
                _selected_values(
                    self._catalog,
                    track_id,
                    embedding_specs,
                    profile.minimum_confidence,
                )
                if embedding_specs
                else None
            )
            self._representations[track_id] = TrackAudioRepresentation(
                track_id=track_id,
                descriptors=descriptors,
                embedding=embedding,
            )

    @property
    def descriptor_feature_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for spec in self.profile.descriptor_inputs
            for name in spec.feature_names
        )

    @property
    def embedding_feature_names(self) -> tuple[str, ...]:
        if self.profile.embedding_input is None:
            return ()
        return self.profile.embedding_input.feature_names

    @property
    def descriptor_tracks(self) -> frozenset[str]:
        return frozenset(
            track_id
            for track_id, representation in self._representations.items()
            if representation.descriptors is not None
        )

    @property
    def embedding_tracks(self) -> frozenset[str]:
        return frozenset(
            track_id
            for track_id, representation in self._representations.items()
            if representation.embedding is not None
        )

    @property
    def fair_cohort_tracks(self) -> frozenset[str]:
        cohorts: list[frozenset[str]] = []
        if self.profile.descriptor_inputs:
            cohorts.append(self.descriptor_tracks)
        if self.profile.embedding_input is not None:
            cohorts.append(self.embedding_tracks)
        if not cohorts:
            return frozenset()
        fair = set(cohorts[0])
        for cohort in cohorts[1:]:
            fair.intersection_update(cohort)
        return frozenset(fair)

    def get(self, track_id: str) -> TrackAudioRepresentation | None:
        return self._representations.get(track_id)

    def vector(
        self,
        track_id: str,
        *,
        include_descriptors: bool,
        include_embedding: bool,
    ) -> tuple[float, ...] | None:
        representation = self.get(track_id)
        if representation is None:
            return None
        values: list[float] = []
        if include_descriptors:
            if representation.descriptors is None:
                return None
            values.extend(representation.descriptors)
        if include_embedding:
            if representation.embedding is None:
                return None
            values.extend(representation.embedding)
        return tuple(values)
