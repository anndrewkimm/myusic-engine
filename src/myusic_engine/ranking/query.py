"""A natural-language query with an exact audio-space identity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from myusic_engine.features import FeatureSelector
from myusic_engine.ranking.similarity import SimilarityError, weighted_query_embedding


@dataclass(frozen=True, slots=True)
class SoundQuery:
    text: str
    vector: tuple[float, ...]
    selector: FeatureSelector
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise SimilarityError("Sound query text must not be empty")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight <= 0:
            raise SimilarityError("Sound query weight must be positive and finite")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "vector", weighted_query_embedding([(self.vector, 1.0)]))

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "vector": list(self.vector),
            "selector": self.selector.label,
            "weight": self.weight,
        }
