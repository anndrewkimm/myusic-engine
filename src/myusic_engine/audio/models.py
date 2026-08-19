"""Validated local-audio types shared by phase-3 extractors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

RIGHTS_BASES = frozenset({"owned", "licensed", "public_domain", "creative_commons"})


class AudioInputError(ValueError):
    """Raised when a permitted-audio input is absent, malformed, or unusable."""


@dataclass(frozen=True, slots=True)
class AudioAsset:
    """A private mapping from one stable track ID to one permitted local recording."""

    track_id: str
    path: Path
    rights_basis: str

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise AudioInputError("track_id must be non-empty text")
        if self.track_id != self.track_id.strip():
            raise AudioInputError("track_id must not contain surrounding whitespace")
        normalized_rights = self.rights_basis.strip().casefold()
        if normalized_rights not in RIGHTS_BASES:
            choices = ", ".join(sorted(RIGHTS_BASES))
            raise AudioInputError(f"rights_basis must be one of: {choices}")
        object.__setattr__(self, "rights_basis", normalized_rights)
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Finite mono floating-point samples and their exact sample rate."""

    samples: NDArray[np.float32]
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate_hz, bool) or not isinstance(self.sample_rate_hz, int):
            raise AudioInputError("sample_rate_hz must be an integer")
        if not 8_000 <= self.sample_rate_hz <= 384_000:
            raise AudioInputError("sample_rate_hz must be between 8000 and 384000")
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise AudioInputError("Decoded audio must be mono")
        if samples.size == 0:
            raise AudioInputError("Decoded audio must contain samples")
        if not np.all(np.isfinite(samples)):
            raise AudioInputError("Decoded audio samples must be finite")
        samples = np.ascontiguousarray(samples)
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def duration_seconds(self) -> float:
        return math.fsum((self.samples.size / self.sample_rate_hz,))
