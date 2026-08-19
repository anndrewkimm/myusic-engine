"""Cross-platform audio decoding and deterministic polyphase resampling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from myusic_engine.audio.models import AudioInputError, DecodedAudio


def resample_audio(audio: DecodedAudio, target_sample_rate_hz: int) -> DecodedAudio:
    """Resample mono audio with a rational polyphase filter."""

    if isinstance(target_sample_rate_hz, bool) or not isinstance(target_sample_rate_hz, int):
        raise AudioInputError("target_sample_rate_hz must be an integer")
    if not 8_000 <= target_sample_rate_hz <= 384_000:
        raise AudioInputError("target_sample_rate_hz must be between 8000 and 384000")
    if audio.sample_rate_hz == target_sample_rate_hz:
        return audio
    divisor = math.gcd(audio.sample_rate_hz, target_sample_rate_hz)
    samples = resample_poly(
        audio.samples,
        up=target_sample_rate_hz // divisor,
        down=audio.sample_rate_hz // divisor,
        padtype="constant",
    )
    return DecodedAudio(np.asarray(samples, dtype=np.float32), target_sample_rate_hz)


def decode_audio(path: str | Path, *, target_sample_rate_hz: int | None = None) -> DecodedAudio:
    """Decode an audio file, mix channels to mono, and optionally resample it."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise AudioInputError(
            "Audio decoding requires the phase3 extra: pip install -e '.[phase3]'"
        ) from exc

    source = Path(path)
    if not source.is_file():
        raise AudioInputError(f"Audio input is not a file: {source}")
    try:
        samples, sample_rate_hz = sf.read(
            source,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise AudioInputError(f"Could not decode audio file {source}: {exc}") from exc
    if samples.shape[1] < 1:
        raise AudioInputError(f"Audio file has no channels: {source}")
    mono = np.asarray(np.mean(samples, axis=1, dtype=np.float32), dtype=np.float32)
    decoded = DecodedAudio(mono, int(sample_rate_hz))
    if target_sample_rate_hz is not None:
        return resample_audio(decoded, target_sample_rate_hz)
    return decoded
