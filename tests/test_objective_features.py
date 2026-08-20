import math

import numpy as np
import pytest

from myusic_engine.audio import DecodedAudio
from myusic_engine.features.config import ObjectiveFeatureConfig
from myusic_engine.features.objective import AudioAnalysisError, ObjectiveFeatureExtractor

SAMPLE_RATE = 22_050
CONFIG = ObjectiveFeatureConfig(
    target_sample_rate_hz=SAMPLE_RATE,
    minimum_coverage_seconds=8.0,
    frame_length=2048,
    hop_length=512,
    rhythm_frame_length=1024,
    rhythm_hop_length=256,
)


def _tones(frequencies: tuple[float, ...], duration: float, amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(round(duration * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    signal = sum(np.sin(2 * np.pi * frequency * time) for frequency in frequencies)
    signal *= amplitude / max(1, len(frequencies))
    return np.asarray(signal, dtype=np.float32)


def _values(samples: np.ndarray) -> dict[str, float | str | tuple[float, ...]]:
    observations = ObjectiveFeatureExtractor(CONFIG).extract(
        "synthetic-track", DecodedAudio(samples, SAMPLE_RATE)
    )
    return {observation.feature_name: observation.value for observation in observations}


def test_known_click_track_has_expected_tempo() -> None:
    duration = 12.0
    samples = _tones((261.63, 329.63, 392.00), duration, amplitude=0.04)
    pulse = np.hanning(round(0.02 * SAMPLE_RATE)).astype(np.float32)
    for start in np.arange(0.5, duration, 0.5):
        index = round(start * SAMPLE_RATE)
        samples[index : index + pulse.size] += pulse

    values = _values(samples)

    assert float(values["tempo_bpm_estimate_v1"]) == pytest.approx(120.0, abs=3.0)
    assert float(values["beat_strength_v1"]) > 0.3
    assert float(values["onset_rate_hz_v1"]) == pytest.approx(2.0, abs=0.25)


def test_stationary_tone_abstains_from_spurious_rhythm_measurements() -> None:
    observations = ObjectiveFeatureExtractor(CONFIG).extract(
        "stationary-tone",
        DecodedAudio(_tones((440.0,), 8.0), SAMPLE_RATE),
    )
    values = {observation.feature_name: observation.value for observation in observations}

    assert "tempo_bpm_estimate_v1" not in values
    assert values["beat_strength_v1"] == 0.0
    assert values["onset_rate_hz_v1"] == 0.0


def test_bass_energy_ratios_respond_monotonically_to_frequency() -> None:
    bass = _values(_tones((80.0,), 5.0))
    treble = _values(_tones((2_000.0,), 5.0))

    assert float(bass["bass_energy_ratio_150hz_v1"]) > 0.9
    assert float(treble["bass_energy_ratio_150hz_v1"]) < 0.01
    assert float(bass["spectral_centroid_hz_v1"]) < float(treble["spectral_centroid_hz_v1"])


def test_chroma_and_timbre_vectors_have_pinned_dimensions() -> None:
    values = _values(_tones((261.63, 329.63, 392.00), 5.0))

    chroma = values["chroma_mean_v1"]
    mfcc_mean = values["mfcc_mean_v1"]
    mfcc_std = values["mfcc_std_v1"]
    assert isinstance(chroma, tuple) and len(chroma) == 12
    assert isinstance(mfcc_mean, tuple) and len(mfcc_mean) == 20
    assert isinstance(mfcc_std, tuple) and len(mfcc_std) == 20
    assert math.fsum(chroma) == pytest.approx(1.0)
    assert values["mode_estimate_v1"] in {"major", "minor"}


def test_loudness_changes_but_spectral_ratios_survive_gain_change() -> None:
    source = _tones((80.0, 440.0), 5.0, amplitude=0.4)
    loud = _values(source)
    quiet = _values(source * 0.25)

    assert float(loud["integrated_loudness_lufs_v1"]) > float(quiet["integrated_loudness_lufs_v1"])
    assert float(loud["bass_energy_ratio_150hz_v1"]) == pytest.approx(
        float(quiet["bass_energy_ratio_150hz_v1"]), rel=1e-5
    )


def test_silence_is_rejected_instead_of_emitting_fake_measurements() -> None:
    with pytest.raises(AudioAnalysisError, match="silent"):
        ObjectiveFeatureExtractor(CONFIG).extract(
            "silent", DecodedAudio(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        )
