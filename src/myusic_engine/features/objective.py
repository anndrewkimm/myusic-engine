"""Deterministic clean-room descriptors computed from permitted local audio."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from numpy.typing import NDArray

from myusic_engine.audio import DecodedAudio, resample_audio
from myusic_engine.audio._dependencies import load_librosa
from myusic_engine.features.config import ObjectiveFeatureConfig
from myusic_engine.features.records import FeatureObservation

_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_PROFILE = np.asarray(
    (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
    dtype=np.float64,
)
_MINOR_PROFILE = np.asarray(
    (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    dtype=np.float64,
)


class AudioAnalysisError(ValueError):
    """Raised when objective descriptors cannot be measured defensibly."""


@dataclass(frozen=True, slots=True)
class _TempoAnalysis:
    tempo_bpm: float | None
    confidence: float
    beat_strength: float
    onset_rate_hz: float


@dataclass(frozen=True, slots=True)
class _KeyAnalysis:
    key: str
    mode: str
    strength: float
    chroma: tuple[float, ...]


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _key_analysis(chroma_frames: NDArray[np.float64]) -> _KeyAnalysis:
    chroma = np.asarray(np.mean(chroma_frames, axis=1, dtype=np.float64), dtype=np.float64)
    total = float(np.sum(chroma))
    if not math.isfinite(total) or total <= 1e-12:
        raise AudioAnalysisError("Audio has insufficient tonal energy for chroma")
    chroma /= total
    centered = chroma - float(np.mean(chroma))
    centered_norm = float(np.linalg.norm(centered))
    if centered_norm <= 1e-12:
        raise AudioAnalysisError("Audio chroma is too uniform for a key estimate")
    normalized_chroma = centered / centered_norm
    candidates: list[tuple[float, int, str]] = []
    for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        normalized_profile = profile - float(np.mean(profile))
        normalized_profile /= float(np.linalg.norm(normalized_profile))
        for root in range(12):
            score = float(np.dot(normalized_chroma, np.roll(normalized_profile, root)))
            candidates.append((score, root, mode))
    candidates.sort(reverse=True)
    best_score, root, mode = candidates[0]
    entropy = -float(np.sum(chroma * np.log(np.maximum(chroma, 1e-12)))) / math.log(12)
    tonal_concentration = _clamp_ratio(1.0 - entropy)
    strength = _clamp_ratio(max(0.0, best_score) * math.sqrt(tonal_concentration))
    return _KeyAnalysis(
        key=f"{_PITCH_CLASSES[root]} {mode}",
        mode=mode,
        strength=strength,
        chroma=tuple(float(value) for value in chroma),
    )


def _tempo_analysis(
    samples: NDArray[np.float32], sample_rate_hz: int, config: ObjectiveFeatureConfig
) -> _TempoAnalysis:
    librosa = load_librosa()
    onset_envelope = librosa.onset.onset_strength(
        y=samples,
        sr=sample_rate_hz,
        n_fft=config.rhythm_frame_length,
        hop_length=config.rhythm_hop_length,
        max_size=config.rhythm_onset_max_size,
    )
    onset_median = float(np.median(onset_envelope))
    onset_mad = float(np.median(np.abs(onset_envelope - onset_median)))
    robust_floor = onset_median + (
        config.rhythm_onset_mad_multiplier * 1.4826 * onset_mad
    )
    onset_floor = max(config.rhythm_onset_absolute_floor, robust_floor)
    gated_envelope = np.maximum(onset_envelope - onset_floor, 0.0)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=gated_envelope,
        sr=sample_rate_hz,
        hop_length=config.rhythm_hop_length,
        units="frames",
        normalize=False,
        delta=1e-6,
        wait=config.rhythm_onset_wait_frames,
    )
    duration_seconds = samples.size / sample_rate_hz
    onset_rate = float(len(onset_frames) / duration_seconds)
    if gated_envelope.size < 2 or float(np.max(gated_envelope)) <= 1e-10:
        return _TempoAnalysis(None, 0.0, 0.0, onset_rate)
    raw_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=gated_envelope,
        sr=sample_rate_hz,
        hop_length=config.rhythm_hop_length,
        sparse=True,
    )
    tempo_values = np.asarray(raw_tempo, dtype=np.float64).reshape(-1)
    tempo = float(tempo_values[0]) if tempo_values.size else float("nan")
    if not math.isfinite(tempo) or not 20.0 <= tempo <= 300.0:
        return _TempoAnalysis(None, 0.0, 0.0, onset_rate)
    beat_frames_array = np.asarray(beat_frames, dtype=np.int64)
    if beat_frames_array.size < config.minimum_tempo_beats:
        return _TempoAnalysis(None, 0.0, 0.0, onset_rate)
    period = int(round(60.0 * sample_rate_hz / (tempo * config.rhythm_hop_length)))
    centered = np.asarray(gated_envelope, dtype=np.float64) - float(np.mean(gated_envelope))
    denominator = float(np.dot(centered, centered))
    periodicity = 0.0
    if denominator > 1e-12 and 0 < period < centered.size:
        periodicity = _clamp_ratio(
            float(np.dot(centered[:-period], centered[period:])) / denominator
        )
    beat_count_factor = min(1.0, beat_frames_array.size / 8.0)
    confidence = _clamp_ratio(math.sqrt(periodicity) * beat_count_factor)
    active_onsets = gated_envelope[gated_envelope > 0.0]
    peak = float(np.percentile(active_onsets, 95)) if active_onsets.size else 0.0
    beat_strength = 0.0
    valid_beats = beat_frames_array[beat_frames_array < gated_envelope.size]
    if peak > 1e-12 and valid_beats.size:
        beat_strength = _clamp_ratio(float(np.mean(gated_envelope[valid_beats])) / peak)
    return _TempoAnalysis(tempo, confidence, beat_strength, onset_rate)


class ObjectiveFeatureExtractor:
    """Compute stable objective descriptors before any learned subjective proxies."""

    feature_source = "clean_room_audio"

    def __init__(self, config: ObjectiveFeatureConfig | None = None) -> None:
        self.config = config or ObjectiveFeatureConfig()

    def extract(self, track_id: str, audio: DecodedAudio) -> tuple[FeatureObservation, ...]:
        """Extract one provenance-tagged observation per available descriptor."""

        if audio.sample_rate_hz != self.config.target_sample_rate_hz:
            audio = resample_audio(audio, self.config.target_sample_rate_hz)
        samples = audio.samples
        sample_rate_hz = audio.sample_rate_hz
        duration = audio.duration_seconds
        if duration < 1.0:
            raise AudioAnalysisError("At least one second of audio is required")
        rms_signal = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms_signal <= 1e-7:
            raise AudioAnalysisError("Audio is effectively silent")
        coverage_confidence = _clamp_ratio(duration / self.config.minimum_coverage_seconds)
        librosa = load_librosa()

        spectrum = np.abs(
            librosa.stft(
                samples,
                n_fft=self.config.frame_length,
                hop_length=self.config.hop_length,
                win_length=self.config.frame_length,
                window="hann",
                center=True,
                pad_mode="constant",
            )
        ).astype(np.float64, copy=False)
        power = np.square(spectrum)
        mean_power = np.asarray(np.mean(power, axis=1, dtype=np.float64), dtype=np.float64)
        total_power = float(np.sum(mean_power))
        if total_power <= 1e-16:
            raise AudioAnalysisError("Audio has insufficient spectral energy")
        frequencies = librosa.fft_frequencies(sr=sample_rate_hz, n_fft=self.config.frame_length)
        bass_150 = float(np.sum(mean_power[frequencies <= 150.0]) / total_power)
        bass_250 = float(np.sum(mean_power[frequencies <= 250.0]) / total_power)
        spectral_centroid = float(np.dot(frequencies, mean_power) / total_power)
        cumulative_power = np.cumsum(mean_power)
        rolloff_index = int(np.searchsorted(cumulative_power, 0.85 * total_power))
        rolloff_index = min(rolloff_index, frequencies.size - 1)
        spectral_rolloff = float(frequencies[rolloff_index])
        positive_power = np.maximum(power, 1e-20)
        frame_flatness = np.exp(np.mean(np.log(positive_power), axis=0)) / np.mean(
            positive_power, axis=0
        )
        spectral_flatness = _clamp_ratio(float(np.mean(frame_flatness)))

        rms_frames = librosa.feature.rms(
            y=samples,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
            center=True,
            pad_mode="constant",
        )[0]
        rms_db = 20.0 * np.log10(np.maximum(rms_frames, 1e-10))
        dynamic_range_db = max(0.0, float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)))
        try:
            loudness = float(pyln.Meter(sample_rate_hz).integrated_loudness(samples))
        except (FloatingPointError, OverflowError, ValueError):
            loudness = float("nan")

        chroma_frames = librosa.feature.chroma_stft(
            S=power,
            sr=sample_rate_hz,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length,
            tuning=0.0,
            norm=2,
        ).astype(np.float64, copy=False)
        key = _key_analysis(chroma_frames)
        tempo = _tempo_analysis(samples, sample_rate_hz, self.config)
        mfcc = librosa.feature.mfcc(
            y=samples,
            sr=sample_rate_hz,
            n_mfcc=20,
            n_fft=self.config.frame_length,
            hop_length=self.config.hop_length,
        ).astype(np.float64, copy=False)

        def observation(
            feature_name: str,
            value: float | str | tuple[float, ...],
            confidence: float = coverage_confidence,
        ) -> FeatureObservation:
            return FeatureObservation(
                track_id=track_id,
                feature_name=feature_name,
                value=value,
                feature_source=self.feature_source,
                source_version=self.config.source_version,
                coverage_seconds=duration,
                feature_confidence=_clamp_ratio(confidence),
            )

        observations = [
            observation("beat_strength_v1", tempo.beat_strength),
            observation("onset_rate_hz_v1", tempo.onset_rate_hz),
            observation("key_estimate_v1", key.key, coverage_confidence * key.strength),
            observation("mode_estimate_v1", key.mode, coverage_confidence * key.strength),
            observation("key_strength_v1", key.strength),
            observation("bass_energy_ratio_150hz_v1", bass_150),
            observation("bass_energy_ratio_250hz_v1", bass_250),
            observation("spectral_centroid_hz_v1", spectral_centroid),
            observation("spectral_rolloff_85_hz_v1", spectral_rolloff),
            observation("spectral_flatness_v1", spectral_flatness),
            observation("dynamic_range_db_v1", dynamic_range_db),
            observation("chroma_mean_v1", key.chroma),
            observation("mfcc_mean_v1", tuple(float(value) for value in np.mean(mfcc, axis=1))),
            observation("mfcc_std_v1", tuple(float(value) for value in np.std(mfcc, axis=1))),
        ]
        if tempo.tempo_bpm is not None:
            observations.append(
                observation(
                    "tempo_bpm_estimate_v1",
                    tempo.tempo_bpm,
                    coverage_confidence * tempo.confidence,
                )
            )
        if math.isfinite(loudness):
            observations.append(observation("integrated_loudness_lufs_v1", loudness))
        return tuple(observations)
