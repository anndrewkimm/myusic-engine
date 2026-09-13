from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from myusic_engine.audio import AudioAsset
from myusic_engine.evaluation.audio_validation import validate_audio_embeddings
from myusic_engine.features import FeatureObservation


class FrequencyBackend:
    def extract(self, track_id, audio):
        # Derive representation from actual audio, never the input track ID.
        spectrum = np.abs(np.fft.rfft(audio.samples))
        frequencies = np.fft.rfftfreq(len(audio.samples), 1 / audio.sample_rate_hz)
        peak = frequencies[np.argmax(spectrum)]
        vector = (1.0, 0.0) if peak < 500 else (0.0, 1.0)
        return SimpleNamespace(
            observation=FeatureObservation(
                track_id,
                "synthetic_embedding_v1",
                vector,
                "synthetic",
                "v1",
                audio.duration_seconds,
                1,
            )
        )


def test_recording_validation_uses_transformed_audio_and_persists_evidence(tmp_path):
    assets = []
    for index, frequency in enumerate((220, 880)):
        path = tmp_path / f"{index}.wav"
        samples = 0.1 * np.sin(2 * np.pi * frequency * np.arange(48000 * 2) / 48000)
        sf.write(path, samples, 48000)
        assets.append(AudioAsset(str(index), path, "owned"))
    report = validate_audio_embeddings(assets, FrequencyBackend(), tmp_path / "report")
    assert report["tracks"] == 2
    assert len(report["measurements"]) == 4
    assert all(item["correct_unique_top1"] for item in report["measurements"])
    assert (tmp_path / "report/audio_validation_report.json").is_file()
    with pytest.raises(ValueError, match="identical"):
        validate_audio_embeddings(
            [assets[0], AudioAsset("copy", assets[0].path, "owned")], FrequencyBackend(), tmp_path
        )
