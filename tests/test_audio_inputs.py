import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from myusic_engine.audio import AudioInputError, decode_audio, read_audio_manifest


def _write_stereo_wav(path: Path, sample_rate_hz: int = 22_050) -> None:
    time = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    stereo = np.column_stack(
        (
            0.4 * np.sin(2 * np.pi * 220 * time),
            0.2 * np.sin(2 * np.pi * 440 * time),
        )
    )
    sf.write(path, stereo, sample_rate_hz, subtype="PCM_16")


def test_manifest_resolves_relative_audio_and_records_rights(tmp_path: Path) -> None:
    audio_path = tmp_path / "fixture.wav"
    _write_stereo_wav(audio_path)
    manifest = tmp_path / "audio_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "track_id": "synthetic-track-a",
                "audio_path": "fixture.wav",
                "rights_basis": "creative_commons",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assets = read_audio_manifest(manifest)

    assert len(assets) == 1
    assert assets[0].path == audio_path.resolve()
    assert assets[0].rights_basis == "creative_commons"


def test_decoder_mixes_to_mono_and_resamples(tmp_path: Path) -> None:
    audio_path = tmp_path / "fixture.wav"
    _write_stereo_wav(audio_path)

    decoded = decode_audio(audio_path, target_sample_rate_hz=16_000)

    assert decoded.sample_rate_hz == 16_000
    assert decoded.samples.ndim == 1
    assert decoded.duration_seconds == pytest.approx(1.0, abs=0.001)
    assert np.all(np.isfinite(decoded.samples))


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {
                    "track_id": "track-a",
                    "audio_path": "fixture.wav",
                    "rights_basis": "stream-rip",
                }
            ],
            "rights_basis",
        ),
        (
            [
                {
                    "track_id": "track-a",
                    "audio_path": "fixture.wav",
                    "rights_basis": "owned",
                },
                {
                    "track_id": "track-a",
                    "audio_path": "fixture.wav",
                    "rights_basis": "owned",
                },
            ],
            "Duplicate track_id",
        ),
    ],
)
def test_manifest_rejects_untrusted_rights_and_duplicate_ids(
    tmp_path: Path, records: list[dict[str, str]], message: str
) -> None:
    _write_stereo_wav(tmp_path / "fixture.wav")
    manifest = tmp_path / "audio_manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(AudioInputError, match=message):
        read_audio_manifest(manifest)
