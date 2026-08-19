import json
from pathlib import Path

import numpy as np
import soundfile as sf

from myusic_engine.cli import main
from myusic_engine.features import read_feature_observations


def test_cli_analyzes_private_manifest_without_embedding_model(tmp_path: Path) -> None:
    sample_rate = 22_050
    time = np.arange(3 * sample_rate, dtype=np.float32) / sample_rate
    samples = (
        0.08 * np.sin(2 * np.pi * 261.63 * time)
        + 0.08 * np.sin(2 * np.pi * 329.63 * time)
        + 0.08 * np.sin(2 * np.pi * 392.00 * time)
    )
    audio_path = tmp_path / "permitted.wav"
    sf.write(audio_path, samples, sample_rate, subtype="PCM_16")
    manifest_path = tmp_path / "audio_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "track_id": "synthetic-track-a",
                "audio_path": audio_path.name,
                "rights_basis": "owned",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "features.jsonl"

    exit_code = main(
        [
            "analyze-audio",
            str(manifest_path),
            "--output",
            str(output_path),
            "--skip-embeddings",
        ]
    )

    observations = read_feature_observations(output_path)
    assert exit_code == 0
    assert len(observations) >= 14
    assert {observation.track_id for observation in observations} == {"synthetic-track-a"}
    assert {observation.feature_source for observation in observations} == {"clean_room_audio"}
