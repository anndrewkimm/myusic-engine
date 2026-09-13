"""Content-addressed per-track checkpoints for long-running local extraction jobs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

from myusic_engine.audio import AudioAsset
from myusic_engine.embeddings import DiscogsEffnetOnnxBackend
from myusic_engine.embeddings.clap import SOURCE_VERSION as CLAP_VERSION
from myusic_engine.embeddings.clap import ClapBackend
from myusic_engine.embeddings.discogs_effnet import SOURCE_VERSION as DISCOGS_VERSION
from myusic_engine.embeddings.discogs_effnet import file_sha256
from myusic_engine.features.config import ObjectiveFeatureConfig
from myusic_engine.features.learned import DiscogsEffnetFeatureHeadBackend
from myusic_engine.features.pipeline import (
    AudioEmbeddingBackend,
    AudioFeaturePipelineResult,
    analyze_audio_assets,
)
from myusic_engine.features.records import FeatureObservation
from myusic_engine.io import atomic_write_text


def analyze_audio_job(
    assets: Iterable[AudioAsset],
    *,
    cache_dir: str | Path,
    config: ObjectiveFeatureConfig | None = None,
    embedding_backend: AudioEmbeddingBackend | None = None,
    feature_head_backend: DiscogsEffnetFeatureHeadBackend | None = None,
    window_output_dir: str | Path | None = None,
    progress: Callable[[int, int, bool], None] | None = None,
) -> AudioFeaturePipelineResult:
    """Resume only exact input/config/model matches; publish each completed track atomically.

    Invalid checkpoints and missing/changed window artifacts are recomputed. Failed tracks
    abort the command, preserving earlier checkpoints and any previous complete output file.
    """
    active = config or ObjectiveFeatureConfig()
    backend_identity = None
    if isinstance(embedding_backend, ClapBackend):
        backend_identity = CLAP_VERSION
    elif isinstance(embedding_backend, DiscogsEffnetOnnxBackend):
        backend_identity = DISCOGS_VERSION + ":" + file_sha256(embedding_backend.model_path)
    elif embedding_backend is not None:
        raise ValueError("Checkpointing requires a versioned built-in embedding backend")
    heads = (
        [
            (spec.source_version, file_sha256(feature_head_backend.model_directory / spec.filename))
            for spec in feature_head_backend.specs
        ]
        if feature_head_backend
        else []
    )
    window_dir = Path(window_output_dir).resolve() if window_output_dir is not None else None
    signature = {
        "job_version": 1,
        "objective_config": asdict(active),
        "embedding": backend_identity,
        "embedding_batch_size": getattr(embedding_backend, "batch_size", None),
        "embedding_device": getattr(embedding_backend, "device", "cpu"),
        "heads": heads,
        "window_directory": str(window_dir) if window_dir else None,
    }
    ordered = sorted(assets, key=lambda asset: asset.track_id)
    if not ordered or len({asset.track_id for asset in ordered}) != len(ordered):
        raise ValueError("Audio job needs nonempty, unique track identities")
    observations: list[FeatureObservation] = []
    window_count = score_count = 0
    for number, asset in enumerate(ordered, 1):
        identity = {
            **signature,
            "track_id": asset.track_id,
            "rights_basis": asset.rights_basis,
            "audio_sha256": file_sha256(asset.path),
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        checkpoint = Path(cache_dir) / f"{key}.json"
        window_path = (
            window_dir / (hashlib.sha256(asset.track_id.encode()).hexdigest() + ".npz")
            if window_dir and embedding_backend
            else None
        )
        cached = None
        try:
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if record["key"] == key:
                checksum = record.pop("result_sha256")
                if (
                    checksum
                    != hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                ):
                    raise ValueError("Checkpoint checksum mismatch")
                restored = tuple(
                    FeatureObservation.from_dict(row) for row in record["observations"]
                )
                if not restored or any(row.track_id != asset.track_id for row in restored):
                    raise ValueError("Checkpoint identity mismatch")
                if window_path and (
                    not window_path.is_file() or file_sha256(window_path) != record["window_sha256"]
                ):
                    raise ValueError("Checkpoint window artifact mismatch")
                counts = (record["embedding_windows"], record["learned_scores"])
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in counts
                ):
                    raise ValueError("Checkpoint counts invalid")
                cached = AudioFeaturePipelineResult(restored, 1, *counts)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        result = cached or analyze_audio_assets(
            [asset],
            config=active,
            embedding_backend=embedding_backend,
            feature_head_backend=feature_head_backend,
            window_output_dir=window_dir,
        )
        if cached is None:
            record = {
                "key": key,
                "observations": [row.to_dict() for row in result.observations],
                "embedding_windows": result.embedding_windows,
                "learned_scores": result.learned_scores,
                "window_sha256": file_sha256(window_path) if window_path else None,
            }
            record["result_sha256"] = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()
            atomic_write_text(checkpoint, json.dumps(record, sort_keys=True) + "\n")
        observations.extend(result.observations)
        window_count += result.embedding_windows
        score_count += result.learned_scores
        if progress:
            progress(number, len(ordered), cached is not None)
    return AudioFeaturePipelineResult(tuple(observations), len(ordered), window_count, score_count)
