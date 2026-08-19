"""Orchestration from permitted local audio to phase-3 feature observations."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from myusic_engine.audio import AudioAsset, decode_audio
from myusic_engine.embeddings import DiscogsEffnetOnnxBackend, EmbeddingAnalysis
from myusic_engine.features.config import ObjectiveFeatureConfig
from myusic_engine.features.objective import ObjectiveFeatureExtractor
from myusic_engine.features.records import FeatureObservation
from myusic_engine.io import atomic_write_bytes


@dataclass(frozen=True, slots=True)
class AudioFeaturePipelineResult:
    """In-memory result and counts for one deterministic local extraction run."""

    observations: tuple[FeatureObservation, ...]
    tracks_analyzed: int
    embedding_windows: int


def _write_window_embeddings(analysis: EmbeddingAnalysis, track_id: str, output_dir: Path) -> Path:
    """Persist private window vectors without using track IDs in file names."""

    safe_name = hashlib.sha256(track_id.encode("utf-8")).hexdigest()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        track_id=np.asarray(track_id),
        feature_name=np.asarray(analysis.observation.feature_name),
        feature_source=np.asarray(analysis.observation.feature_source),
        source_version=np.asarray(analysis.observation.source_version),
        start_seconds=np.asarray(analysis.window_start_seconds, dtype=np.float32),
        embeddings=analysis.window_vectors,
    )
    return atomic_write_bytes(output_dir / f"{safe_name}.npz", buffer.getvalue())


def analyze_audio_assets(
    assets: Iterable[AudioAsset],
    *,
    config: ObjectiveFeatureConfig | None = None,
    embedding_backend: DiscogsEffnetOnnxBackend | None = None,
    window_output_dir: str | Path | None = None,
) -> AudioFeaturePipelineResult:
    """Decode each asset once, extract objective features, and optionally embed it."""

    extractor = ObjectiveFeatureExtractor(config)
    ordered_assets = sorted(assets, key=lambda asset: asset.track_id)
    observations: list[FeatureObservation] = []
    embedding_windows = 0
    private_window_dir = Path(window_output_dir) if window_output_dir is not None else None
    for asset in ordered_assets:
        audio = decode_audio(
            asset.path,
            target_sample_rate_hz=extractor.config.target_sample_rate_hz,
        )
        observations.extend(extractor.extract(asset.track_id, audio))
        if embedding_backend is not None:
            embedding_analysis = embedding_backend.extract(asset.track_id, audio)
            observations.append(embedding_analysis.observation)
            embedding_windows += embedding_analysis.window_count
            if private_window_dir is not None:
                _write_window_embeddings(
                    embedding_analysis,
                    asset.track_id,
                    private_window_dir,
                )
    return AudioFeaturePipelineResult(
        observations=tuple(observations),
        tracks_analyzed=len(ordered_assets),
        embedding_windows=embedding_windows,
    )
