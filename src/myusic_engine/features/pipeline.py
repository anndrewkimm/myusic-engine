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
from myusic_engine.features.learned import (
    DiscogsEffnetFeatureHeadBackend,
    LearnedFeatureAnalysis,
    LearnedFeatureError,
)
from myusic_engine.features.objective import ObjectiveFeatureExtractor
from myusic_engine.features.records import FeatureObservation
from myusic_engine.io import atomic_write_bytes


@dataclass(frozen=True, slots=True)
class AudioFeaturePipelineResult:
    """In-memory result and counts for one deterministic local extraction run."""

    observations: tuple[FeatureObservation, ...]
    tracks_analyzed: int
    embedding_windows: int
    learned_scores: int


def _write_window_embeddings(
    analysis: EmbeddingAnalysis,
    learned_analysis: LearnedFeatureAnalysis | None,
    track_id: str,
    output_dir: Path,
) -> Path:
    """Persist private window vectors without using track IDs in file names."""

    safe_name = hashlib.sha256(track_id.encode("utf-8")).hexdigest()
    buffer = io.BytesIO()
    if learned_analysis is None:
        np.savez_compressed(
            buffer,
            track_id=np.asarray(track_id),
            feature_name=np.asarray(analysis.observation.feature_name),
            feature_source=np.asarray(analysis.observation.feature_source),
            source_version=np.asarray(analysis.observation.source_version),
            start_seconds=np.asarray(analysis.window_start_seconds, dtype=np.float32),
            embeddings=analysis.window_vectors,
        )
    else:
        np.savez_compressed(
            buffer,
            track_id=np.asarray(track_id),
            feature_name=np.asarray(analysis.observation.feature_name),
            feature_source=np.asarray(analysis.observation.feature_source),
            source_version=np.asarray(analysis.observation.source_version),
            start_seconds=np.asarray(analysis.window_start_seconds, dtype=np.float32),
            embeddings=analysis.window_vectors,
            learned_feature_names=np.asarray(learned_analysis.feature_names),
            learned_window_scores=learned_analysis.window_scores,
        )
    return atomic_write_bytes(output_dir / f"{safe_name}.npz", buffer.getvalue())


def analyze_audio_assets(
    assets: Iterable[AudioAsset],
    *,
    config: ObjectiveFeatureConfig | None = None,
    embedding_backend: DiscogsEffnetOnnxBackend | None = None,
    feature_head_backend: DiscogsEffnetFeatureHeadBackend | None = None,
    window_output_dir: str | Path | None = None,
) -> AudioFeaturePipelineResult:
    """Decode each asset once, extract objective features, and optionally embed it."""

    extractor = ObjectiveFeatureExtractor(config)
    ordered_assets = sorted(assets, key=lambda asset: asset.track_id)
    observations: list[FeatureObservation] = []
    embedding_windows = 0
    learned_scores = 0
    private_window_dir = Path(window_output_dir) if window_output_dir is not None else None
    if feature_head_backend is not None and embedding_backend is None:
        raise LearnedFeatureError("Learned feature heads require the Discogs-EffNet backend")
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
            learned_analysis = (
                feature_head_backend.extract(embedding_analysis)
                if feature_head_backend is not None
                else None
            )
            if learned_analysis is not None:
                observations.extend(learned_analysis.observations)
                learned_scores += len(learned_analysis.observations)
            if private_window_dir is not None:
                _write_window_embeddings(
                    embedding_analysis,
                    learned_analysis,
                    asset.track_id,
                    private_window_dir,
                )
    return AudioFeaturePipelineResult(
        observations=tuple(observations),
        tracks_analyzed=len(ordered_assets),
        embedding_windows=embedding_windows,
        learned_scores=learned_scores,
    )
