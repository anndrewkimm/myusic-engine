"""Reproducible recording-identity checks on permitted real audio."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypedDict

import numpy as np

from myusic_engine.audio import AudioAsset, DecodedAudio, decode_audio
from myusic_engine.embeddings.discogs_effnet import file_sha256
from myusic_engine.features.pipeline import AudioEmbeddingBackend
from myusic_engine.features.records import write_feature_observations
from myusic_engine.io import atomic_write_text


class _Measurement(TypedDict):
    track_id: str
    transform: str
    same_recording_cosine: float
    strongest_other_cosine: float
    correct_unique_top1: bool


def validate_audio_embeddings(
    assets: Iterable[AudioAsset],
    backend: AudioEmbeddingBackend,
    output_dir: str | Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Measure gain and middle-excerpt invariance; this is not a human taste benchmark."""
    ordered = sorted(assets, key=lambda asset: asset.track_id)
    if len(ordered) < 2 or len({asset.track_id for asset in ordered}) != len(ordered):
        raise ValueError("Audio validation needs at least two distinct track identities")
    content_hashes = [file_sha256(asset.path) for asset in ordered]
    if len(set(content_hashes)) != len(content_hashes):
        raise ValueError("Audio validation cannot treat identical input files as different tracks")
    originals = []
    transformed: list[tuple[str, str, tuple[float, ...]]] = []
    for index, asset in enumerate(ordered, 1):
        audio = decode_audio(asset.path, target_sample_rate_hz=48_000)
        original = backend.extract(asset.track_id, audio).observation
        originals.append(original)
        excerpt_samples = min(audio.samples.size, 20 * audio.sample_rate_hz)
        start = (audio.samples.size - excerpt_samples) // 2
        variants = {
            "gain_minus_6db": DecodedAudio(
                audio.samples * np.float32(0.501187), audio.sample_rate_hz
            ),
            "middle_excerpt_up_to_20s": DecodedAudio(
                audio.samples[start : start + excerpt_samples], audio.sample_rate_hz
            ),
        }
        for name, variant in variants.items():
            observation = backend.extract(asset.track_id, variant).observation
            if observation.selector != original.selector or not isinstance(
                observation.value, tuple
            ):
                raise ValueError("Validation backend changed embedding provenance")
            transformed.append((asset.track_id, name, observation.value))
        if progress:
            progress(index, len(ordered))
    matrix = np.asarray([item.value for item in originals], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Validation backend returned invalid vectors")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Validation backend returned zero vectors")
    matrix /= norms
    ids = [asset.track_id for asset in ordered]
    rows: list[_Measurement] = []
    for track_id, transform, vector in transformed:
        query = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm <= 1e-12:
            raise ValueError("Validation transformation returned an invalid vector")
        similarities = np.clip(matrix @ (query / norm), -1, 1)
        expected_index = ids.index(track_id)
        expected_similarity = float(similarities[expected_index])
        others = np.delete(similarities, expected_index)
        # A tie is ambiguous, not evidence of correct identity retrieval.
        correct = bool(expected_similarity > float(others.max()) + 1e-8)
        rows.append(
            {
                "track_id": track_id,
                "transform": transform,
                "same_recording_cosine": expected_similarity,
                "strongest_other_cosine": float(others.max()),
                "correct_unique_top1": correct,
            }
        )
    summary = {}
    for transform in sorted({str(row["transform"]) for row in rows}):
        selected = [row for row in rows if row["transform"] == transform]
        summary[transform] = {
            "queries": len(selected),
            "unique_top1_accuracy": sum(bool(row["correct_unique_top1"]) for row in selected)
            / len(selected),
            "mean_same_recording_cosine": float(
                np.mean([row["same_recording_cosine"] for row in selected])
            ),
            "minimum_same_recording_cosine": min(row["same_recording_cosine"] for row in selected),
        }
    source = originals[0].selector.label
    identity = json.dumps(
        {"inputs": dict(zip(ids, content_hashes, strict=True)), "source": source, "version": 1},
        sort_keys=True,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "recording_identity_invariance_v1",
        "run_id": hashlib.sha256(identity.encode()).hexdigest(),
        "tracks": len(ids),
        "embedding_selector": source,
        "input_sha256": dict(zip(ids, content_hashes, strict=True)),
        "summary": summary,
        "measurements": rows,
        "limitations": [
            "Measures recording identity under gain/excerpt changes; excludes preference quality.",
            "Small-corpus top-1 accuracy does not establish catalog-scale recommendation quality.",
            "No human judgments or personal listening labels are used.",
        ],
    }
    destination = Path(output_dir)
    write_feature_observations(originals, destination / "original_embeddings.jsonl")
    atomic_write_text(
        destination / "audio_validation_report.json", json.dumps(report, indent=2) + "\n"
    )
    return report
