"""Pinned, local-only Hugging Face CLAP audio and natural-language embeddings."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from myusic_engine.audio import DecodedAudio, resample_audio
from myusic_engine.embeddings.discogs_effnet import EmbeddingAnalysis
from myusic_engine.embeddings.pooling import (
    EmbeddingExtractionError,
    embedding_observation,
    mean_pool_l2_normalize,
)
from myusic_engine.features import FeatureSelector

MODEL_ID = "laion/clap-htsat-unfused"
MODEL_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
MODEL_LICENSE = "Apache-2.0"
FEATURE_NAME = "clap_audio_embedding_v1"
FEATURE_SOURCE = "huggingface_clap"
SOURCE_VERSION = f"clap-htsat-unfused@{MODEL_REVISION}+10s-tail-v1+mean-l2-v1"
DIMENSIONS = 512
SAMPLE_RATE_HZ = 48_000
WINDOW_SAMPLES = 10 * SAMPLE_RATE_HZ
SELECTOR = FeatureSelector(FEATURE_NAME, FEATURE_SOURCE, SOURCE_VERSION)

# Exact upstream Git blob IDs for config/tokenizer files; SHA-256 for the LFS weights.
# Verifying every input prevents changed local preprocessing from retaining old provenance.
MODEL_FILES = {
    "config.json": (5390, "70624c940dc3d35b36bc9de57ee5a958c86e032a"),
    "merges.txt": (456356, "6636bda4a1fd7a63653dffb22683b8162c8de956"),
    "preprocessor_config.json": (541, "e8636819eeb1b19f1e5b6f572eca1a101144062e"),
    "pytorch_model.bin": (
        614525833,
        "1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb",
    ),
    "special_tokens_map.json": (280, "d5698132694f4f1bcff08fa7d937b1701812598e"),
    "tokenizer.json": (2108746, "99f518e1ee65361b4d772c6f805508dbf30cfd8b"),
    "tokenizer_config.json": (384, "058e2e071e2a76af9dc9a10940053477ce8329d3"),
    "vocab.json": (798293, "4ebe4bb3f3114daf2e4cc349f24873a1175a35d7"),
}


def _verified(path: Path, size: int, expected: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1(usedforsecurity=False)
    if len(expected) == 40:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def verify_clap_model(directory: str | Path) -> Path:
    directory = Path(directory)
    for name, (size, digest) in MODEL_FILES.items():
        if not _verified(directory / name, size, digest):
            raise EmbeddingExtractionError(
                f"CLAP file missing or checksum mismatch: {name}. Run download-clap-model."
            )
    # from_pretrained prefers safetensors when present; don't allow an unverified override.
    if any(
        path.name not in MODEL_FILES
        for path in directory.iterdir()
        if path.suffix in {".safetensors", ".json", ".bin"}
    ):
        raise EmbeddingExtractionError("CLAP directory contains unverified alternate model files")
    return directory


def download_clap_model(
    directory: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    timeout_seconds: float = 120.0,
) -> Path:
    """Stream the pinned public checkpoint; reuse verified files after an interrupted run."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in MODEL_FILES.items():
        target = destination / name
        if _verified(target, size, digest):
            continue
        if progress:
            progress(name)
        descriptor, temporary_name = tempfile.mkstemp(dir=destination, suffix=".download")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                    written = 0
                    while chunk := response.read(1024 * 1024):
                        written += len(chunk)
                        if written > size:
                            raise EmbeddingExtractionError("CLAP download exceeds pinned size")
                        stream.write(chunk)
            if not _verified(temporary, size, digest):
                raise EmbeddingExtractionError(f"CLAP download checksum mismatch: {name}")
            os.replace(temporary, target)
        except OSError as exc:
            raise EmbeddingExtractionError(f"CLAP download failed for {name}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return verify_clap_model(destination)


def window_starts(sample_count: int) -> tuple[int, ...]:
    """Cover the complete recording with 10-second windows and an aligned final window."""
    if sample_count < 1:
        raise EmbeddingExtractionError("CLAP requires nonempty audio")
    if sample_count <= WINDOW_SAMPLES:
        return (0,)
    starts = list(range(0, sample_count - WINDOW_SAMPLES + 1, WINDOW_SAMPLES))
    tail = sample_count - WINDOW_SAMPLES
    if starts[-1] != tail:
        starts.append(tail)
    return tuple(starts)


class ClapBackend:
    """CPU/GPU inference with built-in Transformers code and verified local weights only."""

    def __init__(self, directory: str | Path, *, device: str = "cpu", batch_size: int = 4):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise EmbeddingExtractionError("CLAP batch_size must be a positive integer")
        model_dir = verify_clap_model(directory)
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise EmbeddingExtractionError(
                "Install CLAP dependencies: pip install -e '.[clap]'"
            ) from exc
        if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
            raise EmbeddingExtractionError("Choose cpu or an available cuda device")
        self.device = device
        self.batch_size = batch_size
        self._torch = torch
        try:
            self._processor = ClapProcessor.from_pretrained(str(model_dir), local_files_only=True)
            self._model = (
                cast(Any, ClapModel)
                .from_pretrained(
                    str(model_dir), local_files_only=True, use_safetensors=False, weights_only=True
                )
                .to(device)
                .eval()
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise EmbeddingExtractionError("Could not load the verified local CLAP model") from exc

    @staticmethod
    def _vectors(output: Any, count: int) -> NDArray[np.float32]:
        # Transformers 4 returns a tensor, Transformers 5 returns a pooled ModelOutput.
        tensor = getattr(output, "pooler_output", output)
        vectors = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        if vectors.shape != (count, DIMENSIONS) or not np.isfinite(vectors).all():
            raise EmbeddingExtractionError("CLAP returned invalid embedding dimensions or values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise EmbeddingExtractionError("CLAP returned a zero embedding")
        return np.asarray(vectors / norms, dtype=np.float32)

    def extract(self, track_id: str, audio: DecodedAudio) -> EmbeddingAnalysis:
        audio = (
            resample_audio(audio, SAMPLE_RATE_HZ)
            if audio.sample_rate_hz != SAMPLE_RATE_HZ
            else audio
        )
        starts = window_starts(audio.samples.size)
        batches = []
        for offset in range(0, len(starts), self.batch_size):
            selected = starts[offset : offset + self.batch_size]
            windows = [audio.samples[start : start + WINDOW_SAMPLES].copy() for start in selected]
            # No random long-audio crops: every submitted window is <= 10 seconds.
            inputs = self._processor(
                audio=windows,
                sampling_rate=SAMPLE_RATE_HZ,
                return_tensors="pt",
                padding="repeatpad",
                truncation="rand_trunc",
            ).to(self.device)
            with self._torch.inference_mode():
                batches.append(
                    self._vectors(self._model.get_audio_features(**inputs), len(windows))
                )
        vectors = np.concatenate(batches)
        pooled = mean_pool_l2_normalize(
            vectors, dimensions=DIMENSIONS, coverage_seconds=audio.duration_seconds
        )
        return EmbeddingAnalysis(
            observation=embedding_observation(
                track_id,
                pooled,
                feature_name=FEATURE_NAME,
                feature_source=FEATURE_SOURCE,
                source_version=SOURCE_VERSION,
            ),
            window_vectors=vectors,
            window_start_seconds=tuple(start / SAMPLE_RATE_HZ for start in starts),
        )

    def encode_text(self, text: str) -> tuple[float, ...]:
        """Encode an acoustic description in the same space as the audio embeddings."""
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingExtractionError("Sound description must not be empty")
        # Fail instead of silently discarding the end of a user's description.
        tokens = self._processor.tokenizer(text.strip(), truncation=False)["input_ids"]
        if len(tokens) > 77:
            raise EmbeddingExtractionError("Sound description exceeds CLAP's 77-token limit")
        inputs = self._processor(text=[text.strip()], return_tensors="pt", padding=True).to(
            self.device
        )
        with self._torch.inference_mode():
            vectors = self._vectors(self._model.get_text_features(**inputs), 1)
        return tuple(float(value) for value in vectors[0])
