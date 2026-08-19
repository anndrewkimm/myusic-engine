"""Cross-platform ONNX inference for MTG's Discogs-EffNet music embeddings."""

from __future__ import annotations

import hashlib
import math
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from myusic_engine.audio import DecodedAudio, resample_audio
from myusic_engine.audio._dependencies import load_librosa
from myusic_engine.embeddings.pooling import (
    EmbeddingExtractionError,
    embedding_observation,
    mean_pool_l2_normalize,
)
from myusic_engine.features.records import FeatureObservation
from myusic_engine.io import atomic_write_bytes

MODEL_FILENAME = "discogs-effnet-bsdynamic-1.onnx"
MODEL_URL = f"https://essentia.upf.edu/models/feature-extractors/discogs-effnet/{MODEL_FILENAME}"
MODEL_SHA256 = "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c"
MODEL_LICENSE = "CC BY-NC-SA 4.0"
FEATURE_NAME = "discogs_effnet_embedding_v1"
FEATURE_SOURCE = "mtg_essentia_onnx"
SOURCE_VERSION = "discogs-effnet-bsdynamic-1+musicnn-preprocess-v1+mean-l2-v1"
SAMPLE_RATE_HZ = 16_000
FRAME_LENGTH = 512
HOP_LENGTH = 256
MEL_BANDS = 96
PATCH_FRAMES = 128
PATCH_HOP_FRAMES = 62
EMBEDDING_DIMENSIONS = 1280


class _ModelNode(Protocol):
    name: str
    shape: Sequence[object]


class _InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_ModelNode]: ...

    def get_outputs(self) -> Sequence[_ModelNode]: ...

    def run(
        self, output_names: Sequence[str], input_feed: dict[str, NDArray[np.float32]]
    ) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class PreparedDiscogsEffnetInput:
    """Exact log-mel patches and their first mel-frame positions."""

    patches: NDArray[np.float32]
    start_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingAnalysis:
    """Track observation plus retained window vectors for diagnostics or later pooling."""

    observation: FeatureObservation
    window_vectors: NDArray[np.float32]
    window_start_seconds: tuple[float, ...]

    @property
    def window_count(self) -> int:
        return int(self.window_vectors.shape[0])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    destination: str | Path,
    *,
    accept_noncommercial_license: bool,
    timeout_seconds: float = 120.0,
) -> Path:
    """Download the pinned official model only after explicit license acknowledgement."""

    if not accept_noncommercial_license:
        raise EmbeddingExtractionError(
            f"The model is {MODEL_LICENSE}; explicit license acknowledgement is required"
        )
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=timeout_seconds) as response:
            content = response.read()
    except OSError as exc:
        raise EmbeddingExtractionError(f"Could not download embedding model: {exc}") from exc
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != MODEL_SHA256:
        raise EmbeddingExtractionError(
            f"Downloaded model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_hash}"
        )
    return atomic_write_bytes(destination, content)


def prepare_discogs_effnet_input(audio: DecodedAudio) -> PreparedDiscogsEffnetInput:
    """Reproduce Essentia's MusiCNN log-mel frontend and overlapping model patches."""

    if audio.sample_rate_hz != SAMPLE_RATE_HZ:
        audio = resample_audio(audio, SAMPLE_RATE_HZ)
    librosa = load_librosa()
    spectrum = np.abs(
        librosa.stft(
            audio.samples,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=FRAME_LENGTH,
            window="hann",
            center=True,
            pad_mode="constant",
        )
    ).astype(np.float32, copy=False)
    mel_filter = librosa.filters.mel(
        sr=SAMPLE_RATE_HZ,
        n_fft=FRAME_LENGTH,
        n_mels=MEL_BANDS,
        fmin=0.0,
        fmax=SAMPLE_RATE_HZ / 2,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )
    mel = mel_filter @ spectrum
    log_mel = np.log10(1.0 + 10_000.0 * mel).T.astype(np.float32, copy=False)
    if log_mel.shape[0] < PATCH_FRAMES:
        minimum_seconds = ((PATCH_FRAMES - 1) * HOP_LENGTH + FRAME_LENGTH / 2) / SAMPLE_RATE_HZ
        raise EmbeddingExtractionError(
            f"Discogs-EffNet needs at least {minimum_seconds:.2f} seconds of audio"
        )
    starts = tuple(range(0, log_mel.shape[0] - PATCH_FRAMES + 1, PATCH_HOP_FRAMES))
    patches = np.stack([log_mel[start : start + PATCH_FRAMES] for start in starts])
    if patches.shape[1:] != (PATCH_FRAMES, MEL_BANDS):
        raise EmbeddingExtractionError("Discogs-EffNet preprocessing produced a wrong shape")
    if not np.all(np.isfinite(patches)):
        raise EmbeddingExtractionError("Discogs-EffNet input contains non-finite values")
    return PreparedDiscogsEffnetInput(
        patches=np.ascontiguousarray(patches, dtype=np.float32),
        start_frames=starts,
    )


class DiscogsEffnetOnnxBackend:
    """Pinned 1,280-dimensional Discogs-style music embedding extractor."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = 64,
        verify_model_hash: bool = True,
        session: _InferenceSession | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise EmbeddingExtractionError("batch_size must be a positive integer")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise EmbeddingExtractionError(f"Embedding model is not a file: {self.model_path}")
        if verify_model_hash:
            actual_hash = file_sha256(self.model_path)
            if actual_hash != MODEL_SHA256:
                raise EmbeddingExtractionError(
                    f"Embedding model SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_hash}"
                )
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise EmbeddingExtractionError(
                    "ONNX inference requires the phase3 extra: pip install -e '.[phase3]'"
                ) from exc
            session = cast(
                _InferenceSession,
                ort.InferenceSession(
                    str(self.model_path),
                    providers=["CPUExecutionProvider"],
                ),
            )
        self._session = session
        self.batch_size = batch_size
        input_names = {node.name for node in session.get_inputs()}
        output_names = {node.name for node in session.get_outputs()}
        if "melspectrogram" not in input_names or "embeddings" not in output_names:
            raise EmbeddingExtractionError(
                "ONNX model does not expose the pinned melspectrogram/embeddings interface"
            )

    def extract(self, track_id: str, audio: DecodedAudio) -> EmbeddingAnalysis:
        """Infer window embeddings, retain them, and emit a normalized track record."""

        if audio.sample_rate_hz != SAMPLE_RATE_HZ:
            audio = resample_audio(audio, SAMPLE_RATE_HZ)
        prepared = prepare_discogs_effnet_input(audio)
        output_batches: list[NDArray[np.float32]] = []
        for offset in range(0, prepared.patches.shape[0], self.batch_size):
            batch = prepared.patches[offset : offset + self.batch_size]
            raw_outputs = self._session.run(["embeddings"], {"melspectrogram": batch})
            if len(raw_outputs) != 1:
                raise EmbeddingExtractionError("ONNX runtime returned an unexpected output count")
            output = np.asarray(raw_outputs[0], dtype=np.float32)
            if output.shape != (batch.shape[0], EMBEDDING_DIMENSIONS):
                raise EmbeddingExtractionError("ONNX model returned an unexpected embedding shape")
            if not np.all(np.isfinite(output)):
                raise EmbeddingExtractionError("ONNX model returned non-finite embeddings")
            output_batches.append(output)
        window_vectors = np.ascontiguousarray(np.concatenate(output_batches, axis=0))
        last_start = prepared.start_frames[-1]
        covered_samples = min(
            audio.samples.size,
            (last_start + PATCH_FRAMES - 1) * HOP_LENGTH + FRAME_LENGTH // 2,
        )
        coverage_seconds = covered_samples / SAMPLE_RATE_HZ
        pooled = mean_pool_l2_normalize(
            window_vectors,
            dimensions=EMBEDDING_DIMENSIONS,
            coverage_seconds=coverage_seconds,
        )
        observation = embedding_observation(
            track_id,
            pooled,
            feature_name=FEATURE_NAME,
            feature_source=FEATURE_SOURCE,
            source_version=SOURCE_VERSION,
        )
        start_seconds = tuple(
            start * HOP_LENGTH / SAMPLE_RATE_HZ for start in prepared.start_frames
        )
        if not math.isclose(
            math.sqrt(math.fsum(value * value for value in pooled.vector)),
            1.0,
            rel_tol=1e-6,
        ):
            raise EmbeddingExtractionError("Pooled embedding was not normalized")
        return EmbeddingAnalysis(
            observation=observation,
            window_vectors=window_vectors,
            window_start_seconds=start_seconds,
        )
