"""Versioned learned audio scores built on retained Discogs-EffNet windows."""

from __future__ import annotations

import hashlib
import math
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from myusic_engine.embeddings.discogs_effnet import (
    EMBEDDING_DIMENSIONS,
    EmbeddingAnalysis,
    file_sha256,
)
from myusic_engine.features.records import FeatureObservation
from myusic_engine.io import atomic_write_bytes

MODEL_LICENSE = "CC BY-NC-SA 4.0"
MODEL_BASE_URL = "https://essentia.upf.edu/models/classification-heads"
FEATURE_SOURCE = "mtg_essentia_classifier_onnx"
POOLING_VERSION = "mean-window-softmax-v1"


class LearnedFeatureError(ValueError):
    """Raised when a learned feature model or its output is invalid."""


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
class FeatureHeadSpec:
    """Pinned metadata needed to interpret one binary Essentia classifier head."""

    task: str
    filename: str
    sha256: str
    feature_name: str
    classes: tuple[str, str]
    positive_class: str
    validation_normalized_accuracy: float
    training_items: int

    def __post_init__(self) -> None:
        if self.positive_class not in self.classes:
            raise LearnedFeatureError("positive_class must appear exactly once in classes")
        if self.classes.count(self.positive_class) != 1:
            raise LearnedFeatureError("positive_class must appear exactly once in classes")
        if not 0.0 <= self.validation_normalized_accuracy <= 1.0:
            raise LearnedFeatureError("validation_normalized_accuracy must be in [0, 1]")
        if self.training_items < 1:
            raise LearnedFeatureError("training_items must be positive")

    @property
    def positive_index(self) -> int:
        return self.classes.index(self.positive_class)

    @property
    def url(self) -> str:
        return f"{MODEL_BASE_URL}/{self.task}/{self.filename}"

    @property
    def source_version(self) -> str:
        stem = self.filename.removesuffix(".onnx")
        return f"{stem}+discogs-effnet-bsdynamic-1+{POOLING_VERSION}"


DEFAULT_FEATURE_HEAD_SPECS: tuple[FeatureHeadSpec, ...] = (
    FeatureHeadSpec(
        task="danceability",
        filename="danceability-discogs-effnet-1.onnx",
        sha256="9ce9b8c44f1dd5df5ffc124e5d41d67acf254232c1b90c7e057e079ab7cead73",
        feature_name="danceable_score_v1",
        classes=("danceable", "not_danceable"),
        positive_class="danceable",
        validation_normalized_accuracy=0.97,
        training_items=306,
    ),
    FeatureHeadSpec(
        task="mood_acoustic",
        filename="mood_acoustic-discogs-effnet-1.onnx",
        sha256="56b02abf772b9c1cf528e4d6521d6e09515cedccce47895ee5de8de33fbdf848",
        feature_name="acoustic_score_v1",
        classes=("acoustic", "non_acoustic"),
        positive_class="acoustic",
        validation_normalized_accuracy=0.95,
        training_items=321,
    ),
    FeatureHeadSpec(
        task="voice_instrumental",
        filename="voice_instrumental-discogs-effnet-1.onnx",
        sha256="20155e4c439714b0c45c08644b73c8e12d9dccb173bd4ab9934bf1e5aee837ca",
        feature_name="instrumental_score_v1",
        classes=("instrumental", "voice"),
        positive_class="instrumental",
        validation_normalized_accuracy=0.96,
        training_items=1_000,
    ),
    FeatureHeadSpec(
        task="mood_happy",
        filename="mood_happy-discogs-effnet-1.onnx",
        sha256="0ca322819ef137b4b87e9866bffe7370a630e6f1165184ec106326cef6f81e06",
        feature_name="happy_score_v1",
        classes=("happy", "non_happy"),
        positive_class="happy",
        validation_normalized_accuracy=0.87,
        training_items=302,
    ),
    FeatureHeadSpec(
        task="mood_aggressive",
        filename="mood_aggressive-discogs-effnet-1.onnx",
        sha256="de36550b5d1660791ad732ed6de6ebfdc3e65dcf50b928b2578ddf103dbfb400",
        feature_name="aggressive_score_v1",
        classes=("aggressive", "not_aggressive"),
        positive_class="aggressive",
        validation_normalized_accuracy=0.98,
        training_items=280,
    ),
    FeatureHeadSpec(
        task="mood_relaxed",
        filename="mood_relaxed-discogs-effnet-1.onnx",
        sha256="8ba6515a1e5943a72b3b475e3a25fc7a2ff04142c3eaa6aa0716fca371efdfff",
        feature_name="relaxed_score_v1",
        classes=("non_relaxed", "relaxed"),
        positive_class="relaxed",
        validation_normalized_accuracy=0.91,
        training_items=446,
    ),
)


@dataclass(frozen=True, slots=True)
class LearnedFeatureAnalysis:
    """Track-level scores plus their private per-window values for later calibration."""

    observations: tuple[FeatureObservation, ...]
    feature_names: tuple[str, ...]
    window_scores: NDArray[np.float32]

    @property
    def window_count(self) -> int:
        return int(self.window_scores.shape[0])


def download_feature_head_models(
    destination_directory: str | Path,
    *,
    accept_noncommercial_license: bool,
    timeout_seconds: float = 120.0,
    specs: Sequence[FeatureHeadSpec] = DEFAULT_FEATURE_HEAD_SPECS,
) -> tuple[Path, ...]:
    """Download and verify the pinned official classifier-head pack."""

    if not accept_noncommercial_license:
        raise LearnedFeatureError(
            f"The feature-head models are {MODEL_LICENSE}; explicit license acknowledgement "
            "is required"
        )
    if not specs:
        raise LearnedFeatureError("At least one feature-head model is required")
    destination = Path(destination_directory)
    downloaded: list[Path] = []
    for spec in specs:
        try:
            with urllib.request.urlopen(spec.url, timeout=timeout_seconds) as response:
                content = response.read()
        except OSError as exc:
            raise LearnedFeatureError(
                f"Could not download feature-head model {spec.task}: {exc}"
            ) from exc
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != spec.sha256:
            raise LearnedFeatureError(
                f"Downloaded {spec.filename} SHA-256 mismatch: expected {spec.sha256}, "
                f"got {actual_hash}"
            )
        downloaded.append(atomic_write_bytes(destination / spec.filename, content))
    return tuple(downloaded)


def _binary_score_confidence(
    probabilities: NDArray[np.float32], coverage_confidence: float
) -> float:
    """Combine coverage, per-window margin, and cross-window agreement as a heuristic."""

    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    certainty = max(0.0, 1.0 - float(np.mean(entropy)) / math.log(2.0))
    agreement = max(0.0, 1.0 - min(1.0, 2.0 * float(np.std(clipped))))
    return max(0.0, min(1.0, coverage_confidence * math.sqrt(certainty * agreement)))


class DiscogsEffnetFeatureHeadBackend:
    """Run a pinned pack of binary heads over raw Discogs-EffNet window embeddings."""

    def __init__(
        self,
        model_directory: str | Path,
        *,
        specs: Sequence[FeatureHeadSpec] = DEFAULT_FEATURE_HEAD_SPECS,
        verify_model_hashes: bool = True,
        sessions: Mapping[str, _InferenceSession] | None = None,
    ) -> None:
        if not specs:
            raise LearnedFeatureError("At least one feature-head model is required")
        self.model_directory = Path(model_directory)
        self.specs = tuple(specs)
        loaded_sessions: dict[str, _InferenceSession] = {}
        for spec in self.specs:
            model_path = self.model_directory / spec.filename
            if not model_path.is_file():
                raise LearnedFeatureError(f"Feature-head model is not a file: {model_path}")
            if verify_model_hashes:
                actual_hash = file_sha256(model_path)
                if actual_hash != spec.sha256:
                    raise LearnedFeatureError(
                        f"Feature-head model SHA-256 mismatch for {spec.task}: "
                        f"expected {spec.sha256}, got {actual_hash}"
                    )
            if sessions is None:
                try:
                    import onnxruntime as ort
                except ImportError as exc:  # pragma: no cover - environment-specific
                    raise LearnedFeatureError(
                        "ONNX inference requires the phase3 extra: pip install -e '.[phase3]'"
                    ) from exc
                session = cast(
                    _InferenceSession,
                    ort.InferenceSession(
                        str(model_path),
                        providers=["CPUExecutionProvider"],
                    ),
                )
            else:
                try:
                    session = sessions[spec.task]
                except KeyError as exc:
                    raise LearnedFeatureError(
                        f"No injected inference session for feature-head task {spec.task}"
                    ) from exc
            inputs = {node.name: node for node in session.get_inputs()}
            outputs = {node.name: node for node in session.get_outputs()}
            if "embeddings" not in inputs or "activations" not in outputs:
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} lacks the embeddings/activations interface"
                )
            if not inputs["embeddings"].shape or inputs["embeddings"].shape[-1] != 1280:
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} does not accept 1,280-dimensional embeddings"
                )
            if not outputs["activations"].shape or outputs["activations"].shape[-1] != 2:
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} does not emit two-class activations"
                )
            loaded_sessions[spec.task] = session
        self._sessions = loaded_sessions

    def extract(self, analysis: EmbeddingAnalysis) -> LearnedFeatureAnalysis:
        """Mean window Softmax scores while retaining window-level calibration data."""

        vectors = np.asarray(analysis.window_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] != EMBEDDING_DIMENSIONS:
            raise LearnedFeatureError(
                "Feature heads require a non-empty matrix of 1,280-dimensional window embeddings"
            )
        if not np.all(np.isfinite(vectors)):
            raise LearnedFeatureError("Feature-head input contains non-finite embeddings")
        columns: list[NDArray[np.float32]] = []
        observations: list[FeatureObservation] = []
        embedding_observation = analysis.observation
        for spec in self.specs:
            raw_outputs = self._sessions[spec.task].run(
                ["activations"], {"embeddings": np.ascontiguousarray(vectors)}
            )
            if len(raw_outputs) != 1:
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} returned an unexpected output count"
                )
            activations = np.asarray(raw_outputs[0], dtype=np.float32)
            if activations.shape != (vectors.shape[0], 2):
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} returned an unexpected activation shape"
                )
            if not np.all(np.isfinite(activations)):
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} returned non-finite activations"
                )
            if float(np.min(activations)) < -1e-5 or float(np.max(activations)) > 1.0 + 1e-5:
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} returned values outside [0, 1]"
                )
            row_sums = np.sum(activations, axis=1, dtype=np.float64)
            if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5):
                raise LearnedFeatureError(
                    f"Feature-head model {spec.task} activations are not Softmax scores"
                )
            positive_scores = np.ascontiguousarray(
                np.clip(activations[:, spec.positive_index], 0.0, 1.0),
                dtype=np.float32,
            )
            columns.append(positive_scores)
            observations.append(
                FeatureObservation(
                    track_id=embedding_observation.track_id,
                    feature_name=spec.feature_name,
                    value=float(np.mean(positive_scores, dtype=np.float64)),
                    feature_source=FEATURE_SOURCE,
                    source_version=spec.source_version,
                    coverage_seconds=embedding_observation.coverage_seconds,
                    feature_confidence=_binary_score_confidence(
                        positive_scores,
                        embedding_observation.feature_confidence,
                    ),
                )
            )
        return LearnedFeatureAnalysis(
            observations=tuple(observations),
            feature_names=tuple(spec.feature_name for spec in self.specs),
            window_scores=np.ascontiguousarray(np.column_stack(columns), dtype=np.float32),
        )
