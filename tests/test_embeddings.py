from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from myusic_engine.audio import DecodedAudio
from myusic_engine.embeddings import (
    EMBEDDING_DIMENSIONS,
    FEATURE_NAME,
    FEATURE_SOURCE,
    SOURCE_VERSION,
    DiscogsEffnetOnnxBackend,
    EmbeddingExtractionError,
    download_model,
    mean_pool_l2_normalize,
    prepare_discogs_effnet_input,
)


def test_window_embeddings_are_mean_pooled_then_l2_normalized() -> None:
    pooled = mean_pool_l2_normalize(
        [(2.0, 0.0), (0.0, 2.0)],
        dimensions=2,
        coverage_seconds=30.0,
    )

    assert pooled.vector == pytest.approx((2**-0.5, 2**-0.5))
    assert pooled.window_count == 2
    assert pooled.confidence == 0.95


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([], "At least one"),
        ([(1.0,)], "dimensions"),
        ([(float("nan"), 1.0)], "finite"),
        ([(0.0, 0.0)], "zero norm"),
    ],
)
def test_invalid_window_embeddings_are_rejected(
    vectors: list[tuple[float, ...]], message: str
) -> None:
    with pytest.raises(EmbeddingExtractionError, match=message):
        mean_pool_l2_normalize(vectors, dimensions=2, coverage_seconds=10.0)


def test_discogs_frontend_produces_overlapping_128_by_96_patches() -> None:
    sample_rate = 16_000
    time = np.arange(4 * sample_rate, dtype=np.float32) / sample_rate
    audio = DecodedAudio(0.2 * np.sin(2 * np.pi * 440 * time), sample_rate)

    prepared = prepare_discogs_effnet_input(audio)

    assert prepared.patches.shape[1:] == (128, 96)
    assert prepared.patches.shape[0] >= 2
    assert prepared.start_frames[:2] == (0, 62)
    assert prepared.patches.dtype == np.float32
    assert np.all(np.isfinite(prepared.patches))


@dataclass
class _Node:
    name: str
    shape: tuple[object, ...]


class _FakeSession:
    def get_inputs(self) -> tuple[_Node, ...]:
        return (_Node("melspectrogram", ("batch", 128, 96)),)

    def get_outputs(self) -> tuple[_Node, ...]:
        return (_Node("activations", ("batch", 400)), _Node("embeddings", ("batch", 1280)))

    def run(
        self, output_names: list[str] | tuple[str, ...], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["embeddings"]
        batch = input_feed["melspectrogram"]
        means = np.mean(batch, axis=(1, 2), keepdims=False)
        return [np.repeat(means[:, None], EMBEDDING_DIMENSIONS, axis=1).astype(np.float32)]


def test_backend_emits_similarity_ready_provenance_record(tmp_path: Path) -> None:
    model_path = tmp_path / "fake.onnx"
    model_path.write_bytes(b"synthetic test model placeholder")
    backend = DiscogsEffnetOnnxBackend(
        model_path,
        verify_model_hash=False,
        session=_FakeSession(),
        batch_size=2,
    )
    sample_rate = 16_000
    time = np.arange(5 * sample_rate, dtype=np.float32) / sample_rate
    audio = DecodedAudio(0.2 * np.sin(2 * np.pi * 440 * time), sample_rate)

    analysis = backend.extract("track-a", audio)

    observation = analysis.observation
    assert observation.feature_name == FEATURE_NAME
    assert observation.feature_source == FEATURE_SOURCE
    assert observation.source_version == SOURCE_VERSION
    assert isinstance(observation.value, tuple)
    assert len(observation.value) == EMBEDDING_DIMENSIONS
    assert np.linalg.norm(observation.value) == pytest.approx(1.0)
    assert analysis.window_count == analysis.window_vectors.shape[0]


def test_model_download_requires_license_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingExtractionError, match="license"):
        download_model(tmp_path / "model.onnx", accept_noncommercial_license=False)
