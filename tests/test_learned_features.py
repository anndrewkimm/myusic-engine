from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from myusic_engine.embeddings import EMBEDDING_DIMENSIONS, EmbeddingAnalysis
from myusic_engine.features.learned import (
    DEFAULT_FEATURE_HEAD_SPECS,
    FEATURE_SOURCE,
    DiscogsEffnetFeatureHeadBackend,
    LearnedFeatureError,
    download_feature_head_models,
)
from myusic_engine.features.records import FeatureObservation


@dataclass
class _Node:
    name: str
    shape: tuple[object, ...]


class _FakeHeadSession:
    def __init__(self, activations: np.ndarray) -> None:
        self.activations = np.asarray(activations, dtype=np.float32)

    def get_inputs(self) -> tuple[_Node, ...]:
        return (_Node("embeddings", ("batch", EMBEDDING_DIMENSIONS)),)

    def get_outputs(self) -> tuple[_Node, ...]:
        return (_Node("activations", ("batch", 2)),)

    def run(
        self, output_names: list[str] | tuple[str, ...], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["activations"]
        assert input_feed["embeddings"].shape[0] == self.activations.shape[0]
        return [self.activations]


def _embedding_analysis(window_count: int = 3) -> EmbeddingAnalysis:
    vectors = np.zeros((window_count, EMBEDDING_DIMENSIONS), dtype=np.float32)
    vectors[:, 0] = 1.0
    observation = FeatureObservation(
        track_id="track-a",
        feature_name="discogs_effnet_embedding_v1",
        value=tuple(float(value) for value in vectors[0]),
        feature_source="mtg_essentia_onnx",
        source_version="test-embedding-v1",
        coverage_seconds=12.0,
        feature_confidence=0.8,
    )
    return EmbeddingAnalysis(
        observation=observation,
        window_vectors=vectors,
        window_start_seconds=tuple(float(index) for index in range(window_count)),
    )


def test_feature_heads_pool_the_named_positive_classes(tmp_path: Path) -> None:
    danceability = DEFAULT_FEATURE_HEAD_SPECS[0]
    relaxed = DEFAULT_FEATURE_HEAD_SPECS[-1]
    for spec in (danceability, relaxed):
        (tmp_path / spec.filename).write_bytes(b"synthetic model placeholder")
    sessions = {
        danceability.task: _FakeHeadSession(np.asarray(((0.9, 0.1), (0.7, 0.3), (0.5, 0.5)))),
        relaxed.task: _FakeHeadSession(np.asarray(((0.8, 0.2), (0.4, 0.6), (0.1, 0.9)))),
    }
    backend = DiscogsEffnetFeatureHeadBackend(
        tmp_path,
        specs=(danceability, relaxed),
        verify_model_hashes=False,
        sessions=sessions,
    )

    analysis = backend.extract(_embedding_analysis())
    observations = {item.feature_name: item for item in analysis.observations}

    assert observations["danceable_score_v1"].value == pytest.approx(0.7)
    assert observations["relaxed_score_v1"].value == pytest.approx((0.2 + 0.6 + 0.9) / 3)
    assert observations["danceable_score_v1"].feature_source == FEATURE_SOURCE
    assert observations["danceable_score_v1"].feature_confidence < 0.8
    assert analysis.feature_names == ("danceable_score_v1", "relaxed_score_v1")
    assert analysis.window_scores.shape == (3, 2)
    assert analysis.window_scores[:, 1] == pytest.approx((0.2, 0.6, 0.9))


def test_feature_heads_reject_outputs_that_are_not_softmax_scores(tmp_path: Path) -> None:
    spec = DEFAULT_FEATURE_HEAD_SPECS[0]
    (tmp_path / spec.filename).write_bytes(b"synthetic model placeholder")
    backend = DiscogsEffnetFeatureHeadBackend(
        tmp_path,
        specs=(spec,),
        verify_model_hashes=False,
        sessions={spec.task: _FakeHeadSession(np.asarray(((0.8, 0.4),) * 3))},
    )

    with pytest.raises(LearnedFeatureError, match="Softmax"):
        backend.extract(_embedding_analysis())


def test_feature_head_download_requires_license_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(LearnedFeatureError, match="license"):
        download_feature_head_models(
            tmp_path,
            accept_noncommercial_license=False,
        )
