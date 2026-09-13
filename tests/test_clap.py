from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from myusic_engine.audio import DecodedAudio
from myusic_engine.embeddings import EmbeddingExtractionError
from myusic_engine.embeddings.clap import (
    DIMENSIONS,
    SAMPLE_RATE_HZ,
    SELECTOR,
    WINDOW_SAMPLES,
    ClapBackend,
    verify_clap_model,
    window_starts,
)


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class Inputs(dict):
    def to(self, device):
        assert device == "cpu"
        return self


class Processor:
    def __init__(self):
        self.windows = []

    def tokenizer(self, text, **kwargs):
        return {"input_ids": text.split()}

    def __call__(self, *, audio=None, text=None, **kwargs):
        if audio is not None:
            assert kwargs["sampling_rate"] == SAMPLE_RATE_HZ
            assert all(0 < len(window) <= WINDOW_SAMPLES for window in audio)
            self.windows.extend(audio)
            return Inputs(count=len(audio))
        return Inputs(count=len(text))


def backend():
    # Exercise orchestration without downloading a model or requiring torch in core CI.
    instance = object.__new__(ClapBackend)
    instance._processor = Processor()
    instance._torch = SimpleNamespace(inference_mode=nullcontext)
    instance._model = SimpleNamespace(
        get_audio_features=lambda count: SimpleNamespace(
            pooler_output=Tensor(np.ones((count, DIMENSIONS)))
        ),
        get_text_features=lambda count: Tensor(np.ones((count, DIMENSIONS))),
    )
    instance.device = "cpu"
    instance.batch_size = 2
    return instance


@pytest.mark.parametrize("seconds", [0.5, 10, 10.1, 20, 24, 31])
def test_clap_windows_cover_every_sample_without_random_crops(seconds):
    samples = int(seconds * SAMPLE_RATE_HZ)
    covered = np.zeros(samples, dtype=bool)
    for start in window_starts(samples):
        covered[start : start + WINDOW_SAMPLES] = True
    assert covered.all()
    assert len(set(window_starts(samples))) == len(window_starts(samples))


def test_clap_extract_preserves_full_coverage_and_provenance():
    instance = backend()
    audio = DecodedAudio(np.ones(24 * SAMPLE_RATE_HZ), SAMPLE_RATE_HZ)
    result = instance.extract("synthetic", audio)
    assert result.window_count == 3
    assert result.window_start_seconds == (0, 10, 14)
    assert result.observation.coverage_seconds == 24
    assert result.observation.selector == SELECTOR
    assert np.linalg.norm(result.observation.value) == pytest.approx(1)
    assert result.window_vectors.shape == (3, DIMENSIONS)
    assert len(instance._processor.windows) == 3


def test_clap_text_and_invalid_outputs():
    instance = backend()
    assert len(instance.encode_text("Instrumental piano and bass")) == DIMENSIONS
    for text in ("", "word " * 78):
        with pytest.raises(EmbeddingExtractionError):
            instance.encode_text(text)
    for array in (np.zeros((1, DIMENSIONS)), np.ones((1, 3)), np.full((1, DIMENSIONS), np.nan)):
        with pytest.raises(EmbeddingExtractionError):
            instance._vectors(Tensor(array), 1)


def test_clap_refuses_missing_or_changed_local_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(EmbeddingExtractionError, match="checksum"):
        verify_clap_model(tmp_path)


def test_clap_download_reuses_verified_files_and_rejects_corruption(tmp_path, monkeypatch):
    import hashlib
    import io

    import myusic_engine.embeddings.clap as clap

    content = b"synthetic weights"
    monkeypatch.setattr(
        clap, "MODEL_FILES", {"model.bin": (len(content), hashlib.sha256(content).hexdigest())}
    )
    calls = []

    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(clap.urllib.request, "urlopen", download)
    clap.download_clap_model(tmp_path)
    clap.download_clap_model(tmp_path)
    assert len(calls) == 1
    assert clap.MODEL_REVISION in calls[0]
    (tmp_path / "model.bin").write_bytes(b"changed")
    monkeypatch.setattr(clap.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupted"))
    with pytest.raises(EmbeddingExtractionError, match="checksum"):
        clap.download_clap_model(tmp_path)
    assert (tmp_path / "model.bin").read_bytes() == b"changed"
    assert not list(tmp_path.glob("*.download"))
