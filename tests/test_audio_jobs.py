from dataclasses import replace

import pytest

from myusic_engine.audio import AudioAsset
from myusic_engine.features import FeatureObservation
from myusic_engine.features.config import ObjectiveFeatureConfig
from myusic_engine.features.jobs import analyze_audio_job
from myusic_engine.features.pipeline import AudioFeaturePipelineResult


def test_audio_job_reuses_only_unchanged_input_and_config(tmp_path, monkeypatch):
    import myusic_engine.features.jobs as jobs

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"synthetic input")
    asset = AudioAsset("synthetic", audio, "owned")
    calls = []

    def analyze(assets, **kwargs):
        calls.append(assets[0].track_id)
        return AudioFeaturePipelineResult(
            (FeatureObservation("synthetic", "synthetic_value_v1", 1.0, "synthetic", "v1", 30, 1),),
            1,
            0,
            0,
        )

    monkeypatch.setattr(jobs, "analyze_audio_assets", analyze)
    cache = tmp_path / "cache"
    first = analyze_audio_job([asset], cache_dir=cache)
    assert analyze_audio_job([asset], cache_dir=cache) == first
    assert len(calls) == 1
    audio.write_bytes(b"changed input")
    analyze_audio_job([asset], cache_dir=cache)
    assert len(calls) == 2
    analyze_audio_job(
        [asset],
        cache_dir=cache,
        config=replace(ObjectiveFeatureConfig(), minimum_coverage_seconds=10),
    )
    assert len(calls) == 3
    for path in cache.glob("*.json"):
        path.write_text("corrupt checkpoint")
    analyze_audio_job([asset], cache_dir=cache)
    assert len(calls) == 4


def test_audio_job_preserves_completed_tracks_after_later_failure(tmp_path, monkeypatch):
    import myusic_engine.features.jobs as jobs

    assets = []
    for track in ("a", "b"):
        path = tmp_path / f"{track}.wav"
        path.write_bytes(track.encode())
        assets.append(AudioAsset(track, path, "owned"))
    calls = []
    fail = True

    def analyze(selected, **kwargs):
        track = selected[0].track_id
        calls.append(track)
        if track == "b" and fail:
            raise ValueError("Synthetic interrupted extraction")
        return AudioFeaturePipelineResult(
            (FeatureObservation(track, "synthetic_value_v1", 1.0, "synthetic", "v1", 30, 1),),
            1,
            0,
            0,
        )

    monkeypatch.setattr(jobs, "analyze_audio_assets", analyze)
    with pytest.raises(ValueError, match="interrupted"):
        analyze_audio_job(assets, cache_dir=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    fail = False
    result = analyze_audio_job(assets, cache_dir=tmp_path / "cache")
    assert calls == ["a", "b", "b"]
    assert result.tracks_analyzed == 2
