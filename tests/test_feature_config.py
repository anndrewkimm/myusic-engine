from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from myusic_engine.features.config import (
    FeatureConfigError,
    ObjectiveFeatureConfig,
    load_objective_feature_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "features.yaml"


def test_repository_feature_config_loads_versioned_silence_policy() -> None:
    config = load_objective_feature_config(CONFIG_PATH)

    assert config.source_version == "objective-dsp-0.3.0"
    assert config.silence_trim_db == 60.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_sample_rate_hz": 44_100.0}, "target_sample_rate_hz"),
        ({"minimum_coverage_seconds": float("nan")}, "positive and finite"),
        ({"minimum_coverage_seconds": float("inf")}, "positive and finite"),
        ({"silence_trim_db": 19.9}, "between 20 and 120"),
        ({"silence_trim_db": 120.1}, "between 20 and 120"),
    ],
)
def test_objective_config_rejects_unsafe_numeric_controls(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(FeatureConfigError, match=message):
        ObjectiveFeatureConfig(**overrides)  # type: ignore[arg-type]


def test_feature_config_rejects_unknown_extractor_fields(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["extractor"]["silence_trim_typo"] = 60.0
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(FeatureConfigError, match="silence_trim_typo"):
        load_objective_feature_config(path)
