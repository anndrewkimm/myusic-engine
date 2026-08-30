"""Deterministic logistic baselines, audio ablations, and JSON-safe model artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from myusic_engine.evaluation import PredictionMetrics, evaluate_predictions
from myusic_engine.features import FeatureObservation
from myusic_engine.io import atomic_write_text
from myusic_engine.modeling.config import AudioFeatureProfile, TasteModelConfig
from myusic_engine.modeling.representation import ProfiledFeatureCatalog
from myusic_engine.modeling.temporal import BEHAVIOR_FEATURE_NAMES, TemporalTasteSample
from myusic_engine.privacy import assert_privacy_safe

ModelStatus: TypeAlias = Literal["trained", "skipped"]


class TasteTrainingError(ValueError):
    """Raised when taste models cannot be fit or reconstructed safely."""


@dataclass(frozen=True, slots=True)
class LinearTasteModel:
    """Portable standardized logistic model; no pickle or executable payloads."""

    model_id: str
    model_name: str
    model_version: str
    dataset_version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    includes_behavior: bool
    includes_descriptors: bool
    includes_embedding: bool
    profile_name: str | None
    profile_version: str | None
    training_rows: int
    training_positive_rate: float
    training_period_end: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        dimensions = len(self.feature_names)
        if self.schema_version != 1 or dimensions < 1:
            raise TasteTrainingError("Taste model schema or feature list is invalid")
        if not (len(self.means) == len(self.scales) == len(self.coefficients) == dimensions):
            raise TasteTrainingError("Taste model parameter dimensions do not align")
        values = (*self.means, *self.scales, *self.coefficients, self.intercept)
        if any(not math.isfinite(value) for value in values):
            raise TasteTrainingError("Taste model parameters must be finite")
        if any(scale <= 0 for scale in self.scales):
            raise TasteTrainingError("Taste model scales must be positive")
        if self.training_rows < 1 or not 0 <= self.training_positive_rate <= 1:
            raise TasteTrainingError("Taste model training summary is invalid")
        if (self.profile_name is None) != (self.profile_version is None):
            raise TasteTrainingError("Taste model profile name and version must appear together")

    def predict_probability(self, values: Sequence[float]) -> float:
        """Score one already-joined feature row with numerically stable sigmoid."""

        if len(values) != len(self.feature_names):
            raise TasteTrainingError("Taste model input has unexpected dimensions")
        logit = self.intercept
        for value, mean, scale, coefficient in zip(
            values, self.means, self.scales, self.coefficients, strict=True
        ):
            if not math.isfinite(value):
                raise TasteTrainingError("Taste model input must be finite")
            logit += coefficient * ((value - mean) / scale)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)

    def contributions(self, values: Sequence[float]) -> tuple[tuple[str, float], ...]:
        """Return additive log-odds contributions for recommendation explanations."""

        if len(values) != len(self.feature_names):
            raise TasteTrainingError("Taste model input has unexpected dimensions")
        return tuple(
            (name, coefficient * ((value - mean) / scale))
            for name, value, mean, scale, coefficient in zip(
                self.feature_names,
                values,
                self.means,
                self.scales,
                self.coefficients,
                strict=True,
            )
        )

    def to_dict(self, *, include_model_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_names": list(self.feature_names),
            "preprocessing": {
                "kind": "training_only_standard_scaler",
                "means": list(self.means),
                "scales": list(self.scales),
            },
            "linear_model": {
                "kind": "binary_logistic_regression_l2",
                "coefficients": list(self.coefficients),
                "intercept": self.intercept,
            },
            "feature_groups": {
                "behavior": self.includes_behavior,
                "descriptors": self.includes_descriptors,
                "embedding": self.includes_embedding,
            },
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "training_rows": self.training_rows,
            "training_positive_rate": self.training_positive_rate,
            "training_period_end": self.training_period_end,
        }
        if include_model_id:
            record["model_id"] = self.model_id
        return record


@dataclass(frozen=True, slots=True)
class TastePrediction:
    """Private held-out prediction retained for later error analysis."""

    model_id: str
    model_name: str
    sample_id: str
    track_id: str
    split: str
    period_index: int
    label: int
    probability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "sample_id": self.sample_id,
            "track_id": self.track_id,
            "split": self.split,
            "period_index": self.period_index,
            "label": self.label,
            "probability": round(self.probability, 8),
        }


@dataclass(frozen=True, slots=True)
class VariantEvaluation:
    """Outcome of one behavior/audio ablation."""

    model_name: str
    status: ModelStatus
    reason: str | None
    feature_count: int
    cohort: str
    split_rows: dict[str, int]
    validation_metrics: PredictionMetrics | None
    test_metrics: PredictionMetrics | None
    model_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "reason": self.reason,
            "feature_count": self.feature_count,
            "cohort": self.cohort,
            "split_rows": dict(sorted(self.split_rows.items())),
            "validation_metrics": (
                self.validation_metrics.to_dict() if self.validation_metrics is not None else None
            ),
            "test_metrics": (
                self.test_metrics.to_dict() if self.test_metrics is not None else None
            ),
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class TasteTrainingReport:
    """Aggregate, comparison-safe ablation results."""

    dataset_version: str
    model_version: str
    profile_name: str | None
    profile_version: str | None
    samples_seen: int
    unique_tracks_seen: int
    descriptor_tracks: int
    embedding_tracks: int
    fair_cohort_tracks: int
    selected_model_name: str | None
    selected_model_id: str | None
    variants: tuple[VariantEvaluation, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "samples_seen": self.samples_seen,
            "unique_tracks_seen": self.unique_tracks_seen,
            "descriptor_tracks": self.descriptor_tracks,
            "embedding_tracks": self.embedding_tracks,
            "fair_cohort_tracks": self.fair_cohort_tracks,
            "selection_rule": "highest validation NDCG@K, then average precision; test untouched",
            "selected_model_name": self.selected_model_name,
            "selected_model_id": self.selected_model_id,
            "variants": [variant.to_dict() for variant in self.variants],
        }


@dataclass(frozen=True, slots=True)
class TasteTrainingResult:
    models: tuple[LinearTasteModel, ...]
    predictions: tuple[TastePrediction, ...]
    report: TasteTrainingReport


@dataclass(frozen=True, slots=True)
class _VariantSpec:
    name: str
    behavior_indices: tuple[int, ...]
    include_descriptors: bool
    include_embedding: bool
    matched_cohort: bool


def _variant_specs(has_descriptors: bool, has_embedding: bool) -> tuple[_VariantSpec, ...]:
    behavior_all = tuple(range(len(BEHAVIOR_FEATURE_NAMES)))
    repeat_recency = tuple(
        BEHAVIOR_FEATURE_NAMES.index(name)
        for name in ("prior_log_play_count", "prior_recency_score")
    )
    artist = tuple(
        BEHAVIOR_FEATURE_NAMES.index(name)
        for name in (
            "prior_artist_log_play_count",
            "prior_artist_outcome_coverage",
            "prior_artist_positive_rate",
        )
    )
    specs = [
        _VariantSpec("repeat_recency_baseline", repeat_recency, False, False, False),
        _VariantSpec("artist_baseline", artist, False, False, False),
        _VariantSpec("behavior_all", behavior_all, False, False, False),
    ]
    if has_descriptors or has_embedding:
        specs.append(_VariantSpec("behavior_matched", behavior_all, False, False, True))
    if has_descriptors:
        specs.extend(
            (
                _VariantSpec("descriptors_only", (), True, False, True),
                _VariantSpec("behavior_descriptors", behavior_all, True, False, True),
            )
        )
    if has_embedding:
        specs.extend(
            (
                _VariantSpec("embedding_only", (), False, True, True),
                _VariantSpec("behavior_embedding", behavior_all, False, True, True),
            )
        )
    if has_descriptors and has_embedding:
        specs.extend(
            (
                _VariantSpec("audio_full", (), True, True, True),
                _VariantSpec("full_combined", behavior_all, True, True, True),
            )
        )
    return tuple(specs)


def _feature_names(spec: _VariantSpec, catalog: ProfiledFeatureCatalog | None) -> tuple[str, ...]:
    names = tuple(f"behavior:{BEHAVIOR_FEATURE_NAMES[index]}" for index in spec.behavior_indices)
    if spec.include_descriptors:
        assert catalog is not None
        names += tuple(f"descriptor:{name}" for name in catalog.descriptor_feature_names)
    if spec.include_embedding:
        assert catalog is not None
        names += tuple(f"embedding:{name}" for name in catalog.embedding_feature_names)
    return names


def _row(
    sample: TemporalTasteSample,
    spec: _VariantSpec,
    catalog: ProfiledFeatureCatalog | None,
) -> tuple[float, ...] | None:
    values = [sample.behavior_features[index] for index in spec.behavior_indices]
    if spec.include_descriptors or spec.include_embedding:
        assert catalog is not None
        audio = catalog.vector(
            sample.track_id,
            include_descriptors=spec.include_descriptors,
            include_embedding=spec.include_embedding,
        )
        if audio is None:
            return None
        values.extend(audio)
    return tuple(values)


def model_input_vector(
    model: LinearTasteModel,
    behavior_features: Sequence[float],
    audio_catalog: ProfiledFeatureCatalog | None,
    track_id: str,
) -> tuple[float, ...] | None:
    """Reconstruct a model row for current-time recommendation inference."""

    if len(behavior_features) != len(BEHAVIOR_FEATURE_NAMES):
        raise TasteTrainingError("Current behavior snapshot has unexpected dimensions")
    values: list[float] = []
    for feature_name in model.feature_names:
        if feature_name.startswith("behavior:"):
            behavior_name = feature_name.removeprefix("behavior:")
            try:
                index = BEHAVIOR_FEATURE_NAMES.index(behavior_name)
            except ValueError as exc:
                raise TasteTrainingError("Model references an unknown behavior feature") from exc
            values.append(float(behavior_features[index]))
    if model.includes_descriptors or model.includes_embedding:
        if audio_catalog is None:
            return None
        audio = audio_catalog.vector(
            track_id,
            include_descriptors=model.includes_descriptors,
            include_embedding=model.includes_embedding,
        )
        if audio is None:
            return None
        values.extend(audio)
    if len(values) != len(model.feature_names):
        raise TasteTrainingError("Model input reconstruction did not match artifact features")
    return tuple(values)


def _model_id(record: Mapping[str, object]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fit_variant(
    spec: _VariantSpec,
    samples: tuple[TemporalTasteSample, ...],
    catalog: ProfiledFeatureCatalog | None,
    profile_name: str | None,
    profile_version: str | None,
    config: TasteModelConfig,
) -> tuple[LinearTasteModel | None, tuple[TastePrediction, ...], VariantEvaluation]:
    fair_tracks = catalog.fair_cohort_tracks if catalog is not None else frozenset()
    selected: list[tuple[TemporalTasteSample, tuple[float, ...]]] = []
    for sample in samples:
        if spec.matched_cohort and sample.track_id not in fair_tracks:
            continue
        values = _row(sample, spec, catalog)
        if values is not None:
            selected.append((sample, values))
    counts: Counter[str] = Counter(sample.split for sample, _ in selected)
    split_rows = {split: counts[split] for split in ("train", "validation", "test")}
    feature_names = _feature_names(spec, catalog)
    train_rows = [(sample, row) for sample, row in selected if sample.split == "train"]
    validation_rows = [(sample, row) for sample, row in selected if sample.split == "validation"]
    test_rows = [(sample, row) for sample, row in selected if sample.split == "test"]
    reason: str | None = None
    if not feature_names:
        reason = "variant has no input features"
    elif not train_rows or not validation_rows or not test_rows:
        reason = "one or more chronological splits have no covered samples"
    elif len({sample.label for sample, _ in train_rows}) < 2:
        reason = "training split does not contain both labels"
    if reason is not None:
        evaluation = VariantEvaluation(
            model_name=spec.name,
            status="skipped",
            reason=reason,
            feature_count=len(feature_names),
            cohort="audio_matched" if spec.matched_cohort else "all_labeled",
            split_rows=split_rows,
            validation_metrics=None,
            test_metrics=None,
            model_id=None,
        )
        return None, (), evaluation

    train_x = np.asarray([row for _, row in train_rows], dtype=np.float64)
    train_y = np.asarray([sample.label for sample, _ in train_rows], dtype=np.int64)
    train_weight = np.asarray([sample.sample_weight for sample, _ in train_rows], dtype=np.float64)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x, sample_weight=train_weight)
    estimator = LogisticRegression(
        C=config.regularization_c,
        max_iter=config.maximum_iterations,
        random_state=config.random_seed,
        solver="liblinear",
    )
    estimator.fit(scaled_train, train_y, sample_weight=train_weight)
    means = tuple(float(value) for value in scaler.mean_)
    scales = tuple(float(value) for value in scaler.scale_)
    coefficients = tuple(float(value) for value in estimator.coef_[0])
    intercept = float(estimator.intercept_[0])
    provisional = LinearTasteModel(
        model_id="pending",
        model_name=spec.name,
        model_version=config.model_version,
        dataset_version=train_rows[0][0].dataset_version,
        feature_names=feature_names,
        means=means,
        scales=scales,
        coefficients=coefficients,
        intercept=intercept,
        includes_behavior=bool(spec.behavior_indices),
        includes_descriptors=spec.include_descriptors,
        includes_embedding=spec.include_embedding,
        profile_name=(profile_name if spec.include_descriptors or spec.include_embedding else None),
        profile_version=(
            profile_version if spec.include_descriptors or spec.include_embedding else None
        ),
        training_rows=len(train_rows),
        training_positive_rate=round(float(np.mean(train_y)), 8),
        training_period_end=max(sample.period_end for sample, _ in train_rows),
    )
    identifier = _model_id(provisional.to_dict(include_model_id=False))
    model = replace(provisional, model_id=identifier)

    predictions: list[TastePrediction] = []

    def evaluate(rows: list[tuple[TemporalTasteSample, tuple[float, ...]]]) -> PredictionMetrics:
        probabilities = [model.predict_probability(row) for _, row in rows]
        for (sample, _), probability in zip(rows, probabilities, strict=True):
            predictions.append(
                TastePrediction(
                    model_id=model.model_id,
                    model_name=model.model_name,
                    sample_id=sample.sample_id,
                    track_id=sample.track_id,
                    split=sample.split,
                    period_index=sample.period_index,
                    label=sample.label,
                    probability=probability,
                )
            )
        return evaluate_predictions(
            [sample.label for sample, _ in rows],
            probabilities,
            [sample.period_index for sample, _ in rows],
            ranking_k=config.ranking_k,
        )

    validation_metrics = evaluate(validation_rows)
    test_metrics = evaluate(test_rows)
    evaluation = VariantEvaluation(
        model_name=spec.name,
        status="trained",
        reason=None,
        feature_count=len(feature_names),
        cohort="audio_matched" if spec.matched_cohort else "all_labeled",
        split_rows=split_rows,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        model_id=model.model_id,
    )
    return model, tuple(predictions), evaluation


def _selection_key(evaluation: VariantEvaluation) -> tuple[float, float, str]:
    metrics = evaluation.validation_metrics
    if metrics is None:
        return (-math.inf, -math.inf, evaluation.model_name)
    ndcg = metrics.ndcg_at_k if metrics.ndcg_at_k is not None else -math.inf
    average_precision = (
        metrics.average_precision if metrics.average_precision is not None else -math.inf
    )
    return (ndcg, average_precision, evaluation.model_name)


def train_taste_models(
    samples: Iterable[TemporalTasteSample],
    *,
    config: TasteModelConfig | None = None,
    feature_observations: Iterable[FeatureObservation] = (),
    profile: AudioFeatureProfile | None = None,
    profile_name: str | None = None,
) -> TasteTrainingResult:
    """Train behavior baselines and fair-cohort audio ablations on whole time splits."""

    ordered_samples = tuple(sorted(samples, key=lambda item: (item.period_index, item.track_id)))
    if not ordered_samples:
        raise TasteTrainingError("Taste training needs temporal samples")
    dataset_versions = {sample.dataset_version for sample in ordered_samples}
    if len(dataset_versions) != 1:
        raise TasteTrainingError("Taste training cannot mix dataset versions")
    active = config or TasteModelConfig()
    if (profile is None) != (profile_name is None):
        raise TasteTrainingError("Audio profile and profile_name must be supplied together")
    catalog = ProfiledFeatureCatalog(feature_observations, profile) if profile is not None else None
    specs = _variant_specs(
        bool(profile and profile.descriptor_inputs),
        bool(profile and profile.embedding_input),
    )
    models: list[LinearTasteModel] = []
    predictions: list[TastePrediction] = []
    evaluations: list[VariantEvaluation] = []
    for spec in specs:
        fitted = _fit_variant(
            spec,
            ordered_samples,
            catalog,
            profile_name,
            profile.profile_version if profile is not None else None,
            active,
        )
        model, variant_predictions, evaluation = fitted
        if model is not None:
            models.append(model)
            predictions.extend(variant_predictions)
        evaluations.append(evaluation)
    if not models:
        raise TasteTrainingError("No model variant had two training labels and held-out rows")
    trained_evaluations = [
        evaluation for evaluation in evaluations if evaluation.status == "trained"
    ]
    selected_evaluation = max(trained_evaluations, key=_selection_key)
    selected_model = next(
        model for model in models if model.model_id == selected_evaluation.model_id
    )
    report = TasteTrainingReport(
        dataset_version=next(iter(dataset_versions)),
        model_version=active.model_version,
        profile_name=profile_name,
        profile_version=profile.profile_version if profile is not None else None,
        samples_seen=len(ordered_samples),
        unique_tracks_seen=len({sample.track_id for sample in ordered_samples}),
        descriptor_tracks=len(catalog.descriptor_tracks) if catalog is not None else 0,
        embedding_tracks=len(catalog.embedding_tracks) if catalog is not None else 0,
        fair_cohort_tracks=len(catalog.fair_cohort_tracks) if catalog is not None else 0,
        selected_model_name=selected_model.model_name,
        selected_model_id=selected_model.model_id,
        variants=tuple(evaluations),
    )
    assert_privacy_safe(report.to_dict())
    return TasteTrainingResult(models=tuple(models), predictions=tuple(predictions), report=report)


def _float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise TasteTrainingError(f"{label} must be a non-empty array")
    result: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TasteTrainingError(f"{label} must contain numbers")
        result.append(float(raw))
    return tuple(result)


def read_taste_model(path: str | Path) -> LinearTasteModel:
    """Load a non-executable JSON model artifact and verify its content hash."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TasteTrainingError("Taste model artifact is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TasteTrainingError("Taste model artifact must be an object")
    record = cast(Mapping[str, object], payload)
    preprocessing = record.get("preprocessing")
    linear = record.get("linear_model")
    groups = record.get("feature_groups")
    if not isinstance(preprocessing, Mapping) or not isinstance(linear, Mapping):
        raise TasteTrainingError("Taste model artifact parameters are missing")
    if not isinstance(groups, Mapping):
        raise TasteTrainingError("Taste model artifact feature groups are missing")
    feature_names_value = record.get("feature_names")
    if not isinstance(feature_names_value, list) or not all(
        isinstance(item, str) and item for item in feature_names_value
    ):
        raise TasteTrainingError("Taste model feature_names must be text")
    profile_name = record.get("profile_name")
    profile_version = record.get("profile_version")
    if profile_name is not None and not isinstance(profile_name, str):
        raise TasteTrainingError("Taste model profile_name must be null or text")
    if profile_version is not None and not isinstance(profile_version, str):
        raise TasteTrainingError("Taste model profile_version must be null or text")

    def required_text(key: str) -> str:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise TasteTrainingError(f"Taste model field {key} must be text")
        return value

    def required_bool(key: str) -> bool:
        value = groups.get(key)
        if not isinstance(value, bool):
            raise TasteTrainingError(f"Taste model feature group {key} must be boolean")
        return value

    raw_intercept = linear.get("intercept")
    raw_rows = record.get("training_rows")
    raw_rate = record.get("training_positive_rate")
    if isinstance(raw_intercept, bool) or not isinstance(raw_intercept, (int, float)):
        raise TasteTrainingError("Taste model intercept must be numeric")
    if isinstance(raw_rows, bool) or not isinstance(raw_rows, int):
        raise TasteTrainingError("Taste model training_rows must be an integer")
    if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)):
        raise TasteTrainingError("Taste model training_positive_rate must be numeric")
    model = LinearTasteModel(
        model_id=required_text("model_id"),
        model_name=required_text("model_name"),
        model_version=required_text("model_version"),
        dataset_version=required_text("dataset_version"),
        feature_names=tuple(cast(list[str], feature_names_value)),
        means=_float_tuple(preprocessing.get("means"), "preprocessing.means"),
        scales=_float_tuple(preprocessing.get("scales"), "preprocessing.scales"),
        coefficients=_float_tuple(linear.get("coefficients"), "linear_model.coefficients"),
        intercept=float(raw_intercept),
        includes_behavior=required_bool("behavior"),
        includes_descriptors=required_bool("descriptors"),
        includes_embedding=required_bool("embedding"),
        profile_name=profile_name,
        profile_version=profile_version,
        training_rows=raw_rows,
        training_positive_rate=float(raw_rate),
        training_period_end=required_text("training_period_end"),
        schema_version=cast(int, record.get("schema_version")),
    )
    expected = _model_id(model.to_dict(include_model_id=False))
    if model.model_id != expected:
        raise TasteTrainingError("Taste model artifact content hash does not match model_id")
    return model


def write_taste_training_result(
    result: TasteTrainingResult, output_dir: str | Path
) -> tuple[tuple[Path, ...], Path, Path, Path]:
    """Write portable artifacts, held-out predictions, report, and selected model."""

    destination = Path(output_dir)
    model_paths: list[Path] = []
    for model in result.models:
        record = model.to_dict()
        assert_privacy_safe(record)
        model_paths.append(
            atomic_write_text(
                destination / "models" / f"{model.model_name}.json",
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        )
    selected_id = result.report.selected_model_id
    selected = next(model for model in result.models if model.model_id == selected_id)
    selected_path = atomic_write_text(
        destination / "selected_model.json",
        json.dumps(selected.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    prediction_lines = []
    for prediction in result.predictions:
        record = prediction.to_dict()
        assert_privacy_safe(record)
        prediction_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    predictions_path = atomic_write_text(
        destination / "heldout_predictions.jsonl",
        "\n".join(prediction_lines) + ("\n" if prediction_lines else ""),
    )
    report_record = result.report.to_dict()
    assert_privacy_safe(report_record)
    report_path = atomic_write_text(
        destination / "taste_model_report.json",
        json.dumps(report_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return tuple(model_paths), selected_path, predictions_path, report_path
