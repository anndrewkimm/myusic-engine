"""Append-only local explicit feedback for later chronological model updates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias, cast

from myusic_engine.io import atomic_write_text
from myusic_engine.privacy import assert_privacy_safe

FeedbackOutcome: TypeAlias = Literal["accepted", "rejected", "saved", "skipped", "listened"]


class FeedbackError(ValueError):
    """Raised when an explicit recommendation feedback event is malformed."""


@dataclass(frozen=True, slots=True)
class RecommendationFeedback:
    """One user-authored feedback outcome tied to a reproducible recommendation run."""

    feedback_id: str
    recommendation_run_id: str
    track_id: str
    outcome: FeedbackOutcome
    recorded_at: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feedback_id": self.feedback_id,
            "recommendation_run_id": self.recommendation_run_id,
            "track_id": self.track_id,
            "outcome": self.outcome,
            "recorded_at": self.recorded_at,
        }


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise FeedbackError("Feedback timestamp must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def create_feedback(
    recommendation_run_id: str,
    track_id: str,
    outcome: FeedbackOutcome,
    *,
    recorded_at: datetime | None = None,
) -> RecommendationFeedback:
    """Create a deterministic event identity from its run, track, outcome, and time."""

    if not recommendation_run_id.strip() or not track_id.strip():
        raise FeedbackError("Feedback run and track IDs must be non-empty")
    if outcome not in {"accepted", "rejected", "saved", "skipped", "listened"}:
        raise FeedbackError("Feedback outcome is unsupported")
    timestamp = _timestamp(recorded_at or datetime.now(UTC))
    canonical = f"{recommendation_run_id}\u001f{track_id}\u001f{outcome}\u001f{timestamp}"
    return RecommendationFeedback(
        feedback_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        recommendation_run_id=recommendation_run_id,
        track_id=track_id,
        outcome=outcome,
        recorded_at=timestamp,
    )


def read_feedback(path: str | Path) -> tuple[RecommendationFeedback, ...]:
    """Read an existing append-only feedback log."""

    source = Path(path)
    if not source.exists():
        return ()
    events: list[RecommendationFeedback] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeedbackError(f"Feedback line {line_number} is not valid JSON") from exc
            if not isinstance(payload, Mapping):
                raise FeedbackError(f"Feedback line {line_number} must be an object")
            record = cast(Mapping[str, object], payload)
            values = {
                key: record.get(key)
                for key in (
                    "feedback_id",
                    "recommendation_run_id",
                    "track_id",
                    "outcome",
                    "recorded_at",
                )
            }
            if record.get("schema_version") != 1 or not all(
                isinstance(value, str) and value for value in values.values()
            ):
                raise FeedbackError("Feedback record does not match schema version 1")
            outcome = values["outcome"]
            if outcome not in {"accepted", "rejected", "saved", "skipped", "listened"}:
                raise FeedbackError("Feedback record has an unsupported outcome")
            event = RecommendationFeedback(
                feedback_id=cast(str, values["feedback_id"]),
                recommendation_run_id=cast(str, values["recommendation_run_id"]),
                track_id=cast(str, values["track_id"]),
                outcome=cast(FeedbackOutcome, outcome),
                recorded_at=cast(str, values["recorded_at"]),
            )
            if event.feedback_id in seen:
                raise FeedbackError("Feedback log contains a duplicate feedback_id")
            seen.add(event.feedback_id)
            events.append(event)
    return tuple(events)


def append_feedback(path: str | Path, event: RecommendationFeedback) -> Path:
    """Atomically append one unique event without risking a partially written log."""

    existing = read_feedback(path)
    if any(item.feedback_id == event.feedback_id for item in existing):
        raise FeedbackError("Feedback event already exists")
    ordered = tuple(
        sorted((*existing, event), key=lambda item: (item.recorded_at, item.feedback_id))
    )
    lines = []
    for item in ordered:
        record = item.to_dict()
        assert_privacy_safe(record)
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return atomic_write_text(path, "\n".join(lines) + "\n")
