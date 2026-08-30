from datetime import UTC, datetime

import pytest

from myusic_engine.ranking import (
    FeedbackError,
    append_feedback,
    create_feedback,
    read_feedback,
)


def test_feedback_log_is_atomic_unique_and_round_trips(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    event = create_feedback(
        "run-1",
        "track-1",
        "accepted",
        recorded_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )
    append_feedback(path, event)
    assert read_feedback(path) == (event,)
    with pytest.raises(FeedbackError, match="already exists"):
        append_feedback(path, event)
