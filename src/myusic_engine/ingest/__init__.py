"""Input readers and normalization pipelines."""

from myusic_engine.ingest.models import (
    IngestionIssue,
    IngestionReport,
    IngestionResult,
    NormalizedListeningEvent,
)
from myusic_engine.ingest.processed import (
    ProcessedHistoryError,
    iter_normalized_events,
    read_normalized_events,
)
from myusic_engine.ingest.spotify_history import (
    HistoryIngestionError,
    HistoryInputError,
    HistoryRecordError,
    load_history,
    normalize_history_record,
    prepare_history,
    write_ingestion_result,
)

__all__ = [
    "HistoryIngestionError",
    "HistoryInputError",
    "HistoryRecordError",
    "IngestionIssue",
    "IngestionReport",
    "IngestionResult",
    "NormalizedListeningEvent",
    "ProcessedHistoryError",
    "load_history",
    "iter_normalized_events",
    "normalize_history_record",
    "prepare_history",
    "read_normalized_events",
    "write_ingestion_result",
]
