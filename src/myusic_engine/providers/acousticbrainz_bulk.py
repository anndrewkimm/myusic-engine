"""Offline access to the frozen 2022 AcousticBrainz CC0 low-level CSV dumps.

AcousticBrainz stopped accepting submissions in 2022 but published its low-level dataset as
three derived CSV dumps (lowlevel, rhythm, tonal), each keyed by recording MBID and submission
offset. Scanning these local files never sends a recording MBID over the network, unlike the
live ``AcousticBrainzClient``.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from myusic_engine.bulk_dump import open_dump_csv_lines
from myusic_engine.providers.acousticbrainz import AcousticBrainzDocument

_MBID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_TEXT_FIELDS = frozenset({"key_key", "key_scale"})
_EXPECTED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "lowlevel": (
        "mbid",
        "submission_offset",
        "average_loudness",
        "dynamic_complexity",
        "mfcc_zero_mean",
    ),
    "rhythm": (
        "mbid",
        "submission_offset",
        "bpm",
        "bpm_histogram_first_peak_bpm_mean",
        "bpm_histogram_first_peak_bpm_median",
        "bpm_histogram_second_peak_bpm_mean",
        "bpm_histogram_second_peak_bpm_median",
        "danceability",
        "onset_rate",
    ),
    "tonal": (
        "mbid",
        "submission_offset",
        "key_key",
        "key_scale",
        "tuning_frequency",
        "tuning_equal_tempered_deviation",
    ),
}


class AcousticBrainzBulkError(ValueError):
    """Raised when a local AcousticBrainz bulk dump cannot be safely streamed or matched."""


@dataclass(frozen=True, slots=True)
class BulkDumpScanReport:
    """Aggregate evidence for one local multi-file AcousticBrainz bulk scan."""

    sections_scanned: tuple[str, ...]
    target_mbids: int
    rows_scanned: int
    mbids_covered: int


@dataclass(frozen=True, slots=True)
class OfflineAcousticBrainzProvider:
    """In-memory ``AcousticBrainzProvider`` populated from local bulk dumps only."""

    documents: Mapping[str, AcousticBrainzDocument]

    def fetch(self, recording_mbids: Sequence[str]) -> Mapping[str, AcousticBrainzDocument]:
        normalized = tuple(_validate_mbid(mbid) for mbid in recording_mbids)
        return {mbid: self.documents[mbid] for mbid in normalized if mbid in self.documents}


def _validate_mbid(value: str) -> str:
    if not isinstance(value, str) or _MBID_PATTERN.fullmatch(value) is None:
        raise AcousticBrainzBulkError("Offline AcousticBrainz matching requires a valid MBID")
    return value.casefold()


def _member_name(source: Path) -> str:
    name = source.name
    for suffix in (".tar.zst", ".tar", ".csv"):
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)] + ".csv"
    raise AcousticBrainzBulkError(f"{name} must be a CSV, tar, or tar.zst dump")


def _parse_value(raw: str, field_name: str, *, source_name: str, row_number: int) -> object | None:
    if raw == "":
        return None
    if field_name in _TEXT_FIELDS:
        return raw
    try:
        return float(raw)
    except ValueError as exc:
        raise AcousticBrainzBulkError(
            f"{source_name} row {row_number} has an invalid {field_name} value"
        ) from exc


def _scan_section(
    section: str,
    source: Path,
    target_mbids: frozenset[str],
    nested_by_mbid: dict[str, dict[str, dict[str, object]]],
    *,
    progress: Callable[[str, int], None] | None,
    progress_interval: int,
) -> int:
    expected_header = _EXPECTED_COLUMNS[section]
    best_offset: dict[str, int] = {}
    rows_scanned = 0
    with open_dump_csv_lines(
        source, member_name=_member_name(source), error_cls=AcousticBrainzBulkError
    ) as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise AcousticBrainzBulkError(f"{source.name} is empty") from exc
        if header != expected_header:
            raise AcousticBrainzBulkError(f"{source.name} has an unsupported header")
        fields = expected_header[2:]
        for rows_scanned, row in enumerate(reader, start=1):
            if len(row) != len(expected_header):
                raise AcousticBrainzBulkError(
                    f"{source.name} row {rows_scanned + 1} has {len(row)} fields instead of "
                    f"{len(expected_header)}"
                )
            mbid = row[0].strip().casefold()
            if mbid not in target_mbids:
                if progress is not None and rows_scanned % progress_interval == 0:
                    progress(section, rows_scanned)
                continue
            try:
                offset = int(row[1])
            except ValueError as exc:
                raise AcousticBrainzBulkError(
                    f"{source.name} row {rows_scanned + 1} has an invalid submission_offset"
                ) from exc
            if offset < 0:
                raise AcousticBrainzBulkError(
                    f"{source.name} row {rows_scanned + 1} has a negative submission_offset"
                )
            # A recording can have several submissions; keep only the lowest offset, matching
            # the live API client's own "offset 0, else lowest available" preference.
            existing = best_offset.get(mbid)
            if existing is not None and offset >= existing:
                if progress is not None and rows_scanned % progress_interval == 0:
                    progress(section, rows_scanned)
                continue
            best_offset[mbid] = offset
            nested_by_mbid.setdefault(mbid, {})[section] = {
                field_name: _parse_value(
                    value, field_name, source_name=source.name, row_number=rows_scanned + 1
                )
                for field_name, value in zip(fields, row[2:], strict=True)
            }
            if progress is not None and rows_scanned % progress_interval == 0:
                progress(section, rows_scanned)
    if progress is not None and rows_scanned % progress_interval:
        progress(section, rows_scanned)
    return rows_scanned


def build_offline_acousticbrainz_provider(
    recording_mbids: Iterable[str],
    *,
    lowlevel_dump: str | Path | None = None,
    rhythm_dump: str | Path | None = None,
    tonal_dump: str | Path | None = None,
    progress: Callable[[str, int], None] | None = None,
    progress_interval: int = 1_000_000,
) -> tuple[OfflineAcousticBrainzProvider, BulkDumpScanReport]:
    """Scan the requested local dumps and retain only rows for the given recording MBIDs.

    Each of the three dumps is independent; supplying a subset still yields a valid provider
    with narrower coverage, matching the rest of the pipeline's "absent, not invented" policy.
    """

    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, int)
        or progress_interval < 1
    ):
        raise AcousticBrainzBulkError("progress_interval must be a positive integer")
    sources: dict[str, Path] = {
        section: Path(path)
        for section, path in (
            ("lowlevel", lowlevel_dump),
            ("rhythm", rhythm_dump),
            ("tonal", tonal_dump),
        )
        if path is not None
    }
    if not sources:
        raise AcousticBrainzBulkError(
            "At least one of lowlevel_dump, rhythm_dump, or tonal_dump is required"
        )
    target_mbids = frozenset(_validate_mbid(mbid) for mbid in recording_mbids)
    if not target_mbids:
        raise AcousticBrainzBulkError("Offline AcousticBrainz matching received no recording MBIDs")

    nested_by_mbid: dict[str, dict[str, dict[str, object]]] = {}
    rows_scanned = 0
    for section, source in sources.items():
        rows_scanned += _scan_section(
            section,
            source,
            target_mbids,
            nested_by_mbid,
            progress=progress,
            progress_interval=progress_interval,
        )

    documents = {
        mbid: AcousticBrainzDocument(recording_mbid=mbid, low_level=sections, high_level=None)
        for mbid, sections in nested_by_mbid.items()
    }
    report = BulkDumpScanReport(
        sections_scanned=tuple(sources),
        target_mbids=len(target_mbids),
        rows_scanned=rows_scanned,
        mbids_covered=len(documents),
    )
    return OfflineAcousticBrainzProvider(documents=documents), report
