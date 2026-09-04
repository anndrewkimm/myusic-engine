import csv
from pathlib import Path

import pytest

from myusic_engine.providers.acousticbrainz_bulk import (
    AcousticBrainzBulkError,
    build_offline_acousticbrainz_provider,
)

TARGET_MBID = "00000000-0000-4000-8000-000000000030"
OTHER_MBID = "00000000-0000-4000-8000-000000000040"
UNTARGETED_MBID = "00000000-0000-4000-8000-000000000050"


def _write_csv(path: Path, header: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _lowlevel_csv(path: Path, rows: tuple[tuple[object, ...], ...]) -> Path:
    return _write_csv(
        path,
        ("mbid", "submission_offset", "average_loudness", "dynamic_complexity", "mfcc_zero_mean"),
        rows,
    )


def _rhythm_csv(path: Path, rows: tuple[tuple[object, ...], ...]) -> Path:
    return _write_csv(
        path,
        (
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
        rows,
    )


def _tonal_csv(path: Path, rows: tuple[tuple[object, ...], ...]) -> Path:
    return _write_csv(
        path,
        (
            "mbid",
            "submission_offset",
            "key_key",
            "key_scale",
            "tuning_frequency",
            "tuning_equal_tempered_deviation",
        ),
        rows,
    )


def test_bulk_scan_merges_sections_and_prefers_lowest_offset(tmp_path: Path) -> None:
    lowlevel = _lowlevel_csv(
        tmp_path / "lowlevel.csv",
        (
            (TARGET_MBID, 1, 0.5, 5.0, -700.0),
            (TARGET_MBID, 0, 0.7, 5.6, -722.4),
            (OTHER_MBID, 0, 0.1, 1.0, -100.0),
        ),
    )
    rhythm = _rhythm_csv(
        tmp_path / "rhythm.csv",
        ((TARGET_MBID, 0, 120.0, 120, 120, 133, 133, 0.99, 2.86),),
    )
    tonal = _tonal_csv(
        tmp_path / "tonal.csv",
        ((TARGET_MBID, 0, "A", "major", 434.19, 0.14),),
    )

    provider, report = build_offline_acousticbrainz_provider(
        (TARGET_MBID,),
        lowlevel_dump=lowlevel,
        rhythm_dump=rhythm,
        tonal_dump=tonal,
    )

    assert report.mbids_covered == 1
    assert report.target_mbids == 1
    assert set(report.sections_scanned) == {"lowlevel", "rhythm", "tonal"}

    documents = provider.fetch((TARGET_MBID,))
    document = documents[TARGET_MBID]
    assert document.high_level is None
    assert document.low_level is not None
    # Offset 0 must win over offset 1, matching the live client's own tie-break.
    assert document.low_level["lowlevel"]["average_loudness"] == 0.7
    assert document.low_level["rhythm"]["bpm"] == 120.0
    assert document.low_level["tonal"]["key_key"] == "A"


def test_bulk_scan_skips_untargeted_mbids(tmp_path: Path) -> None:
    lowlevel = _lowlevel_csv(
        tmp_path / "lowlevel.csv",
        (
            (TARGET_MBID, 0, 0.7, 5.6, -722.4),
            (UNTARGETED_MBID, 0, 0.1, 1.0, -100.0),
        ),
    )

    provider, report = build_offline_acousticbrainz_provider((TARGET_MBID,), lowlevel_dump=lowlevel)

    assert report.mbids_covered == 1
    assert provider.fetch((UNTARGETED_MBID,)) == {}


def test_partial_section_coverage_when_a_dump_is_omitted(tmp_path: Path) -> None:
    lowlevel = _lowlevel_csv(tmp_path / "lowlevel.csv", ((TARGET_MBID, 0, 0.7, 5.6, -722.4),))

    provider, report = build_offline_acousticbrainz_provider((TARGET_MBID,), lowlevel_dump=lowlevel)

    document = provider.fetch((TARGET_MBID,))[TARGET_MBID]
    assert document.low_level is not None
    assert set(document.low_level) == {"lowlevel"}
    assert report.sections_scanned == ("lowlevel",)


def test_empty_field_becomes_none_and_invalid_field_raises(tmp_path: Path) -> None:
    lowlevel = _lowlevel_csv(tmp_path / "lowlevel.csv", ((TARGET_MBID, 0, "", 5.6, -722.4),))
    provider, _ = build_offline_acousticbrainz_provider((TARGET_MBID,), lowlevel_dump=lowlevel)
    document = provider.fetch((TARGET_MBID,))[TARGET_MBID]
    assert document.low_level is not None
    assert document.low_level["lowlevel"]["average_loudness"] is None

    bad_lowlevel = _lowlevel_csv(
        tmp_path / "bad_lowlevel.csv", ((TARGET_MBID, 0, "not-a-number", 5.6, -722.4),)
    )
    with pytest.raises(AcousticBrainzBulkError, match="average_loudness"):
        build_offline_acousticbrainz_provider((TARGET_MBID,), lowlevel_dump=bad_lowlevel)


def test_header_mismatch_raises(tmp_path: Path) -> None:
    wrong_header = tmp_path / "lowlevel.csv"
    _write_csv(wrong_header, ("mbid", "submission_offset", "unexpected"), ((TARGET_MBID, 0, 1),))

    with pytest.raises(AcousticBrainzBulkError, match="unsupported header"):
        build_offline_acousticbrainz_provider((TARGET_MBID,), lowlevel_dump=wrong_header)


def test_row_field_count_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "lowlevel.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("mbid,submission_offset,average_loudness,dynamic_complexity,mfcc_zero_mean\n")
        stream.write(f"{TARGET_MBID},0,0.7,5.6\n")

    with pytest.raises(AcousticBrainzBulkError, match="fields instead of"):
        build_offline_acousticbrainz_provider((TARGET_MBID,), lowlevel_dump=path)


def test_fetch_normalizes_case_and_rejects_malformed_mbids(tmp_path: Path) -> None:
    lowlevel = _lowlevel_csv(tmp_path / "lowlevel.csv", ((TARGET_MBID, 0, 0.7, 5.6, -722.4),))
    provider, _ = build_offline_acousticbrainz_provider((TARGET_MBID,), lowlevel_dump=lowlevel)

    assert TARGET_MBID in provider.fetch((TARGET_MBID.upper(),))
    with pytest.raises(AcousticBrainzBulkError):
        provider.fetch(("not-an-mbid",))


def test_no_dump_paths_raises(tmp_path: Path) -> None:
    with pytest.raises(AcousticBrainzBulkError, match="At least one"):
        build_offline_acousticbrainz_provider((TARGET_MBID,))


def test_no_target_mbids_raises(tmp_path: Path) -> None:
    lowlevel = _lowlevel_csv(tmp_path / "lowlevel.csv", ((TARGET_MBID, 0, 0.7, 5.6, -722.4),))
    with pytest.raises(AcousticBrainzBulkError, match="no recording MBIDs"):
        build_offline_acousticbrainz_provider((), lowlevel_dump=lowlevel)
