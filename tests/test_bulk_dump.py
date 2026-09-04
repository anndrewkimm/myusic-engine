import tarfile
from pathlib import Path

import pytest

from myusic_engine.bulk_dump import open_dump_csv_lines


class _DumpError(ValueError):
    pass


def test_csv_source_is_streamed_directly(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8", newline="")

    with open_dump_csv_lines(csv_path, member_name="unused.csv", error_cls=_DumpError) as stream:
        assert list(stream) == ["a,b\n", "1,2\n"]


def test_tar_source_finds_a_nested_member_by_basename(tmp_path: Path) -> None:
    inner_csv = tmp_path / "rows.csv"
    inner_csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8", newline="")
    archive_path = tmp_path / "dump.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        archive.add(inner_csv, arcname="nested/dir/rows.csv")

    with open_dump_csv_lines(archive_path, member_name="rows.csv", error_cls=_DumpError) as stream:
        assert list(stream) == ["a,b\n", "1,2\n", "3,4\n"]


def test_tar_source_missing_the_expected_member_raises(tmp_path: Path) -> None:
    inner_csv = tmp_path / "rows.csv"
    inner_csv.write_text("a,b\n1,2\n", encoding="utf-8")
    archive_path = tmp_path / "dump.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        archive.add(inner_csv, arcname="rows.csv")

    with (
        pytest.raises(_DumpError, match="does not contain"),
        open_dump_csv_lines(archive_path, member_name="other.csv", error_cls=_DumpError) as stream,
    ):
        list(stream)


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "dump.json"
    bad_path.write_text("{}", encoding="utf-8")

    with (
        pytest.raises(_DumpError, match="CSV, tar, or tar.zst"),
        open_dump_csv_lines(bad_path, member_name="rows.csv", error_cls=_DumpError) as stream,
    ):
        list(stream)


def test_missing_source_raises(tmp_path: Path) -> None:
    with (
        pytest.raises(_DumpError, match="does not exist"),
        open_dump_csv_lines(
            tmp_path / "missing.csv", member_name="rows.csv", error_cls=_DumpError
        ) as stream,
    ):
        list(stream)
