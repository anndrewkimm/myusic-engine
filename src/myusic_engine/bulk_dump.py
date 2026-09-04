"""Streaming line access to a local CSV, tar, or tar.zst bulk dump.

Shared by pipeline stages that scan a large official dataset export (a single CSV member,
optionally tar- and zstd-wrapped) instead of calling a live network API.
"""

from __future__ import annotations

import codecs
import importlib
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import IO


def _utf8_physical_lines(stream: IO[bytes], *, chunk_size: int = 1 << 16) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    while block := stream.read(chunk_size):
        lines = (pending + decoder.decode(block)).split("\n")
        pending = lines.pop()
        for line in lines:
            yield line + "\n"
    pending += decoder.decode(b"", final=True)
    if pending:
        yield pending


@contextmanager
def _zstd_tar(source: Path, error_cls: type[ValueError]) -> Iterator[tarfile.TarFile]:
    if sys.version_info >= (3, 14):
        try:
            with tarfile.open(source, mode="r|zst") as archive:
                yield archive
        except (OSError, tarfile.TarError) as exc:
            raise error_cls(f"{source.name} is unreadable") from exc
        return

    try:
        zstandard = importlib.import_module("zstandard")
    except ModuleNotFoundError as exc:
        raise error_cls(
            "Reading tar.zst on Python 3.11-3.13 requires the zstandard package"
        ) from exc
    try:
        with source.open("rb") as raw_stream:
            reader = zstandard.ZstdDecompressor().stream_reader(raw_stream)
            try:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    yield archive
            finally:
                reader.close()
    except (OSError, tarfile.TarError) as exc:
        raise error_cls(f"{source.name} is unreadable") from exc


@contextmanager
def _archive_member_lines(
    archive: tarfile.TarFile, member_name: str, error_cls: type[ValueError]
) -> Iterator[Iterator[str]]:
    for member in archive:
        if member.isfile() and PurePosixPath(member.name).name == member_name:
            binary_stream = archive.extractfile(member)
            if binary_stream is None:
                break
            # Streaming tar members are intentionally not seekable. Decode chunks ourselves
            # so U+2028/U+2029 inside CSV fields are not mistaken for physical line endings.
            stream = _utf8_physical_lines(binary_stream)
            try:
                yield stream
            except UnicodeError as exc:
                raise error_cls(f"{member_name} contains invalid UTF-8") from exc
            finally:
                binary_stream.close()
            return
    raise error_cls(f"Dump does not contain {member_name}")


@contextmanager
def open_dump_csv_lines(
    source: Path, *, member_name: str, error_cls: type[ValueError]
) -> Iterator[Iterator[str]]:
    """Stream physical text lines from a local ``.csv``, ``.tar``, or ``.tar.zst`` dump.

    A ``.csv`` file is read directly. A ``.tar``/``.tar.zst`` archive must contain a member
    named ``member_name``; only that member is decompressed and streamed, never the whole
    archive. All failures raise ``error_cls`` so callers can report one domain-specific error.
    """

    if not source.is_file():
        raise error_cls(f"{source} does not exist or is not a file")
    if source.suffix.casefold() == ".csv":
        try:
            with source.open("r", encoding="utf-8", newline="") as stream:
                yield stream
        except (OSError, UnicodeError) as exc:
            raise error_cls(f"{source.name} is unreadable") from exc
        return
    if source.suffix.casefold() == ".tar":
        try:
            with (
                tarfile.open(source, mode="r|") as archive,
                _archive_member_lines(archive, member_name, error_cls) as stream,
            ):
                yield stream
        except error_cls:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise error_cls(f"{source.name} is unreadable") from exc
        return
    if not source.name.casefold().endswith(".tar.zst"):
        raise error_cls(f"{source.name} must be a CSV, tar, or tar.zst dump")

    with (
        _zstd_tar(source, error_cls) as archive,
        _archive_member_lines(archive, member_name, error_cls) as stream,
    ):
        yield stream
