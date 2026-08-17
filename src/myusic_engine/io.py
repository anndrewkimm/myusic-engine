"""Small deterministic file-writing helpers shared by local pipeline stages."""

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Replace one text file atomically after its complete content is on disk."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
