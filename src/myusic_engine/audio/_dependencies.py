"""Lazy phase-3 dependency loading with a writable deterministic JIT cache."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=1)
def load_librosa() -> ModuleType:
    """Import librosa after routing Numba's cache away from read-only installations."""

    cache_dir = Path(tempfile.gettempdir()) / "myusic-engine-numba-cache"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError(
            "Audio analysis requires the phase3 extra: pip install -e '.[phase3]'"
        ) from exc
    return librosa
