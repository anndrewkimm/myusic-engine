"""Small standard-library JSON transport with hashed cache keys and polite retries."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from myusic_engine.io import atomic_write_text


class ProviderError(ValueError):
    """Raised when a permitted provider request or cached response is unusable."""


class JsonCacheTransport:
    """Fetch JSON through a deterministic on-disk cache without exposing query text in paths."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        user_agent: str = "myusic-engine/0.1.0 (private-local-research)",
        timeout_seconds: float = 30.0,
        minimum_interval_seconds: float = 0.0,
        max_retries: int = 3,
        offline: bool = False,
    ) -> None:
        if not user_agent.strip():
            raise ProviderError("Provider user_agent must be non-empty")
        if timeout_seconds <= 0:
            raise ProviderError("Provider timeout_seconds must be positive")
        if minimum_interval_seconds < 0:
            raise ProviderError("Provider minimum_interval_seconds must be non-negative")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ProviderError("Provider max_retries must be a non-negative integer")
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.max_retries = max_retries
        self.offline = offline
        self._last_request_started: float | None = None

    def get_json(
        self,
        namespace: str,
        url: str,
        parameters: Mapping[str, str] | None = None,
    ) -> object | None:
        """Return a cached or freshly fetched JSON value; HTTP 404 is represented by null."""

        if not namespace.strip() or not url.startswith("https://"):
            raise ProviderError("Provider namespace and HTTPS URL are required")
        ordered_parameters = sorted((parameters or {}).items())
        cache_identity = json.dumps(
            [url, ordered_parameters], ensure_ascii=True, separators=(",", ":")
        )
        digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / namespace / f"{digest}.json"
        if cache_path.is_file():
            return self._read_cache(cache_path)
        if self.offline:
            raise ProviderError("Provider cache miss while offline mode is enabled")

        query = urlencode(ordered_parameters)
        request_url = f"{url}?{query}" if query else url
        payload = self._request_json(request_url)
        envelope = {"cache_schema_version": 1, "payload": payload}
        atomic_write_text(
            cache_path,
            json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return payload

    @staticmethod
    def _read_cache(path: Path) -> object | None:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError("A cached provider response is unreadable") from exc
        if not isinstance(envelope, Mapping) or envelope.get("cache_schema_version") != 1:
            raise ProviderError("A cached provider response has an unsupported schema")
        return envelope.get("payload")

    def _request_json(self, request_url: str) -> object | None:
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            request = Request(
                request_url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
            )
            self._last_request_started = time.monotonic()
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read(25_000_000)
                    if response.read(1):
                        raise ProviderError("Provider JSON response exceeded 25 MB")
            except HTTPError as exc:
                if exc.code in {404, 410}:
                    return None
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise ProviderError(f"Provider request failed with HTTP {exc.code}") from exc
                retry_after = self._retry_delay(exc.headers, attempt)
                time.sleep(retry_after)
                continue
            except (TimeoutError, URLError) as exc:
                if attempt >= self.max_retries:
                    raise ProviderError("Provider request failed after retries") from exc
                time.sleep(min(30.0, 2.0**attempt))
                continue
            try:
                return cast(object, json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProviderError("Provider returned invalid JSON") from exc
        raise ProviderError("Provider request exhausted retries")

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_started is None:
            return
        elapsed = time.monotonic() - self._last_request_started
        remaining = self.minimum_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _retry_delay(headers: Message[str, str], attempt: int) -> float:
        for name in ("Retry-After", "X-RateLimit-Reset-In"):
            raw_value = headers.get(name)
            if raw_value is None:
                continue
            try:
                return max(0.1, min(60.0, float(raw_value)))
            except ValueError:
                continue
        return min(30.0, 2.0**attempt)
