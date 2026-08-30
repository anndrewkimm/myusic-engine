from __future__ import annotations

import hashlib
import json
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

import myusic_engine.providers.http as http_module
from myusic_engine.providers import JsonCacheTransport, ProviderError


class FakeResponse:
    def __init__(self, payload: bytes, *, overflow: bool = False) -> None:
        self.payload = payload
        self.overflow = overflow
        self.read_count = 0

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_count += 1
        if self.read_count == 1:
            return self.payload
        return b"x" if self.overflow else b""


def test_transport_caches_encoded_request_and_supports_offline_replay(
    tmp_path, monkeypatch
) -> None:
    requests = []

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)
    transport = JsonCacheTransport(tmp_path, timeout_seconds=4.0)

    assert transport.get_json(
        "provider-test",
        "https://example.test/lookup",
        {"z": "last", "a": "first value"},
    ) == {"ok": True}
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "https://example.test/lookup?a=first+value&z=last"
    assert request.get_header("User-agent").startswith("myusic-engine/")
    assert timeout == 4.0

    monkeypatch.setattr(
        http_module,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("offline replay unexpectedly used the network"),
    )
    replay = JsonCacheTransport(tmp_path, offline=True)
    assert replay.get_json(
        "provider-test",
        "https://example.test/lookup",
        {"a": "first value", "z": "last"},
    ) == {"ok": True}


def test_transport_retries_transient_errors_and_caches_not_found(tmp_path, monkeypatch) -> None:
    sleeps: list[float] = []
    attempts = 0

    def transient_then_success(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            headers = Message()
            headers["Retry-After"] = "0.25"
            raise HTTPError(request.full_url, 429, "rate limited", headers, None)
        return FakeResponse(b"[]")

    monkeypatch.setattr(http_module, "urlopen", transient_then_success)
    monkeypatch.setattr(http_module.time, "sleep", sleeps.append)
    transport = JsonCacheTransport(tmp_path, max_retries=1)
    assert transport.get_json("retry-test", "https://example.test/retry") == []
    assert attempts == 2
    assert sleeps == [0.25]

    def not_found(request, *, timeout):
        raise HTTPError(request.full_url, 404, "missing", Message(), None)

    monkeypatch.setattr(http_module, "urlopen", not_found)
    assert transport.get_json("missing-test", "https://example.test/missing") is None
    cache_file = next((tmp_path / "missing-test").glob("*.json"))
    assert json.loads(cache_file.read_text(encoding="utf-8"))["payload"] is None


@pytest.mark.parametrize(
    ("namespace", "url"),
    [
        ("../escape", "https://example.test"),
        ("valid", "http://example.test"),
        ("UPPERCASE", "https://example.test"),
    ],
)
def test_transport_rejects_unsafe_cache_namespaces_and_urls(
    tmp_path, namespace: str, url: str
) -> None:
    with pytest.raises(ProviderError, match="namespace and HTTPS"):
        JsonCacheTransport(tmp_path).get_json(namespace, url)


@pytest.mark.parametrize(
    "url",
    [
        "https://",
        "https://user:password@example.test",
        "https://example.test/lookup#fragment",
        "https://exa mple.test",
    ],
)
def test_transport_rejects_malformed_or_secret_bearing_https_urls(tmp_path, url: str) -> None:
    with pytest.raises(ProviderError, match="namespace and HTTPS"):
        JsonCacheTransport(tmp_path).get_json("valid", url)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"minimum_interval_seconds": float("nan")},
        {"minimum_interval_seconds": float("inf")},
    ],
)
def test_transport_rejects_non_finite_timing_controls(tmp_path, kwargs: dict[str, object]) -> None:
    with pytest.raises(ProviderError, match="finite"):
        JsonCacheTransport(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_transport_rejects_invalid_or_oversized_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        http_module,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"not-json"),
    )
    with pytest.raises(ProviderError, match="invalid JSON"):
        JsonCacheTransport(tmp_path).get_json("invalid-test", "https://example.test")

    monkeypatch.setattr(
        http_module,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"{}", overflow=True),
    )
    with pytest.raises(ProviderError, match="exceeded 25 MB"):
        JsonCacheTransport(tmp_path).get_json("large-test", "https://example.test")


def test_transport_rejects_cache_envelopes_without_payload(tmp_path) -> None:
    namespace_dir = tmp_path / "broken-cache"
    namespace_dir.mkdir()
    cache_identity = json.dumps(
        ["https://example.test", []], ensure_ascii=True, separators=(",", ":")
    )
    digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
    (namespace_dir / f"{digest}.json").write_text('{"cache_schema_version": 1}\n', encoding="utf-8")

    with pytest.raises(ProviderError, match="unsupported schema"):
        JsonCacheTransport(tmp_path, offline=True).get_json("broken-cache", "https://example.test")


def test_transport_retries_network_errors_and_understands_http_date(monkeypatch, tmp_path) -> None:
    attempts = 0
    sleeps: list[float] = []

    def network_failure_then_success(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("temporary")
        return FakeResponse(b"{}")

    monkeypatch.setattr(http_module, "urlopen", network_failure_then_success)
    monkeypatch.setattr(http_module.time, "sleep", sleeps.append)
    assert (
        JsonCacheTransport(tmp_path, max_retries=1).get_json("network-test", "https://example.test")
        == {}
    )
    assert sleeps == [1.0]

    headers = Message()
    headers["Retry-After"] = "Wed, 21 Oct 2099 07:28:00 GMT"
    assert JsonCacheTransport._retry_delay(headers, 0) == 60.0
