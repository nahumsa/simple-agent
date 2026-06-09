import asyncio
import json

import pytest

from frameworks.barebones.tools import ChallengeDataTools
import frameworks.barebones.tools.url_fetch as url_fetch
from frameworks.barebones.tools.url_fetch import URL_FETCH_USER_AGENT, URLFetchError, fetch_url


class ResponseStub:
    def __init__(
        self,
        *,
        text: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        is_redirect: bool = False,
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.is_redirect = is_redirect
        self.encoding: str | None = None

    def raise_for_status(self) -> None:
        return None


def test_fetch_url_uses_requests_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_get(url: str, **kwargs: object) -> ResponseStub:
        calls.append({"url": url, **kwargs})
        return ResponseStub(
            text="abcdef",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

    monkeypatch.setattr(url_fetch, "_validate_public_host", lambda hostname: None)
    monkeypatch.setattr(url_fetch.requests, "get", fake_get)

    fetched = fetch_url("https://example.com/page", max_chars=3)

    assert fetched.url == "https://example.com/page"
    assert fetched.status == 200
    assert fetched.content_type == "text/plain; charset=utf-8"
    assert fetched.body == "abc"
    assert fetched.truncated is True
    assert calls == [
        {
            "url": "https://example.com/page",
            "headers": {"User-Agent": URL_FETCH_USER_AGENT},
            "timeout": 10,
            "allow_redirects": False,
        }
    ]


def test_fetch_url_rejects_private_ip() -> None:
    with pytest.raises(URLFetchError, match="Refusing to fetch"):
        fetch_url("http://127.0.0.1:8000")


def test_fetch_url_rejects_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: object) -> ResponseStub:
        return ResponseStub(
            text="",
            status_code=302,
            headers={"Location": "/next"},
            is_redirect=True,
        )

    monkeypatch.setattr(url_fetch, "_validate_public_host", lambda hostname: None)
    monkeypatch.setattr(url_fetch.requests, "get", fake_get)

    with pytest.raises(URLFetchError, match="Redirects are not followed"):
        fetch_url("https://example.com/start")


def test_challenge_data_tool_fetch_url_validates_arguments(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch_url(url: str, *, max_chars: int) -> url_fetch.FetchedURL:
        assert url == "https://example.com"
        assert max_chars == 12
        return url_fetch.FetchedURL(
            url=url,
            status=200,
            content_type="text/plain",
            body="hello",
            truncated=False,
        )

    tools = ChallengeDataTools(data_dir=tmp_path, search_db_path=tmp_path / "search.duckdb")
    monkeypatch.setattr(url_fetch, "fetch_url", fake_fetch_url)

    missing_url = asyncio.run(tools.call("fetch_url", {}, session=object(), tool_call_id="test"))
    invalid_max_chars = asyncio.run(
        tools.call(
            "fetch_url",
            {"url": "https://example.com", "max_chars": "12"},
            session=object(),
            tool_call_id="test",
        )
    )
    success = asyncio.run(
        tools.call(
            "fetch_url",
            {"url": "https://example.com", "max_chars": 12},
            session=object(),
            tool_call_id="test",
        )
    )

    assert missing_url == ("Missing required string argument: url", False)
    assert invalid_max_chars == ("max_chars must be an integer", False)
    assert success == (
        json.dumps(
            {
                "url": "https://example.com",
                "status": 200,
                "content_type": "text/plain",
                "body": "hello",
                "truncated": False,
            },
            indent=2,
        ),
        True,
    )
