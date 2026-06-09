"""Safe URL fetching helper for the barebones tool registry."""

from __future__ import annotations

import ipaddress
import socket
import json
import urllib.parse
from dataclasses import asdict, dataclass

import requests
from requests import RequestException

from agent_core.types import JsonObject, ToolSpec

DEFAULT_URL_FETCH_MAX_CHARS = 10_000
MIN_URL_FETCH_MAX_CHARS = 1
MAX_URL_FETCH_MAX_CHARS = 50_000
URL_FETCH_TIMEOUT_SECONDS = 10
URL_FETCH_USER_AGENT = "ai-powered-chatbot/0.1 URL fetch tool"
BLOCKED_ADDRESS_MESSAGE = (
    "Refusing to fetch URLs that resolve to local, private, loopback, link-local, "
    "reserved, or multicast addresses"
)


FETCH_URL_SPEC: ToolSpec = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch a public HTTP or HTTPS URL and return its text content. "
            "Local, private, loopback, link-local, reserved, and multicast "
            "addresses are refused. Redirects are not followed automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Public http(s) URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Maximum response characters to return. Defaults to "
                        f"{DEFAULT_URL_FETCH_MAX_CHARS}."
                    ),
                    "minimum": MIN_URL_FETCH_MAX_CHARS,
                    "maximum": MAX_URL_FETCH_MAX_CHARS,
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}


class URLFetchError(Exception):
    """Raised when a URL cannot be fetched safely."""


@dataclass(frozen=True)
class FetchedURL:
    """Response data returned by the URL fetch tool."""

    url: str
    status: int
    content_type: str
    body: str
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def call_fetch_url(args: JsonObject) -> tuple[str, bool]:
    """Validate tool arguments, fetch the URL, and serialize the result."""
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        return "Missing required string argument: url", False

    max_chars = args.get("max_chars", DEFAULT_URL_FETCH_MAX_CHARS)
    if not isinstance(max_chars, int):
        return "max_chars must be an integer", False

    try:
        fetched = fetch_url(url, max_chars=max_chars)
    except URLFetchError as exc:
        return str(exc), False

    return json.dumps(fetched.to_dict(), indent=2), True


def fetch_url(
    url: str,
    *,
    max_chars: int = DEFAULT_URL_FETCH_MAX_CHARS,
    timeout_seconds: int = URL_FETCH_TIMEOUT_SECONDS,
) -> FetchedURL:
    """Fetch an HTTP(S) URL after validating the target host."""
    normalized_url = _validate_url(url)
    max_chars = _clamp_max_chars(max_chars)

    try:
        response = requests.get(
            normalized_url,
            headers={"User-Agent": URL_FETCH_USER_AGENT},
            timeout=timeout_seconds,
            allow_redirects=False,
        )
        _raise_for_redirect(response, normalized_url)
        response.raise_for_status()
    except RequestException as exc:
        raise URLFetchError(f"Could not fetch {normalized_url}: {exc}") from exc

    response.encoding = response.encoding or "utf-8"
    body = response.text[:max_chars]

    return FetchedURL(
        url=normalized_url,
        status=response.status_code,
        content_type=response.headers.get("Content-Type", ""),
        body=body,
        truncated=len(response.text) > max_chars,
    )


def _validate_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        raise URLFetchError("Missing required string argument: url")

    parsed = urllib.parse.urlparse(stripped)
    if parsed.scheme not in {"http", "https"}:
        raise URLFetchError("Only http and https URLs can be fetched")
    if not parsed.hostname:
        raise URLFetchError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise URLFetchError("URLs with embedded credentials are not allowed")

    _validate_public_host(parsed.hostname)
    return urllib.parse.urlunparse(parsed)


def _validate_public_host(hostname: str) -> None:
    addresses = _resolve_addresses(hostname)
    if any(_is_blocked_address(address) for address in addresses):
        raise URLFetchError(f"{BLOCKED_ADDRESS_MESSAGE}: {hostname}")


def _resolve_addresses(hostname: str) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        return {ipaddress.ip_address(hostname)}
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise URLFetchError(f"Could not resolve hostname: {hostname}") from exc

    addresses = {ipaddress.ip_address(info[4][0]) for info in infos if info[4]}
    if not addresses:
        raise URLFetchError(f"Could not resolve hostname: {hostname}")
    return addresses


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_reserved,
            address.is_multicast,
            address.is_unspecified,
        )
    )


def _raise_for_redirect(response: requests.Response, url: str) -> None:
    if not response.is_redirect:
        return

    location = response.headers.get("Location")
    if not location:
        raise URLFetchError(f"Redirect from {url} did not include a Location header")

    target = urllib.parse.urljoin(url, location)
    raise URLFetchError(
        "Redirects are not followed automatically. "
        f"Fetch the validated Location URL instead: {target}"
    )


def _clamp_max_chars(max_chars: int) -> int:
    return max(MIN_URL_FETCH_MAX_CHARS, min(max_chars, MAX_URL_FETCH_MAX_CHARS))
