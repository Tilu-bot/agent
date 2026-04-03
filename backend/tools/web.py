from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from backend.config import get_config
from backend.tools.bus import BaseTool, ToolResult

# trafilatura extracts clean readable text from HTML pages.
# We import lazily so tests that don't exercise web.fetch don't pay the cost.
try:
    import trafilatura  # type: ignore[import-untyped]
    _HAS_TRAFILATURA = True
except ImportError:  # pragma: no cover
    _HAS_TRAFILATURA = False

# Private / reserved IP networks — requests to these are blocked to prevent SSRF.
_PRIVATE_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),     # loopback
    ipaddress.ip_network("10.0.0.0/8"),      # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),   # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918
    ipaddress.ip_network("169.254.0.0/16"),  # link-local
    ipaddress.ip_network("100.64.0.0/10"),   # shared address (RFC 6598)
    ipaddress.ip_network("::1/128"),         # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),        # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),       # IPv6 link-local
]


def _resolves_to_private(host: str) -> bool:
    """Return True if *host* (name or IP) resolves to a private/reserved address.

    This runs synchronously and should be called inside ``run_in_executor``.
    """
    # First try to parse as a literal IP.
    try:
        ip = ipaddress.ip_address(host)
        return any(ip in net for net in _PRIVATE_NETS)
    except ValueError:
        pass

    # Hostname — resolve all addresses and check each.
    try:
        addr_info = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        # Cannot resolve — not a known private address, allow the request to
        # proceed; httpx will surface a connection error if unreachable.
        return False

    for entry in addr_info:
        try:
            ip = ipaddress.ip_address(entry[4][0])
            if any(ip in net for net in _PRIVATE_NETS):
                return True
        except ValueError:
            continue
    return False


def _extract_text(html: str, url: str) -> str:
    """Return clean readable text; fall back to raw HTML if trafilatura unavailable."""
    if _HAS_TRAFILATURA:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if text:
            return text
    return html


class WebFetchTool(BaseTool):
    name = "web.fetch"
    description = (
        "Fetch a URL and return its readable text content (HTML is stripped). "
        "Pass 'raw=true' to get raw HTML instead."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        url: str = params.get("url", "")
        if not url:
            return ToolResult.from_error(self.name, "Missing 'url' parameter")

        # Only allow http(s) schemes.
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult.from_error(
                self.name, f"Blocked: unsupported URL scheme '{parsed.scheme}'"
            )

        host = parsed.hostname or ""
        if not host:
            return ToolResult.from_error(self.name, "Invalid URL: no host")

        # SSRF protection: block requests to private / internal addresses.
        is_private = await asyncio.get_event_loop().run_in_executor(
            None, _resolves_to_private, host
        )
        if is_private:
            return ToolResult.from_error(
                self.name,
                f"Blocked: '{host}' resolves to a private or reserved address",
            )

        cfg = get_config()
        timeout = params.get("timeout", cfg.tools.web.timeout_seconds)
        max_chars = cfg.tools.web.max_bytes
        raw_mode: bool = str(params.get("raw", "false")).lower() == "true"

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "agentic/0.1"})
                html = resp.text

            if raw_mode:
                content = html[:max_chars]
                truncated = len(html) > max_chars
            else:
                # Run blocking trafilatura in a thread to avoid blocking the event loop
                text = await asyncio.get_event_loop().run_in_executor(
                    None, _extract_text, html, url
                )
                content = text[:max_chars]
                truncated = len(text) > max_chars

            return ToolResult.from_success(
                self.name,
                {
                    "url": url,
                    "status_code": resp.status_code,
                    "content": content,
                    "truncated": truncated,
                    "extraction": "text" if not raw_mode else "raw",
                },
            )
        except Exception as exc:
            return ToolResult.from_error(self.name, str(exc))


class WebSearchTool(BaseTool):
    name = "web.search"
    description = (
        "Search the web using DuckDuckGo (no API key required). "
        "Returns a list of results with title, url, and snippet."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        query: str = params.get("query", "")
        if not query:
            return ToolResult.from_error(self.name, "Missing 'query' parameter")

        max_results: int = int(params.get("max_results", 5))
        max_results = min(max_results, 20)  # hard cap

        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]
        except ImportError:
            return ToolResult.from_error(
                self.name,
                "duckduckgo-search is not installed. Run: pip install duckduckgo-search",
            )

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._ddg_search, query, max_results
            )
            return ToolResult.from_success(self.name, {"query": query, "results": results})
        except Exception as exc:
            return ToolResult.from_error(self.name, str(exc))

    @staticmethod
    def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
