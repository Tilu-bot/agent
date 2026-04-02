from __future__ import annotations

import asyncio
from typing import Any

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
