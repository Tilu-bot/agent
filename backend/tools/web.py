from __future__ import annotations

from typing import Any

import httpx

from backend.config import get_config
from backend.tools.bus import BaseTool, ToolResult


class WebFetchTool(BaseTool):
    name = "web.fetch"
    description = "Fetch a URL and return its text content (up to max_bytes)."

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        url: str = params.get("url", "")
        if not url:
            return ToolResult.from_error(self.name, "Missing 'url' parameter")

        cfg = get_config()
        timeout = params.get("timeout", cfg.tools.web.timeout_seconds)
        max_bytes = cfg.tools.web.max_bytes

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "agentic/0.1"})
                content = resp.text[:max_bytes]
                return ToolResult.from_success(
                    self.name,
                    {
                        "url": url,
                        "status_code": resp.status_code,
                        "content": content,
                        "truncated": len(resp.text) > max_bytes,
                    },
                )
        except Exception as exc:
            return ToolResult.from_error(self.name, str(exc))
