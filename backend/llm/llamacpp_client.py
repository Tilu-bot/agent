"""llama.cpp HTTP server client.

llama.cpp exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint when
started with ``--api`` (or ``--server``).  This client speaks that API so any
local model loaded by llama.cpp becomes immediately available to the router.

Typical llama.cpp startup::

    llama-server --model ./models/Qwen2.5-Coder-3B.gguf --port 8080 --api

The client auto-detects running llama.cpp servers at well-known ports via
:meth:`probe_servers` and is consumed by :class:`ModelRegistry`.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

import httpx

# Ports we probe when auto-discovering llama.cpp servers.
_DISCOVERY_PORTS: list[int] = [8080, 8081, 8082, 8083, 11435]


class LlamaCppClient:
    """Async client for a llama.cpp ``/v1/chat/completions`` endpoint."""

    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ── Public API ─────────────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        """Return True when the server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code < 500
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, str]]:
        """Return models from ``/v1/models`` (id + backend="llamacpp")."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                models = data.get("data", [])
                return [
                    {"id": m.get("id", "unknown"), "base_url": self._base_url, "backend": "llamacpp"}
                    for m in models
                ]
        except Exception:
            return []

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat request; return an Ollama-shaped response dict."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            for k, v in options.items():
                payload[k] = v
        if format == "json":
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        content: str = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Async generator yielding content tokens."""
        import json as _json

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            for k, v in options.items():
                payload[k] = v

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content") or ""
                        if token:
                            yield token
                    except Exception:
                        pass

    # ── Class-level helpers ────────────────────────────────────────────────────

    @classmethod
    async def probe_servers(cls) -> list["LlamaCppClient"]:
        """Return a client for every llama.cpp server found on localhost.

        Probes :data:`_DISCOVERY_PORTS` concurrently; returns only reachable
        servers.
        """
        import asyncio

        async def _probe(port: int) -> "LlamaCppClient | None":
            c = cls(f"http://localhost:{port}")
            if await c.is_available():
                return c
            return None

        results = await asyncio.gather(*[_probe(p) for p in _DISCOVERY_PORTS])
        return [r for r in results if r is not None]
