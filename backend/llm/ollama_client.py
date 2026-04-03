from __future__ import annotations

from typing import Any

import httpx

from backend.config import get_config


class OllamaClient:
    """Minimal async client for Ollama /api/chat endpoint."""

    def __init__(self, base_url: str | None = None):
        cfg = get_config()
        self._base_url = (base_url or cfg.ollama.base_url).rstrip("/")
        self._default_timeout = cfg.llm.timeout_seconds
        self._per_type_timeout = cfg.llm.per_type_timeout

    def _timeout_for(self, task_type: str | None) -> int:
        """Return the timeout (seconds) for a given task type."""
        if task_type and task_type in self._per_type_timeout:
            return self._per_type_timeout[task_type]
        return self._default_timeout

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
        task_type: str | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        # format is a top-level Ollama field (e.g. "json") — not inside options.
        if format:
            payload["format"] = format
        timeout = self._timeout_for(task_type)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        task_type: str | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        timeout = self._timeout_for(task_type)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        task_type: str | None = None,
    ):
        """Async generator yielding content tokens from a streaming Ollama chat.

        Each yielded value is a plain string token/chunk.
        """
        import json as _json

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        timeout = self._timeout_for(task_type)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        token = data.get("message", {}).get("content") or ""
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except _json.JSONDecodeError:
                        pass

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base_url}/api/tags")
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

    async def pull_model(self, model: str):
        """Async generator that streams pull-progress events from Ollama.

        Each yielded value is a dict parsed from one NDJSON line of the
        ``/api/pull`` response, e.g.::

            {"status": "pulling manifest"}
            {"status": "downloading", "digest": "sha256:...",
             "total": 4000000000, "completed": 123456789}
            {"status": "success"}
        """
        payload = {"name": model, "stream": True}
        async with httpx.AsyncClient(timeout=3600) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/pull", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line:
                        import json
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            pass

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self._base_url)
                return resp.status_code < 500
        except Exception:
            return False

    async def embed(self, model: str, text: str) -> list[float]:
        """Return an embedding vector for *text* using the given model.

        Raises on any HTTP or network error so callers can fall back gracefully.
        """
        payload = {"model": model, "prompt": text}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self._base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            return resp.json().get("embedding", [])
