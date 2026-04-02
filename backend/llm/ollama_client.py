from __future__ import annotations

from typing import Any

import httpx

from backend.config import get_config


class OllamaClient:
    """Minimal async client for Ollama /api/chat endpoint."""

    def __init__(self, base_url: str | None = None):
        cfg = get_config()
        self._base_url = (base_url or cfg.ollama.base_url).rstrip("/")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base_url}/api/tags")
            if resp.status_code != 200:
                return []
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

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
