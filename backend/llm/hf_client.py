"""HuggingFace Inference API client.

Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint exposed by
``https://api-inference.huggingface.co``.  Requires a HuggingFace API token
set via the ``HF_API_TOKEN`` environment variable (free tier works for most
public models; a Pro subscription unlocks larger models).

Usage
-----
Any model ID of the form ``org/model-name`` is automatically routed through
this client by ``ModelRouter``.  You can also use the client directly::

    client = HFClient()
    result = await client.chat("Qwen/Qwen2.5-3B-Instruct", messages)
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

import httpx


class HFClient:
    """Async client for the HuggingFace Inference API (OpenAI-compatible)."""

    BASE_URL = "https://api-inference.huggingface.co"

    def __init__(self, token: str | None = None, timeout: int = 120):
        self._token = token or os.environ.get("HF_API_TOKEN", "")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _chat_url(self, model: str) -> str:
        return f"{self.BASE_URL}/models/{model}/v1/chat/completions"

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the HF Inference API chat endpoint.

        Returns a dict shaped like Ollama's chat response so ``ModelRouter``
        can treat both backends uniformly::

            {
                "message": {"role": "assistant", "content": "…"},
                "prompt_eval_count": 42,
                "eval_count": 17,
            }
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "num_predict" in options:
                payload["max_new_tokens"] = options["num_predict"]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self._chat_url(model), json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()

        content: str = data["choices"][0]["message"]["content"]
        usage: dict[str, int] = data.get("usage", {})
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
        """Async generator that yields content tokens from the SSE stream."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "num_predict" in options:
                payload["max_new_tokens"] = options["num_predict"]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                self._chat_url(model),
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content") or ""
                        if token:
                            yield token
                    except Exception:
                        pass

    async def is_available(self) -> bool:
        """Return True if the HF Inference API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self.BASE_URL)
                return resp.status_code < 500
        except Exception:
            return False
