"""Image analysis tool — multimodal vision via Ollama.

Calls an Ollama vision model (default: llava:7b) with an image and a
prompt, returning the model's textual description or answer.

The image can be:
  * A local file path — the file is base64-encoded and embedded inline.
  * A URL — downloaded and base64-encoded automatically.

Requires a vision-capable model pulled in Ollama, e.g.:
    ollama pull llava:7b
    ollama pull minicpm-v
    ollama pull llama3.2-vision:11b
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import httpx

from backend.config import get_config
from backend.llm.ollama_client import OllamaClient
from backend.tools.bus import BaseTool, ToolResult


async def _load_image_b64(source: str) -> str:
    """Return base64-encoded image bytes from a file path or URL."""
    if source.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(source)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        return base64.b64encode(path.read_bytes()).decode()


class ImageAnalyzeTool(BaseTool):
    name = "image.analyze"
    description = (
        "Analyze an image using a local vision model (Ollama). "
        "Params: {\"source\": \"<file path or URL>\", "
        "\"prompt\": \"Describe this image\" (optional), "
        "\"model\": \"llava:7b\" (optional)}. "
        "Returns the model's textual analysis of the image."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        source: str = params.get("source", "")
        if not source:
            return ToolResult.from_error(self.name, "Missing 'source' parameter (file path or URL)")

        cfg = get_config()
        model: str = params.get("model", cfg.models.vision)
        prompt: str = params.get("prompt", "Describe this image in detail.")

        try:
            image_b64 = await _load_image_b64(source)
        except Exception as exc:
            return ToolResult.from_error(self.name, f"Could not load image: {exc}")

        # Ollama vision API: POST /api/generate with images array
        base_url = cfg.ollama.base_url.rstrip("/")
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("response", "").strip()
            return ToolResult.from_success(
                self.name,
                {
                    "source": source,
                    "model": model,
                    "prompt": prompt,
                    "analysis": answer,
                },
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult.from_error(
                    self.name,
                    f"Model '{model}' not found in Ollama. "
                    f"Pull it with: ollama pull {model}",
                )
            return ToolResult.from_error(self.name, f"Ollama request failed: {exc}")
        except Exception as exc:
            return ToolResult.from_error(self.name, f"Image analysis failed: {exc}")
