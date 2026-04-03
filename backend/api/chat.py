"""Chat API — simple multi-turn conversation backed by the ModelRouter.

Endpoints
---------
POST /api/chat/message
    Send a list of messages, receive the assistant reply as JSON.

POST /api/chat/stream
    Same payload, but the response is an SSE stream of tokens so the UI can
    display text as it is generated rather than waiting for the full reply.

Both endpoints accept the same request body::

    {
      "messages": [{"role": "user", "content": "Hello!"}],
      "model":     "llama3.2:3b",          // optional — Ollama OR HF model ID
      "task_type": "fast",                 // optional, default "fast"
    }

When *model* contains a "/" (e.g. ``"Qwen/Qwen2.5-3B-Instruct"``) the
request is routed through the HuggingFace Inference API instead of the local
Ollama instance.  Set the ``HF_API_TOKEN`` environment variable to
authenticate.

SSE token format
----------------
Each event is a ``data:`` line with JSON::

    data: {"token": "Hello"}

A final ``data: [DONE]`` line marks the end of the stream.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.llm.router import ModelRouter, TaskType

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Request / response models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'.")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str | None = Field(
        None,
        description=(
            "Override the model used for this conversation. "
            "Pass an Ollama model name (e.g. 'llama3.2:3b') or a HuggingFace "
            "model ID (e.g. 'Qwen/Qwen2.5-3B-Instruct').  "
            "When omitted the 'fast' slot from the YAML config is used."
        ),
    )
    task_type: str = Field(
        "fast",
        description="Model routing slot: fast | reasoning | code | search | math | vision.",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_router(model: str | None) -> ModelRouter:
    """Return a ModelRouter, optionally pinning every slot to *model*."""
    if model:
        run_models = {slot: model for slot in TaskType.__members__}
        return ModelRouter(run_models=run_models)
    return ModelRouter()


def _to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/message")
async def chat_message(body: ChatRequest) -> dict[str, Any]:
    """Send messages and get the full assistant reply as JSON."""
    mr = _build_router(body.model)
    content = await mr.chat(body.task_type, _to_dicts(body.messages))
    return {"role": "assistant", "content": content}


@router.post("/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    """Send messages and receive the assistant reply as an SSE token stream."""
    mr = _build_router(body.model)
    messages = _to_dicts(body.messages)
    task_type = body.task_type

    async def event_stream():
        try:
            async for token in mr.chat_stream(task_type, messages):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
