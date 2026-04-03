"""Chat API — simple multi-turn conversation backed by the ModelRouter.

Endpoints
---------
POST /api/chat/message
    Send a list of messages, receive the assistant reply as JSON.

POST /api/chat/stream
    Same payload, but the response is an SSE stream of tokens so the UI can
    display text as it is generated rather than waiting for the full reply.

POST /api/chat/classify
    Classify whether the last user message needs the full agent pipeline
    (``"agentic"``) or can be answered directly by the LLM (``"direct"``).
    This is a fast heuristic — no LLM call is made.

Both message endpoints accept the same request body::

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
import re
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.llm.router import ModelRouter, TaskType

# ── Query classifier ──────────────────────────────────────────────────────────
# Patterns that signal the query genuinely needs external tools / multi-step
# agent planning (web search, file I/O, code execution, long research tasks).
_AGENTIC_RE = re.compile(
    r"\b("
    r"research|search (the )?web|look up|browse|find information|"
    r"fetch|download|scrape|crawl|"
    r"save (to|into|a) file|write (to|a) file|"
    r"create (a )?(file|folder|directory|project)|"
    r"run (the )?(code|script|command|program|tests?)|execute|compile|"
    r"analyze (the )?(data|file|dataset|csv)|read (the )?(file|csv|data)|"
    r"build (a |an |the )?\w+|install|set up|configure|"
    r"send (an? )?(email|message|notification)|"
    r"step 1|step one|first .{5,60} then|multiple steps|"
    r"compare .{5,80} and .{5,80} using data|"
    r"generate (a )?(report|chart|graph|plot|diagram)"
    r")\b",
    re.IGNORECASE,
)

# Patterns that strongly indicate a conversational / explanatory query.
_DIRECT_RE = re.compile(
    r"^("
    r"what (is|are|was|were|does|do)|"
    r"who (is|was|are)|when (did|was|is|are)|"
    r"where (is|was|are)|"
    r"how (does|do|did|many|much|long|to)|"
    r"why (is|are|was|did|do)|"
    r"explain|define|describe|tell me (about|what)|"
    r"difference between|compare|summarize|"
    r"can you (write|help|show|give|explain|create|make)|"
    r"write (a |an |the )?(simple |quick |short )?(code|script|function|class|"
    r"example|poem|story|essay|email|letter|snippet|hello world)|"
    r"give me (a |an )?(example|list|summary)|"
    r"translate|convert|calculate|compute|"
    r"is (it|this|there)|"
    r"yes|no|ok|thanks|thank you|hi|hello|hey"
    r")",
    re.IGNORECASE,
)


def _classify_message(message: str) -> str:
    """Heuristically classify *message* as ``'direct'`` or ``'agentic'``.

    The heuristic runs in O(len(message)) with no LLM call:

    1. Very short greetings / single-word messages → always direct.
    2. Agentic keyword match → agentic.
    3. Direct-pattern prefix match → direct.
    4. Long (>5 sentences) messages with no direct prefix → agentic.
    5. Default → direct.
    """
    msg = message.strip()
    if not msg:
        return "direct"

    # Always direct: very short (<= 60 chars) with no agentic signals
    if len(msg) <= 60 and not _AGENTIC_RE.search(msg):
        return "direct"

    # Agentic signals dominate
    if _AGENTIC_RE.search(msg):
        return "agentic"

    # Direct conversational prefix
    if _DIRECT_RE.match(msg):
        return "direct"

    # Long messages with many sentences that don't match a direct prefix
    sentences = [s.strip() for s in re.split(r"[.!?]", msg) if s.strip()]
    if len(sentences) > 5:
        return "agentic"

    return "direct"

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

_REASON_PREFIX_LEN = 30  # characters of the message shown in the classify reason string


def _classify_reason(message: str, mode: str) -> str:
    """Return a short human-readable explanation for why *mode* was chosen."""
    msg = message.strip()
    if not msg:
        return "Empty message."
    if mode == "agentic":
        m = _AGENTIC_RE.search(msg)
        if m:
            # m.group(0) is always a substring captured by our own pre-defined
            # keyword regex, never arbitrary user input, so interpolation is safe.
            keyword = m.group(0)
            return f"Detected agentic keyword: \"{keyword}\". Using agent pipeline."
        if len(msg.split()) > 40:
            return "Long multi-sentence request. Using agent pipeline for structured planning."
        return "Request pattern suggests multi-step execution. Using agent pipeline."
    # direct
    if len(msg) <= 60:
        return "Short conversational message. Answering directly."
    m = _DIRECT_RE.match(msg)
    if m:
        prefix = msg[:_REASON_PREFIX_LEN]
        return f"Conversational prefix detected (\"{prefix}…\"). Answering directly."
    return "No agentic signals found. Answering directly."

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
        # Emit model-info first so the UI can show which model is thinking.
        model_name = mr.select_model(task_type)
        yield f"data: {json.dumps({'type': 'model_info', 'model': model_name, 'task_type': task_type})}\n\n"
        try:
            async for token in mr.chat_stream(task_type, messages):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Classify endpoint ──────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


@router.post("/classify")
async def classify_chat(body: ClassifyRequest) -> dict[str, str]:
    """Classify whether the last user message needs the agent pipeline.

    Returns ``{"mode": "direct"}`` for conversational / simple questions that
    can be answered by a single LLM call, or ``{"mode": "agentic"}`` for
    queries that genuinely need tools, web search, file I/O, multi-step
    planning, or code execution.

    This is a pure heuristic — no LLM call is made — so it is fast and free.
    """
    # Find the last user message
    last_user = next(
        (m.content for m in reversed(body.messages) if m.role == "user"),
        "",
    )
    mode = _classify_message(last_user)
    reason = _classify_reason(last_user, mode)
    return {"mode": mode, "reason": reason}
