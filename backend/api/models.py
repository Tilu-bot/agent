"""Models API — list available Ollama models and manage model slot assignments.

Endpoints
---------
GET  /api/models
    Returns the locally available Ollama models and the current slot → model
    mapping (merging YAML config with any in-memory runtime overrides).

PUT  /api/models/slots
    Update one or more model slot assignments at runtime.  Changes take
    effect immediately for all new runs; existing running runs are not
    affected.  Changes are NOT persisted to config.yaml.

DELETE /api/models/slots
    Reset all runtime overrides back to the YAML config defaults.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import (
    clear_runtime_model_overrides,
    get_config,
    get_runtime_model_overrides,
    set_runtime_model_override,
)
from backend.llm.ollama_client import OllamaClient

router = APIRouter(prefix="/api/models", tags=["models"])

_SLOTS = ("fast", "reasoning", "code", "search", "math", "vision", "embedding")


def _current_slots() -> dict[str, str]:
    """Return the effective slot → model mapping (config + runtime overrides)."""
    cfg = get_config()
    m = cfg.models
    slots: dict[str, str] = {
        "fast": m.fast,
        "reasoning": m.reasoning,
        "code": m.code,
        "search": m.search,
        "math": m.math,
        "vision": m.vision,
        "embedding": m.embedding,
    }
    # Overlay runtime overrides
    slots.update(get_runtime_model_overrides())
    return slots


@router.get("")
async def list_models() -> dict[str, Any]:
    """List available Ollama models, current slot assignments, and capability profiles."""
    from backend.llm.model_registry import get_registry

    ollama = OllamaClient()
    try:
        available = await ollama.list_models()
    except Exception:
        available = []

    # Refresh (or reuse) the model capability registry
    registry = await get_registry()
    capabilities: dict[str, list[str]] = {
        p.name: sorted(p.capabilities)
        for p in registry.profiles()
    }

    # Annotate each slot with the model that will actually be used
    # (registry may have substituted the configured model for a better one)
    from backend.llm.router import ModelRouter, TaskType
    _temp_router = ModelRouter()
    effective_slots: dict[str, str] = {}
    for slot in _SLOTS:
        try:
            tt = TaskType(slot)
            effective_slots[slot] = _temp_router.select_model(tt)
        except ValueError:
            effective_slots[slot] = _current_slots().get(slot, "")

    return {
        "available": available,
        "slots": _current_slots(),
        "effective_slots": effective_slots,
        "slot_names": list(_SLOTS),
        "runtime_overrides": get_runtime_model_overrides(),
        "capabilities": capabilities,
    }


class UpdateSlotsRequest(BaseModel):
    slots: dict[str, str]


@router.put("/slots")
async def update_slots(body: UpdateSlotsRequest) -> dict[str, Any]:
    """Override one or more model slot assignments at runtime.

    Only the slots listed in the request body are changed.  Other slots keep
    their current values (YAML config or previously set runtime override).

    Slots: fast, reasoning, code, search, math, vision, embedding.
    """
    unknown = [k for k in body.slots if k not in _SLOTS]
    if unknown:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Unknown slot(s): {unknown}. Valid slots: {list(_SLOTS)}",
        )
    for slot, model in body.slots.items():
        if model.strip():
            set_runtime_model_override(slot, model.strip())

    return {"updated": body.slots, "slots": _current_slots()}


@router.delete("/slots")
async def reset_slots() -> dict[str, Any]:
    """Reset all runtime model slot overrides to the YAML config defaults."""
    clear_runtime_model_overrides()
    return {"reset": True, "slots": _current_slots()}


@router.post("/registry/refresh")
async def refresh_model_registry() -> dict[str, Any]:
    """Re-probe Ollama for available models and rebuild the capability registry.

    Call this after pulling a new model so the router and planner can
    immediately start using it for appropriate tasks.
    """
    from backend.llm.model_registry import refresh_registry

    registry = await refresh_registry()
    return {
        "available": registry.available_models(),
        "capabilities": {
            p.name: sorted(p.capabilities)
            for p in registry.profiles()
        },
    }


@router.get("/capabilities")
async def model_capabilities() -> dict[str, Any]:
    """Return the full model capability registry.

    Shows which models are installed, what each is expert at, and which
    model will actually be used for each task slot (may differ from the
    YAML-configured value when that model is not installed).
    """
    from backend.llm.model_registry import get_registry
    from backend.llm.router import ModelRouter, TaskType

    registry = await get_registry()
    _temp_router = ModelRouter()

    slot_resolution: list[dict[str, str]] = []
    for slot in _SLOTS:
        configured = _current_slots().get(slot, "")
        try:
            tt = TaskType(slot)
            effective = _temp_router.select_model(tt)
        except ValueError:
            effective = configured
        reason = ""
        if registry.ready and configured != effective:
            reason = (
                f"'{configured}' is not installed; using best available "
                f"'{effective}' for '{slot}' tasks"
            )
        elif registry.ready and configured:
            reason = "configured model is installed"
        slot_resolution.append({
            "slot": slot,
            "configured": configured,
            "effective": effective,
            "reason": reason,
        })

    return {
        "models": [
            {
                "name": p.name,
                "capabilities": sorted(p.capabilities),
            }
            for p in sorted(registry.profiles(), key=lambda x: x.name)
        ],
        "slot_resolution": slot_resolution,
        "summary": registry.capability_summary(),
    }


@router.get("/pull")
async def pull_model(model: str = Query(..., description="Ollama model tag to pull, e.g. llama3.2:3b")):
    """Stream Ollama pull progress as Server-Sent Events.

    Each SSE ``data:`` payload is a JSON object with at minimum a ``status``
    field.  Downloading events also include ``total`` and ``completed`` byte
    counts so the UI can render a progress bar.

    The stream ends with ``{"status": "done"}`` after Ollama reports success,
    or ``{"status": "error", "error": "..."}`` on failure.
    """
    ollama = OllamaClient()

    async def _generate() -> AsyncGenerator[str, None]:
        try:
            async for event in ollama.pull_model(model):
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'status': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
