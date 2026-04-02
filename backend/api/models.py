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

from typing import Any

from fastapi import APIRouter
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
    """List available Ollama models and current slot assignments."""
    ollama = OllamaClient()
    try:
        available = await ollama.list_models()
    except Exception:
        available = []

    return {
        "available": available,
        "slots": _current_slots(),
        "slot_names": list(_SLOTS),
        "runtime_overrides": get_runtime_model_overrides(),
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
