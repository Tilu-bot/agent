"""Scratchpad tools — per-run in-memory key-value store.

The scratchpad lets the agent persist intermediate results across tasks
within a single run (e.g. store a URL found in task 1, read it in task 3).
It is fully in-memory and never persisted to disk.

The run_id is injected by the ToolBus as ``_run_id`` in params.
"""
from __future__ import annotations

from typing import Any

from backend.tools.bus import BaseTool, ToolResult

# Global store: run_id → {key: value}
_STORE: dict[str, dict[str, Any]] = {}


def _get_pad(run_id: str) -> dict[str, Any]:
    if run_id not in _STORE:
        _STORE[run_id] = {}
    return _STORE[run_id]


def clear_run_scratchpad(run_id: str) -> None:
    """Remove all scratchpad entries for a finished run."""
    _STORE.pop(run_id, None)


class ScratchpadWriteTool(BaseTool):
    name = "scratchpad.write"
    description = (
        "Store a value in the run-scoped in-memory scratchpad so other tasks can read it. "
        "Params: {\"key\": \"<name>\", \"value\": <any JSON value>}."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        run_id: str = params.get("_run_id", "")
        key: str = params.get("key", "")
        if not key:
            return ToolResult.from_error(self.name, "Missing 'key' parameter")
        value = params.get("value")
        _get_pad(run_id)[key] = value
        return ToolResult.from_success(
            self.name, {"key": key, "stored": True}
        )


class ScratchpadReadTool(BaseTool):
    name = "scratchpad.read"
    description = (
        "Read a value previously stored in the run-scoped scratchpad. "
        "Params: {\"key\": \"<name>\"}. Returns null if the key does not exist. "
        "Use scratchpad.list to see all stored keys."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        run_id: str = params.get("_run_id", "")
        key: str = params.get("key", "")
        if not key:
            return ToolResult.from_error(self.name, "Missing 'key' parameter")
        pad = _get_pad(run_id)
        if key not in pad:
            return ToolResult.from_success(
                self.name, {"key": key, "found": False, "value": None}
            )
        return ToolResult.from_success(
            self.name, {"key": key, "found": True, "value": pad[key]}
        )


class ScratchpadListTool(BaseTool):
    name = "scratchpad.list"
    description = (
        "List all keys currently stored in the run-scoped scratchpad. "
        "No params required."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        run_id: str = params.get("_run_id", "")
        pad = _get_pad(run_id)
        return ToolResult.from_success(self.name, {"keys": list(pad.keys())})
