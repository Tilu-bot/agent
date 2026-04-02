from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ToolInput:
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""  # injected by the system; not exposed to the LLM


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: str | None = None
    sha256: str | None = None
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "sha256": self.sha256,
            "executed_at": self.executed_at,
        }

    @classmethod
    def from_success(cls, tool_name: str, output: Any) -> "ToolResult":
        serialized = json.dumps(output, default=str)
        sha = hashlib.sha256(serialized.encode()).hexdigest()
        return cls(tool_name=tool_name, success=True, output=output, sha256=sha)

    @classmethod
    def from_error(cls, tool_name: str, error: str) -> "ToolResult":
        return cls(tool_name=tool_name, success=False, output=None, error=error)


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        ...


class ToolBus:
    """Central registry and dispatcher for tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def describe_tools(self) -> dict[str, str]:
        """Return a mapping of tool name → description for all registered tools."""
        return {name: tool.description for name, tool in self._tools.items()}

    async def call(self, tool_input: ToolInput) -> ToolResult:
        tool = self._tools.get(tool_input.tool_name)
        if tool is None:
            return ToolResult.from_error(
                tool_input.tool_name, f"Unknown tool: {tool_input.tool_name}"
            )
        try:
            # Inject system metadata into params so tools can access them
            # without exposing them to the LLM's tool call JSON.
            enriched = {**tool_input.params}
            if tool_input.run_id:
                enriched["_run_id"] = tool_input.run_id
            return await tool.execute(enriched)
        except Exception as exc:
            return ToolResult.from_error(tool_input.tool_name, str(exc))
