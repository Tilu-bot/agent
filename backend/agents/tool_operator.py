from __future__ import annotations

import json
import uuid
from typing import Any

from backend.llm.router import ModelRouter, TaskType
from backend.models.db import EventKind, Task, TaskStatus
from backend.tools.bus import ToolBus, ToolInput


class ToolOperator:
    """Executes tool calls, normalises outputs, and logs them."""

    SYSTEM_PROMPT = (
        "You are a tool operator agent. Given a task, decide which tool to call "
        "and what parameters to pass.\n"
        "Available tools: filesystem.read, filesystem.write, web.fetch, shell.exec\n"
        "Return ONLY valid JSON:\n"
        '{"tool": "<tool_name>", "params": { ... }}\n'
        "Or if no tool is needed:\n"
        '{"tool": null, "result": "<direct answer>"}'
    )

    def __init__(self, router: ModelRouter, tool_bus: ToolBus):
        self._router = router
        self._bus = tool_bus

    async def execute_task(
        self, task: Task, context: str = ""
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task.title}\n"
                    f"Description: {task.description or ''}\n"
                    f"Context: {context}"
                ),
            },
        ]
        raw = await self._router.chat(TaskType.fast, messages)
        tool_call = self._parse_tool_call(raw)

        if tool_call.get("tool") is None:
            return {
                "kind": "direct",
                "result": tool_call.get("result", raw),
                "tool_call": None,
                "tool_result": None,
            }

        tool_input = ToolInput(
            tool_name=tool_call["tool"],
            params=tool_call.get("params", {}),
        )
        tool_result = await self._bus.call(tool_input)
        return {
            "kind": "tool",
            "result": tool_result.output if tool_result.success else tool_result.error,
            "tool_call": tool_call,
            "tool_result": tool_result.to_dict(),
        }

    def _parse_tool_call(self, raw: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {"tool": None, "result": raw}
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return {"tool": None, "result": raw}
