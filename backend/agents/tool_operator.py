from __future__ import annotations

import json
from typing import Any

from backend.llm.router import ModelRouter, TaskType
from backend.models.db import Task
from backend.tools.bus import ToolBus, ToolInput

_MAX_RETRIES = 3

# Map agent roles to the most appropriate model type.
# Coders get the code-specialised model; researchers and analysts get the
# reasoning model; everything else uses the fast model.
_ROLE_TASK_TYPE: dict[str, TaskType] = {
    "coder": TaskType.code,
    "researcher": TaskType.reasoning,
    "analyst": TaskType.reasoning,
    "writer": TaskType.reasoning,
    "verifier": TaskType.fast,
    "tool_operator": TaskType.fast,
}

_SYSTEM_HEADER = (
    "You are a tool operator agent. Given a task, decide which tool to call "
    "and what parameters to pass.\n"
)

_SYSTEM_FOOTER = (
    "\nReturn ONLY valid JSON:\n"
    '{"tool": "<tool_name>", "params": { ... }}\n'
    "Or if no tool is needed:\n"
    '{"tool": null, "result": "<direct answer>"}\n'
    "If a previous attempt failed, learn from the error and try a different approach."
)


class ToolOperator:
    """Executes tool calls using a ReAct retry loop.

    On a tool failure or unparseable LLM response the error is fed back into
    the conversation so the model can self-correct.  Up to ``_MAX_RETRIES``
    attempts are made before the task is marked failed.

    The model used for a task is selected based on the task's ``agent_role``
    so that coding tasks get the code-specialised model and research/analysis
    tasks get the reasoning model.

    The system prompt is built dynamically from the registered tools so the
    LLM always has accurate, up-to-date descriptions of available capabilities.
    """

    def __init__(self, router: ModelRouter, tool_bus: ToolBus):
        self._router = router
        self._bus = tool_bus

    def _build_system_prompt(self) -> str:
        """Build a rich system prompt from the live tool registry."""
        tool_descriptions = self._bus.describe_tools()
        if tool_descriptions:
            tool_lines = "\n".join(
                f"  • {name}: {desc}" for name, desc in tool_descriptions.items()
            )
            tools_section = f"Available tools:\n{tool_lines}"
        else:
            tools_section = "No tools are currently registered."
        return _SYSTEM_HEADER + tools_section + _SYSTEM_FOOTER

    async def execute_task(
        self, task: Task, context: str = "", reflection: str = ""
    ) -> dict[str, Any]:
        """Execute *task*, optionally guided by a *reflection* correction.

        Parameters
        ----------
        task:
            The task to execute.
        context:
            JSON string of dependency results from previous tasks.
        reflection:
            A corrective prompt from the Reflexion agent (injected after a
            failed verification so the model can self-correct).
        """
        # Choose model based on agent role for better quality
        task_type = _ROLE_TASK_TYPE.get(task.agent_role or "", TaskType.fast)

        user_content = (
            f"Task: {task.title}\n"
            f"Description: {task.description or ''}\n"
            f"Context: {context}"
        )
        if reflection:
            user_content += f"\n\nCorrection guidance: {reflection}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        last_error: str = ""
        tool_call: dict[str, Any] = {}
        tool_result = None
        for attempt in range(1, _MAX_RETRIES + 1):
            raw = await self._router.chat(task_type, messages)
            tool_call = self._parse_tool_call(raw)

            if tool_call.get("tool") is None:
                return {
                    "kind": "direct",
                    "result": tool_call.get("result", raw),
                    "tool_call": None,
                    "tool_result": None,
                    "attempts": attempt,
                }

            tool_input = ToolInput(
                tool_name=tool_call["tool"],
                params=tool_call.get("params", {}),
            )
            tool_result = await self._bus.call(tool_input)

            if tool_result.success:
                return {
                    "kind": "tool",
                    "result": tool_result.output,
                    "tool_call": tool_call,
                    "tool_result": tool_result.to_dict(),
                    "attempts": attempt,
                }

            # Tool failed — feed the error back so the model can retry
            last_error = tool_result.error or "unknown error"
            if attempt < _MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That tool call failed (attempt {attempt}/{_MAX_RETRIES}): "
                        f"{last_error}\n"
                        "Please try a different tool or different parameters."
                    ),
                })

        # All retries exhausted
        return {
            "kind": "tool",
            "result": last_error,
            "tool_call": tool_call,
            "tool_result": tool_result.to_dict() if tool_result is not None else None,
            "attempts": _MAX_RETRIES,
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
