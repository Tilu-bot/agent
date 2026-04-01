from __future__ import annotations

import json
from typing import Any

from backend.llm.router import ModelRouter, TaskType
from backend.models.db import Task


class Verifier:
    """Checks task results for evidence and marks them verified/unverified."""

    SYSTEM_PROMPT = (
        "You are a verification agent. Your job is to evaluate whether a task result "
        "is adequately supported by evidence.\n"
        "Evidence means: tool outputs (web fetch, file read, shell exec), citations, or test results.\n"
        "Return ONLY valid JSON:\n"
        '{"verified": true/false, "confidence": 0-100, "notes": "..."}\n'
        "Be strict: if the result is a bare assertion with no evidence, mark verified=false."
    )

    def __init__(self, router: ModelRouter):
        self._router = router

    async def verify(
        self,
        task: Task,
        result: str,
        tool_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_block = ""
        if tool_result:
            evidence_block = f"\nTool used: {tool_result.get('tool_name')}\nTool output: {json.dumps(tool_result.get('output'))[:2000]}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task.title}\n"
                    f"Result: {result}{evidence_block}"
                ),
            },
        ]
        raw = await self._router.chat(TaskType.fast, messages)
        return self._parse(raw)

    def _parse(self, raw: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {"verified": False, "confidence": 0, "notes": raw}
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return {"verified": False, "confidence": 0, "notes": raw}
