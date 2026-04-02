"""Debater agent — generates N independent candidate plans for a goal.

Each candidate is produced by calling the LLM with ``temperature > 0`` so
that the outputs are meaningfully diverse.  The raw JSON strings are returned
to the caller (Voter) without further parsing so that all N plans can be
compared in a single Voter LLM call.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend.llm.router import ModelRouter, TaskType


class Debater:
    """Generates multiple candidate plans from the same goal."""

    _BASE_SYSTEM = (
        "You are an expert project planner. Given a goal, produce a structured task plan.\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "tasks": [\n'
        '    {"id": "t1", "title": "...", "description": "...", "agent_role": "...", "depends_on": []},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "agent_role must be one of: researcher, coder, analyst, writer, tool_operator, verifier.\n"
        "depends_on is a list of task ids that must complete before this task.\n"
        "Keep the plan minimal and actionable. Maximum 8 tasks."
    )

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def generate_candidates(
        self,
        goal: str,
        n: int = 3,
        temperature: float = 0.8,
        available_tools: list[str] | None = None,
        memories: list[str] | None = None,
    ) -> list[str]:
        """Return *n* raw plan JSON strings, each generated independently.

        All candidates are requested concurrently for speed.
        """
        system = self._build_system_prompt(available_tools, memories)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {goal}"},
        ]
        options: dict[str, Any] = {"temperature": temperature}

        tasks = [
            self._router.chat(TaskType.reasoning, messages, options=options)
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[str] = []
        for r in results:
            if isinstance(r, BaseException):
                continue
            candidates.append(str(r))
        return candidates

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        available_tools: list[str] | None,
        memories: list[str] | None,
    ) -> str:
        parts = [self._BASE_SYSTEM]
        if available_tools:
            tool_list = ", ".join(available_tools)
            parts.append(
                f"\nAvailable tools: {tool_list}\n"
                "Design each task so it maps to one of these tools or can be answered directly by the LLM."
            )
        if memories:
            memory_block = "\n".join(f"  - {m}" for m in memories)
            parts.append(
                f"\nLearnings from past similar runs (use these to plan more effectively):\n{memory_block}"
            )
        return "\n".join(parts)
