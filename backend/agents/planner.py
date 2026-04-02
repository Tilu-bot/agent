from __future__ import annotations

import json
import uuid
from typing import Any

from backend.llm.json_utils import extract_json
from backend.llm.router import ModelRouter, TaskType
from backend.models.db import Task


class Planner:
    """Converts a high-level goal into a task DAG.

    Improvements over the baseline:
    * Receives the list of available tools so it can create tasks that map
      cleanly onto real capabilities.
    * Receives ``memories`` — short summaries of learnings from past runs —
      so the planner can avoid strategies that failed before and reuse ones
      that worked.
    """

    _BASE_SYSTEM = (
        "You are an expert project planner. Given a goal, produce a structured task plan.\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "tasks": [\n'
        '    {"id": "t1", "title": "...", "description": "...", "agent_role": "...", "depends_on": []},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "agent_role must be one of:\n"
        "  researcher      — web search, information gathering\n"
        "  coder           — writing or debugging code\n"
        "  mathematician   — symbolic math, proofs, numerical analysis\n"
        "  data_scientist  — data analysis, statistics, pandas/numpy operations\n"
        "  analyst         — reasoning over gathered data to draw conclusions\n"
        "  writer          — drafting prose, reports, summaries, documentation\n"
        "  summarizer      — condensing large inputs into concise summaries\n"
        "  tool_operator   — direct tool calls (files, shell, APIs)\n"
        "  verifier        — checking results for correctness\n"
        "  critic          — evaluating quality and completeness of outputs\n"
        "depends_on is a list of task ids that must complete before this task.\n"
        "Keep the plan minimal and actionable. Maximum 10 tasks."
    )

    def __init__(self, router: ModelRouter):
        self._router = router

    async def create_plan(
        self,
        goal: str,
        run_id: str,
        available_tools: list[str] | None = None,
        memories: list[str] | None = None,
    ) -> list[Task]:
        system = self._build_system_prompt(available_tools, memories)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {goal}"},
        ]
        raw = await self._router.chat(TaskType.reasoning, messages, format="json")
        tasks = self._parse_plan(raw, run_id)
        return tasks

    # ── Internal helpers ───────────────────────────────────────────────────────

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

    def _parse_plan(self, raw: str, run_id: str) -> list[Task]:
        # Extract JSON from the response
        data = extract_json(raw, default={})
        if not isinstance(data, dict):
            data = {}
        raw_tasks: list[dict[str, Any]] = data.get("tasks", [])

        if not raw_tasks:
            return [
                Task(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    title="Execute goal",
                    description="No structured plan was generated; executing goal directly.",
                    agent_role="tool_operator",
                    depends_on=[],
                )
            ]

        # Build a mapping from planner IDs to real UUIDs
        id_map: dict[str, str] = {}
        for rt in raw_tasks:
            real_id = str(uuid.uuid4())
            id_map[rt.get("id", real_id)] = real_id

        tasks = []
        for rt in raw_tasks:
            planner_id = rt.get("id", "")
            real_id = id_map.get(planner_id, str(uuid.uuid4()))
            depends = [id_map[dep] for dep in rt.get("depends_on", []) if dep in id_map]
            tasks.append(
                Task(
                    id=real_id,
                    run_id=run_id,
                    title=rt.get("title", "Unnamed task"),
                    description=rt.get("description", ""),
                    agent_role=rt.get("agent_role", "tool_operator"),
                    depends_on=depends,
                )
            )
        return tasks
