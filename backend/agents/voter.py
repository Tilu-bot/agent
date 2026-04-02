"""Voter agent — compares N candidate plans and selects the best one.

The Voter calls the LLM with all candidates in a single prompt and asks it
to choose the index (0-based) of the plan that is most likely to succeed.
The winning plan is then parsed into Task objects using the same logic as
the Planner.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from backend.llm.json_utils import extract_json
from backend.llm.router import ModelRouter, TaskType
from backend.models.db import Task


class Voter:
    """Selects the best plan from a list of candidates."""

    _SYSTEM_PROMPT = (
        "You are a plan evaluation agent. You will be given several candidate task plans "
        "for the same goal. Evaluate each plan for:\n"
        "  1. Completeness — does it fully address the goal?\n"
        "  2. Efficiency   — is the number of tasks appropriate (no unnecessary steps)?\n"
        "  3. Feasibility  — can each task realistically be executed?\n"
        "  4. Dependencies — are task dependencies sensible and acyclic?\n\n"
        "Return ONLY valid JSON:\n"
        '{"winner": <0-based index of the best plan>, "reason": "<one sentence>"}\n'
        "Do not include any other text."
    )

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def select_best(
        self,
        goal: str,
        run_id: str,
        candidates: list[str],
    ) -> list[Task]:
        """Pick the best plan from *candidates* and return it as Task objects.

        Falls back to the first candidate if the LLM response cannot be parsed
        or if *candidates* is empty.
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            return self._parse_plan(candidates[0], run_id)

        # Build the voting prompt
        candidate_block = "\n\n".join(
            f"--- Plan {i} ---\n{c}" for i, c in enumerate(candidates)
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Candidate plans:\n\n{candidate_block}"
                ),
            },
        ]
        raw = await self._router.chat(TaskType.fast, messages, format="json")
        winner_index = self._parse_winner(raw, len(candidates))
        return self._parse_plan(candidates[winner_index], run_id)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _parse_winner(self, raw: str, num_candidates: int) -> int:
        data = extract_json(raw, default={})
        if isinstance(data, dict):
            try:
                index = int(data.get("winner", 0))
                if 0 <= index < num_candidates:
                    return index
            except (ValueError, TypeError):
                pass
        return 0

    def _parse_plan(self, raw: str, run_id: str) -> list[Task]:
        """Parse a raw plan JSON string into Task objects (mirrors Planner logic)."""
        data = extract_json(raw, default={})
        if not isinstance(data, dict):
            data = {}
        raw_tasks: list[dict[str, Any]] = data.get("tasks", [])

        if not raw_tasks:
            return [self._fallback_task(run_id)]

        id_map: dict[str, str] = {}
        for rt in raw_tasks:
            real_id = str(uuid.uuid4())
            id_map[rt.get("id", real_id)] = real_id

        tasks: list[Task] = []
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

    @staticmethod
    def _fallback_task(run_id: str) -> Task:
        return Task(
            id=str(uuid.uuid4()),
            run_id=run_id,
            title="Execute goal",
            description="Could not select a plan; executing goal directly.",
            agent_role="tool_operator",
            depends_on=[],
        )
