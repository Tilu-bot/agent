"""Critic agent — evaluates the quality of the synthesised final answer.

After the Synthesizer produces a final answer the Critic reads both the
original goal and the synthesis and produces a structured quality assessment.
This catches cases where the synthesis is plausible-sounding but incomplete,
off-topic, or internally inconsistent — things a human reviewer would notice
but that the Synthesizer cannot catch in its own output.

The quality score (0–100) and a list of specific issues are returned as a
dict and emitted as a ``critic`` event on the SSE stream.  If quality is
below the configured threshold the issues are surfaced to the user without
blocking the run — the run still completes successfully.

Design rationale
----------------
Inspired by Constitutional AI and self-critique approaches.  A separate
critic call using a fresh context (no knowledge of how the answer was
produced) can catch errors that "chain of thought" produces but cannot
self-detect.  The critic deliberately uses the *reasoning* model tier rather
than ``fast`` so it has enough capacity to reason carefully.
"""

from __future__ import annotations

import json

from backend.llm.json_utils import extract_json
from backend.llm.router import ModelRouter, TaskType

_SYSTEM_PROMPT = (
    "You are a quality-assurance critic reviewing an AI-generated answer.\n\n"
    "Your task:\n"
    "1. Read the original goal and the generated answer.\n"
    "2. Assess whether the answer fully, correctly, and completely addresses the goal.\n"
    "3. Return ONLY valid JSON in this exact format:\n"
    '{"quality": 0-100, "passed": true/false, "issues": ["issue 1", "issue 2", ...]}\n\n'
    "Scoring guide:\n"
    "  90-100  Excellent — complete, accurate, directly addresses the goal.\n"
    "  70-89   Good — mostly complete; minor gaps or imprecision.\n"
    "  50-69   Partial — meaningful content but significant gaps or errors.\n"
    "  0-49    Poor — wrong, irrelevant, or dangerously incomplete.\n\n"
    "passed is true when quality ≥ 70.\n"
    "issues is an empty list when passed is true.\n"
    "Be concise. Each issue should be a single sentence."
)


class CriticAgent:
    """Evaluates the quality of a synthesis output against the original goal."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def evaluate(self, goal: str, synthesis: str) -> dict:
        """Return a quality assessment dict.

        Parameters
        ----------
        goal:
            The original high-level goal of the run.
        synthesis:
            The final answer produced by the Synthesizer.

        Returns
        -------
        dict with keys:
            quality (int 0-100), passed (bool), issues (list[str])
        """
        if not synthesis.strip():
            return {"quality": 0, "passed": False, "issues": ["No answer was generated."]}

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original goal: {goal}\n\n"
                    f"Generated answer:\n{synthesis[:3000]}"
                ),
            },
        ]
        try:
            raw = await self._router.chat(TaskType.reasoning, messages, format="json")
            return self._parse(raw)
        except Exception:
            return {"quality": 0, "passed": False, "issues": ["Critic evaluation failed."]}

    def _parse(self, raw: str) -> dict:
        data = extract_json(raw, default={})
        if not isinstance(data, dict):
            return {"quality": 0, "passed": False, "issues": [raw.strip()[:200]]}
        try:
            quality = int(data.get("quality", 0))
            passed = bool(data.get("passed", quality >= 70))
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]
            return {"quality": quality, "passed": passed, "issues": issues}
        except (ValueError, TypeError):
            return {"quality": 0, "passed": False, "issues": [raw.strip()[:200]]}
