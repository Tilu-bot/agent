"""Reflexion agent — generates corrective guidance after a failed task.

When a task result fails verification (or confidence is too low), the
Reflexion agent is called to analyse *why* it failed and produce a short
correction prompt.  That prompt is then injected into the next ToolOperator
call so the model can self-correct rather than blindly repeating the same
approach.

Design rationale
----------------
Inspired by the Reflexion paper (Shinn et al., 2023), which showed that
feeding verbal reflections of failures back into the context window improves
task success rates significantly without any weight updates.

The agent intentionally produces *concise* corrections (≤ 3 sentences) to
avoid context-window bloat across multiple retry cycles.
"""

from __future__ import annotations

from backend.llm.router import ModelRouter, TaskType
from backend.models.db import Task

_SYSTEM_PROMPT = (
    "You are a task correction advisor. A previous attempt at a task produced "
    "a result that was judged insufficient by a verification agent.\n\n"
    "Your job is to analyse the failure and give a clear, actionable correction "
    "in at most 3 sentences.\n"
    "Focus on:\n"
    "  • What went wrong (wrong tool, wrong parameters, wrong approach)?\n"
    "  • What should the agent try instead?\n"
    "  • Any concrete detail that would help (different URL, different query, etc.)?\n\n"
    "Be specific and direct. Do NOT repeat the original task description."
)


class ReflexionAgent:
    """Analyses a failed task and returns a corrective prompt string."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def reflect(
        self,
        task: Task,
        result: str,
        verification: dict,
    ) -> str:
        """Return a short correction string to guide the next retry attempt.

        Parameters
        ----------
        task:
            The task that failed.
        result:
            The (unsatisfactory) result string from the previous attempt.
        verification:
            The verifier's JSON dict: ``{"verified": bool, "confidence": int,
            "notes": str}``.

        Returns
        -------
        str
            A correction prompt; empty string on any error (so callers can
            safely concatenate without guarding).
        """
        notes = verification.get("notes", "No notes provided.")
        confidence = verification.get("confidence", 0)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task.title}\n"
                    f"Description: {task.description or ''}\n\n"
                    f"Previous result (confidence {confidence}/100):\n{result[:1000]}\n\n"
                    f"Verifier notes: {notes}"
                ),
            },
        ]
        try:
            correction = await self._router.chat(TaskType.reasoning, messages)
            return correction.strip()
        except Exception:
            return ""
