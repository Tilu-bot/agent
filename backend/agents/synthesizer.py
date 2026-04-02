"""Synthesizer agent — produces a single coherent final answer from all task results.

After all tasks in a run complete, the Synthesizer reads every task result
and produces one concise, well-structured final answer that directly addresses
the original goal.  This answer is stored in ``Run.summary`` and emitted as a
``synthesis`` event on the SSE stream so users get a clean take-away without
having to read every individual task result.

Design rationale
----------------
Individual task results are often fragmented: one task fetches raw web
content, another writes a file, another summarises something.  A synthesis
step mirrors what a human reviewer would do — read everything and produce a
unified answer — making the system's output far more useful.
"""

from __future__ import annotations

from backend.llm.router import ModelRouter, TaskType
from backend.models.db import Task

_SYSTEM_PROMPT = (
    "You are a synthesis agent. You have received the results of all tasks "
    "that were executed to accomplish a goal.\n\n"
    "Your job is to combine these results into a single, coherent, well-structured "
    "final answer that directly and completely addresses the original goal.\n\n"
    "Guidelines:\n"
    "  • Be comprehensive but concise — cover all key findings.\n"
    "  • Use clear prose; bullet points are fine for lists of facts.\n"
    "  • Do NOT say 'Based on the task results...' — just deliver the answer.\n"
    "  • If some tasks produced no useful output, ignore them silently.\n"
    "  • If the goal was a question, answer it directly at the start."
)


class Synthesizer:
    """Reads all completed task results and produces a unified final answer."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def synthesize(
        self,
        goal: str,
        tasks: list[Task],
    ) -> str:
        """Return a synthesized final answer for *goal* from the task results.

        Parameters
        ----------
        goal:
            The original high-level goal of the run.
        tasks:
            All completed ``Task`` objects (with populated ``.result``).

        Returns
        -------
        str
            The synthesized answer; empty string on error.
        """
        results_block = "\n\n".join(
            f"[{t.title}]\n{(t.result or '').strip()[:800]}"
            for t in tasks
            if t.result and t.result.strip()
        )

        if not results_block.strip():
            return ""

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Task results:\n{results_block}"
                ),
            },
        ]
        try:
            answer = await self._router.chat(TaskType.reasoning, messages)
            return answer.strip()
        except Exception:
            return ""
