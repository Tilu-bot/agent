"""Cross-run memory store.

After each completed run, the MemoryStore asks the LLM to distil key
learnings (tools that worked, facts found, patterns discovered) into a short
paragraph and persists it in the ``memories`` SQLite table.

Before planning, the Planner calls ``recall()`` to get the most relevant past
learnings and injects them as context so the agent improves over time.

Recall strategy (in priority order):
1. **Embedding similarity** — cosine similarity between the goal embedding and
   each stored ``goal_summary`` embedding, when the configured embedding model
   is reachable.  This gives true semantic matching (e.g. "write Python code"
   matches "implement a Python script").
2. **Word-overlap** (fallback) — simple keyword intersection used when the
   embedding model is unavailable.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from sqlalchemy import select

from backend.config import get_config
from backend.llm.ollama_client import OllamaClient
from backend.llm.router import ModelRouter, TaskType
from backend.models.database import session_scope
from backend.models.db import Memory


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class MemoryStore:
    """Persists and retrieves cross-run learnings."""

    _SUMMARISE_SYSTEM = (
        "You are an agent memory recorder. Given a completed run, summarise the key learnings "
        "in 2-3 sentences. Focus on: which tools worked, what facts were discovered, and any "
        "strategy that was effective. Be specific and concise. Do NOT repeat the goal verbatim."
    )

    def __init__(self, router: ModelRouter) -> None:
        self._router = router
        self._cfg = get_config()
        self._ollama = OllamaClient()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def recall(self, goal: str, limit: int = 3) -> list[str]:
        """Return up to ``limit`` past learnings most relevant to this goal.

        Uses embedding-based cosine similarity when the embedding model is
        available; falls back to keyword overlap otherwise.
        """
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(Memory).order_by(Memory.created_at.desc()).limit(limit * 10)
                )
                memories = result.scalars().all()

            if not memories:
                return []

            # ── Try embedding-based recall ──────────────────────────────────
            embedding_model = self._cfg.models.embedding
            try:
                goal_vec = await self._ollama.embed(embedding_model, goal)
                if goal_vec:
                    scored: list[tuple[float, Memory]] = []
                    for m in memories:
                        try:
                            mem_vec = await self._ollama.embed(embedding_model, m.goal_summary)
                            sim = _cosine(goal_vec, mem_vec)
                        except Exception:
                            sim = 0.0
                        scored.append((sim, m))
                    scored.sort(key=lambda x: -x[0])
                    return [m.learnings for _, m in scored[:limit]]
            except Exception:
                pass  # Embedding model not available — fall through to keyword overlap

            # ── Keyword-overlap fallback ────────────────────────────────────
            goal_words = set(goal.lower().split())
            kw_scored: list[tuple[int, Memory]] = []
            for m in memories:
                summary_words = set(m.goal_summary.lower().split())
                score = len(goal_words & summary_words)
                kw_scored.append((score, m))

            kw_scored.sort(key=lambda x: -x[0])
            return [m.learnings for _, m in kw_scored[:limit]]
        except Exception:
            return []

    async def record_run(
        self,
        run_id: str,
        goal: str,
        completed: dict[str, Any],
    ) -> None:
        """Summarise a completed run and persist the learnings.

        Failures are silently swallowed so that memory errors never affect the
        outcome of a run.
        """
        if not completed:
            return
        try:
            results_text = "\n".join(
                f"- {str(result)[:300]}"
                for result in completed.values()
                if result
            )
            if not results_text.strip():
                return

            messages = [
                {"role": "system", "content": self._SUMMARISE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Goal: {goal}\n\n"
                        f"Task results:\n{results_text}"
                    ),
                },
            ]
            learnings = await self._router.chat(TaskType.fast, messages)
            if not learnings.strip():
                return

            async with session_scope() as session:
                memory = Memory(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    goal_summary=goal[:256],
                    learnings=learnings.strip(),
                )
                session.add(memory)
        except Exception:
            pass
