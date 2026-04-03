"""Per-run shared knowledge store.

During a run every agent can read and write facts to this store.  Facts are
immediately visible to all other agents — this is how agents "learn from each
other" mid-run without having to restart.

Contrast with :class:`~backend.agents.memory.MemoryStore` (cross-run
learnings):

* **KnowledgeStore** — scoped to a single run; agents write facts while
  executing tasks; available instantly to all concurrent agents.
* **MemoryStore** — cross-run learnings persisted to the DB; used only at
  plan time to inform the next run's strategy.

Design
------
* Thread-safe (asyncio.Lock).
* Keys are arbitrary strings (e.g. ``"missing_library"``, ``"api_endpoint"``).
* Each key stores the *latest* value plus metadata (source agent, timestamp).
* :meth:`query` does a simple keyword-overlap search so agents can ask "do we
  know anything about scipy?" without knowing the exact key.
* :meth:`context_block` returns a compact text block for injection into task
  prompts so each agent automatically benefits from what others learned.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class KnowledgeFact:
    """A single fact in the store."""

    key: str
    value: str
    source_agent: str
    timestamp: float = field(default_factory=time.monotonic)


class KnowledgeStore:
    """Thread-safe, per-run shared fact store."""

    def __init__(self) -> None:
        self._facts: dict[str, KnowledgeFact] = {}
        self._lock = asyncio.Lock()

    # ── Write ──────────────────────────────────────────────────────────────────

    async def add(self, key: str, value: str, source_agent: str) -> None:
        """Add or update a fact.  Last writer wins."""
        async with self._lock:
            self._facts[key] = KnowledgeFact(
                key=key,
                value=value,
                source_agent=source_agent,
            )

    # ── Read ───────────────────────────────────────────────────────────────────

    async def get(self, key: str) -> KnowledgeFact | None:
        """Return a specific fact, or ``None`` if it doesn't exist."""
        async with self._lock:
            return self._facts.get(key)

    async def query(self, topic: str, limit: int = 5) -> list[KnowledgeFact]:
        """Return facts whose key or value overlaps with *topic* keywords.

        Results are ranked by keyword overlap (higher = more relevant) and
        recency, then capped at *limit*.
        """
        topic_words = set(topic.lower().split())

        async with self._lock:
            scored: list[tuple[int, float, KnowledgeFact]] = []
            for fact in self._facts.values():
                text = f"{fact.key} {fact.value}".lower()
                text_words = set(text.split())
                overlap = len(topic_words & text_words)
                if overlap > 0:
                    scored.append((overlap, fact.timestamp, fact))

        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [f for _, _, f in scored[:limit]]

    async def all_facts(self) -> dict[str, KnowledgeFact]:
        """Return a shallow copy of all stored facts."""
        async with self._lock:
            return dict(self._facts)

    async def context_block(self, max_chars: int = 2000) -> str:
        """Return a compact text block of all facts for injection into prompts.

        The block is truncated to *max_chars* characters if necessary.

        Example output::

            [Shared knowledge from this run]
            missing_library: the project uses scipy 1.12 (from researcher)
            api_endpoint: https://api.example.com/v2 (from coder)
        """
        async with self._lock:
            facts = list(self._facts.values())

        if not facts:
            return ""

        # Sort by recency (newest first for prompt relevance)
        facts.sort(key=lambda f: -f.timestamp)
        lines = ["[Shared knowledge from this run]"]
        for f in facts:
            line = f"  {f.key}: {f.value[:300]} (from {f.source_agent})"
            lines.append(line)

        block = "\n".join(lines)
        if len(block) > max_chars:
            block = block[:max_chars] + "\n  …"
        return block

    def size(self) -> int:
        """Return the number of stored facts (non-blocking)."""
        return len(self._facts)


# ---------------------------------------------------------------------------
# Per-run store registry (keyed by run_id)
# ---------------------------------------------------------------------------

_stores: dict[str, KnowledgeStore] = {}
_stores_lock = asyncio.Lock()


async def get_store(run_id: str) -> KnowledgeStore:
    """Return (creating if necessary) the :class:`KnowledgeStore` for *run_id*."""
    async with _stores_lock:
        if run_id not in _stores:
            _stores[run_id] = KnowledgeStore()
        return _stores[run_id]


async def clear_store(run_id: str) -> None:
    """Remove the store for *run_id* from the registry (called at run end)."""
    async with _stores_lock:
        _stores.pop(run_id, None)


def get_store_sync(run_id: str) -> KnowledgeStore | None:
    """Return the store for *run_id* without blocking (may be ``None``)."""
    return _stores.get(run_id)
