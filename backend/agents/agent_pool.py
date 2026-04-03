"""Agent pool with task bidding.

Instead of fixed role → model mapping, the pool holds a collection of
:class:`AgentSpec` objects — each representing a named agent with a
specific model, capability set, and bidding logic.

When a task needs to be executed the pool runs a **task auction**: every
agent in the pool is asked to bid on the task (score 0-100).  The
highest-scoring agent "claims" the task and executes it with the model it
is expert in.

Bid scoring
-----------
An agent's bid for a task is computed from three signals:

1. **Role match** (40 pts) — does the task's ``agent_role`` match this
   agent's primary role?
2. **Capability match** (40 pts) — does the task's role map to any of this
   agent's capability tags?
3. **Keyword match** (20 pts) — do keywords in the task title/description
   appear in this agent's description?

Ties are broken by agent name (alphabetical) so results are deterministic.

Dynamic agents
--------------
The pool is built dynamically at run start from the model registry.  Every
discovered model becomes one or more agents: e.g. a model with
``{"code", "fast"}`` tags produces a "coder" agent and a "fast" agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.models.db import Task


# ---------------------------------------------------------------------------
# Capability → agent_role mapping
# ---------------------------------------------------------------------------

# Maps model capability tag → list of agent roles that capability supports.
_CAP_TO_ROLES: dict[str, list[str]] = {
    "code": ["coder", "data_scientist", "tool_operator"],
    "math": ["mathematician", "data_scientist", "analyst"],
    "vision": ["analyst", "researcher"],
    "reasoning": ["analyst", "writer", "critic", "summarizer", "verifier"],
    "embedding": [],          # embedding models don't execute tasks
    "fast": ["tool_operator", "summarizer", "verifier"],
    "search": ["researcher", "tool_operator"],
}

# Maps agent_role → the best capability tag for that role.
_ROLE_TO_CAP: dict[str, str] = {
    "coder": "code",
    "data_scientist": "code",
    "mathematician": "math",
    "researcher": "search",
    "analyst": "reasoning",
    "writer": "reasoning",
    "summarizer": "fast",
    "verifier": "fast",
    "tool_operator": "fast",
    "critic": "reasoning",
}

# Task title/description keywords that hint at a specific capability.
_KEYWORD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(code|program|implement|function|class|debug|script|unit test)\b", re.I), "code"),
    (re.compile(r"\b(math|equation|proof|calculate|formula|numerical|integral|derivative)\b", re.I), "math"),
    (re.compile(r"\b(image|photo|picture|vision|screenshot|diagram)\b", re.I), "vision"),
    (re.compile(r"\b(search|fetch|web|browse|crawl|scrape|look up|find online)\b", re.I), "search"),
    (re.compile(r"\b(analyz|reason|evaluat|assess|compare|review|summariz|synthesiz)\b", re.I), "reasoning"),
]


# ---------------------------------------------------------------------------
# AgentSpec
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """A named agent with a specific model and set of capabilities.

    Attributes
    ----------
    name:
        Human-readable agent name (e.g. "coder-qwen2.5").
    model:
        The Ollama/llama.cpp/HF model this agent uses.
    capabilities:
        Set of capability tags this agent is proficient in (e.g. {"code", "fast"}).
    primary_role:
        The ``agent_role`` string this agent is primarily designed for.
    backend:
        Which LLM backend hosts this model ("ollama", "llamacpp", "hf").
    """

    name: str
    model: str
    capabilities: set[str]
    primary_role: str
    backend: str = "ollama"
    description: str = field(default="")

    def bid(self, task: "Task") -> int:
        """Return a bid score 0-100 for *task*.

        Higher = more confident the agent can handle the task well.
        """
        score = 0
        task_role = (task.agent_role or "").lower()
        task_text = f"{task.title} {task.description or ''}".lower()

        # 1. Role match (40 pts)
        if task_role == self.primary_role:
            score += 40
        elif task_role in {r for cap in self.capabilities for r in _CAP_TO_ROLES.get(cap, [])}:
            score += 20

        # 2. Capability match via role mapping (40 pts)
        required_cap = _ROLE_TO_CAP.get(task_role, "")
        if required_cap and required_cap in self.capabilities:
            score += 40
        elif any(c in self.capabilities for c in ("reasoning", "fast")):
            score += 10   # general fallback

        # 3. Keyword match in task text (20 pts)
        matched_caps: set[str] = set()
        for pattern, cap in _KEYWORD_PATTERNS:
            if pattern.search(task_text):
                matched_caps.add(cap)
        if matched_caps & self.capabilities:
            score += 20

        return min(score, 100)


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------

class AgentPool:
    """Pool of agent specs.  Runs task auctions to assign the best agent.

    Build the pool from the model registry at run start::

        registry = await get_registry()
        pool = AgentPool.from_registry(registry)

        winner = pool.auction(task)     # → AgentSpec
    """

    def __init__(self, agents: list[AgentSpec]) -> None:
        # Filter out embedding-only agents (they can't execute tasks)
        self._agents = [a for a in agents if "embedding" not in a.capabilities or len(a.capabilities) > 1]

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_registry(cls, registry) -> "AgentPool":
        """Build an :class:`AgentPool` from a live :class:`ModelRegistry`.

        One :class:`AgentSpec` is created per (model, primary_role) combination
        so that models with multiple capabilities produce multiple agents — e.g.
        a math/reasoning model gets both a "mathematician" agent and an
        "analyst" agent, allowing the auction to pick the right specialisation.
        """
        from backend.llm.model_registry import ModelProfile  # local import

        agents: list[AgentSpec] = []
        seen: set[tuple[str, str]] = set()

        for profile in registry.profiles():
            caps = profile.capabilities
            backend = profile.backend

            # Determine which primary roles this model can serve
            primary_roles: list[str] = []
            for cap in caps:
                for role in _CAP_TO_ROLES.get(cap, []):
                    if (profile.name, role) not in seen:
                        primary_roles.append(role)
                        seen.add((profile.name, role))

            if not primary_roles:
                # Fallback: general agent
                primary_roles = ["tool_operator"]

            for role in primary_roles:
                cap_label = "/".join(sorted(caps - {"fast", "search"})) or "general"
                spec = AgentSpec(
                    name=f"{role}-{profile.name}",
                    model=profile.name,
                    capabilities=set(caps),
                    primary_role=role,
                    backend=backend,
                    description=f"Expert at {cap_label} tasks using {profile.name}",
                )
                agents.append(spec)

        if not agents:
            # Safety net: no models discovered → placeholder agent
            agents.append(AgentSpec(
                name="fallback-agent",
                model="",
                capabilities={"fast", "reasoning"},
                primary_role="tool_operator",
            ))

        return cls(agents)

    # ── Auction ────────────────────────────────────────────────────────────────

    def auction(self, task: "Task") -> AgentSpec:
        """Run a task auction and return the winning :class:`AgentSpec`.

        All agents bid concurrently (synchronously — bids are pure Python).
        The agent with the highest bid wins.  Ties are broken by agent name
        so results are deterministic for the same pool/task combination.
        """
        if not self._agents:
            raise RuntimeError("AgentPool is empty — no models available.")

        bids: list[tuple[int, str, AgentSpec]] = [
            (a.bid(task), a.name, a) for a in self._agents
        ]
        bids.sort(key=lambda x: (-x[0], x[1]))
        winner = bids[0][2]
        return winner

    def auction_with_scores(self, task: "Task") -> list[tuple[int, AgentSpec]]:
        """Return all bids sorted highest first (useful for logging)."""
        bids = [(a.bid(task), a) for a in self._agents]
        bids.sort(key=lambda x: (-x[0], x[1].name))
        return bids

    def agents(self) -> list[AgentSpec]:
        return list(self._agents)

    def __len__(self) -> int:
        return len(self._agents)
