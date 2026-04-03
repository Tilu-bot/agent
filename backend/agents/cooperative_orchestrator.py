"""Cooperative multi-agent orchestrator.

This is the new execution engine that replaces the linear role-based pipeline
with a system where agents *know what they're good at*, *bid on tasks*, and
*help each other* when stuck.

Architecture vs the classic Orchestrator
-----------------------------------------
Classic (``orchestrator.py``)::

    Planner → [task] → ToolOperator(fixed-role model) → Verifier → Reflexion

Cooperative (this module)::

    Planner → [task queue] → Auction (agents bid) → Winner executes
                                ↓ agent stuck?
                            MessageBus.request_help()
                                ↓ peer agents respond
                            Retry with peer knowledge
                                ↓ result ready
                            KnowledgeStore.add(fact)   ← immediately visible to ALL agents
                            Verifier → (Reflexion loop if needed)

Key differences
---------------
1. **Task auction** — each pending task is auctioned to the pool of available
   agents.  The agent with the highest bid (based on model capabilities +
   role match + keyword match) claims the task.

2. **Cooperative help** — when an agent's result fails verification and a
   Reflexion correction is not enough, the agent posts a ``help_request`` to
   the shared :class:`~backend.agents.message_bus.MessageBus`.  Another agent
   in the pool responds using *its* model (e.g. a web-search agent responds
   to a researcher's request about a missing library).

3. **Shared knowledge** — every task result (verified or not) is summarised
   into facts and written to the per-run
   :class:`~backend.agents.knowledge.KnowledgeStore`.  Subsequent tasks
   automatically receive this knowledge as context, so agents build on each
   other's work.

4. **Multi-backend** — the agent pool includes models from Ollama *and*
   llama.cpp servers (discovered automatically), so locally hosted models from
   any source are first-class participants.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.agent_pool import AgentPool, AgentSpec
from backend.agents.critic import CriticAgent
from backend.agents.knowledge import KnowledgeStore, clear_store, get_store
from backend.agents.memory import MemoryStore
from backend.agents.message_bus import AgentMessage, MessageBus
from backend.agents.planner import Planner
from backend.agents.reflexion import ReflexionAgent
from backend.agents.synthesizer import Synthesizer
from backend.agents.verifier import Verifier
from backend.config import get_config
from backend.llm.model_registry import get_registry
from backend.llm.ollama_client import OllamaClient
from backend.llm.router import ModelRouter, TaskType
from backend.models.database import get_session_factory
from backend.models.db import (
    Artifact, Event, EventKind, Run, RunStatus, Task, TaskStatus,
)
from backend.tools.bus import ToolBus, ToolInput
from backend.tools.scratchpad import clear_run_scratchpad

_TERMINAL = {RunStatus.completed, RunStatus.failed, RunStatus.cancelled}


class CooperativeOrchestrator:
    """Runs a goal using cooperative multi-agent task auction and help-requests.

    Drop-in replacement for :class:`~backend.agents.orchestrator.Orchestrator`.
    The constructor signature is identical so ``runs.py`` can switch between
    them based on the ``cooperative.enabled`` config flag.
    """

    def __init__(
        self,
        session: AsyncSession,
        tool_bus: ToolBus,
        run_models: dict[str, str] | None = None,
    ) -> None:
        self._session = session
        self._tool_bus = tool_bus
        self._router = ModelRouter(run_models=run_models)
        self._planner = Planner(self._router)
        self._verifier = Verifier(self._router)
        self._reflexion = ReflexionAgent(self._router)
        self._synthesizer = Synthesizer(self._router)
        self._critic = CriticAgent(self._router)
        self._memory = MemoryStore(self._router)
        self._cfg = get_config()

    # ── Public API (mirrors Orchestrator) ──────────────────────────────────────

    async def run(self, run: Run) -> None:
        """Execute the full run lifecycle with cooperative agents."""
        await self._session.refresh(run)
        if run.status in _TERMINAL:
            return

        run.status = RunStatus.running
        self._session.add(run)
        await self._session.commit()

        max_runtime = self._cfg.server.max_runtime_seconds
        if max_runtime > 0:
            try:
                await asyncio.wait_for(self._run_inner(run), timeout=max_runtime)
            except asyncio.TimeoutError:
                await self._session.refresh(run)
                if run.status not in _TERMINAL:
                    run.status = RunStatus.failed
                    await self._log_event(
                        run.id, EventKind.error, "orchestrator",
                        f"Run exceeded maximum runtime of {max_runtime}s.",
                    )
                    self._session.add(run)
                    await self._session.commit()
        else:
            await self._run_inner(run)

        clear_run_scratchpad(run.id)
        await clear_store(run.id)

    # ── Core execution loop ────────────────────────────────────────────────────

    async def _run_inner(self, run: Run) -> None:
        await self._log_event(run.id, EventKind.agent_message, "orchestrator",
                              f"Starting cooperative run: {run.goal}")

        # Preflight: verify Ollama
        if not await OllamaClient().is_available():
            run.status = RunStatus.failed
            await self._log_event(
                run.id, EventKind.agent_message, "orchestrator",
                "Ollama is not running. Start with: `ollama serve`, "
                "then pull a model: `ollama pull llama3.2:3b`",
            )
            self._session.add(run)
            await self._session.commit()
            return

        try:
            # ── 0. Build model registry + agent pool ──────────────────────────
            registry = await get_registry()
            pool = AgentPool.from_registry(registry)
            model_capabilities = (
                registry.capability_summary()
                if registry.ready and registry.available_models()
                else None
            )
            await self._log_event(
                run.id, EventKind.thinking, "model_registry",
                f"Agent pool ready: {len(pool)} specialized agents from "
                f"{len(registry.available_models())} model(s).\n"
                + (model_capabilities or ""),
            )

            # ── 1. Shared knowledge store + message bus for this run ──────────
            knowledge = await get_store(run.id)

            async def _bus_event_logger(msg: AgentMessage) -> None:
                kind_map = {
                    "help_request": EventKind.help_request,
                    "help_response": EventKind.help_response,
                    "knowledge_share": EventKind.knowledge_share,
                    "status": EventKind.agent_message,
                }
                ek = kind_map.get(msg.kind, EventKind.agent_message)
                await self._log_event(
                    run.id, ek, msg.from_agent, msg.content,
                    {"topic": msg.topic, "data": msg.data, "message_id": msg.message_id},
                )

            bus = MessageBus(event_logger=_bus_event_logger)

            # ── 2. Recall past learnings ──────────────────────────────────────
            memories: list[str] = []
            if self._cfg.memory.enabled:
                memories = await self._memory.recall(
                    run.goal, limit=self._cfg.memory.recall_limit
                )

            # ── 3. Plan (capability-aware) ────────────────────────────────────
            tool_names = self._tool_bus.list_tools()
            await self._log_event(
                run.id, EventKind.thinking, "planner",
                "Analyzing goal and building task plan…",
            )
            tasks = await self._planner.create_plan(
                run.goal, run.id,
                available_tools=tool_names,
                memories=memories if memories else None,
                model_capabilities=model_capabilities,
            )
            if (
                len(tasks) == 1
                and tasks[0].agent_role == "tool_operator"
                and tasks[0].title == "Execute goal"
            ):
                await self._log_event(
                    run.id, EventKind.agent_message, "planner",
                    "Initial plan empty; retrying…",
                )
                tasks = await self._planner.create_plan(
                    run.goal, run.id,
                    available_tools=tool_names,
                    memories=memories if memories else None,
                    model_capabilities=model_capabilities,
                )

            for task in tasks:
                self._session.add(task)
            await self._session.commit()
            await self._log_event(run.id, EventKind.plan_created, "planner",
                                  f"Plan created: {len(tasks)} tasks",
                                  {"task_ids": [t.id for t in tasks]})

            # ── 4. Execute tasks in waves with cooperative agents ──────────────
            completed: dict[str, Any] = {}
            waves = self._topological_waves(tasks)

            for wave in waves:
                await self._session.refresh(run)
                if run.status == RunStatus.cancelled:
                    for task in wave:
                        task.status = TaskStatus.skipped
                        self._session.add(task)
                    await self._session.commit()
                    continue

                await self._execute_wave(run.id, [t.id for t in wave], completed, pool, bus, knowledge)

            # ── 5. Record learnings ───────────────────────────────────────────
            await self._session.refresh(run)
            if run.status not in _TERMINAL and self._cfg.memory.enabled:
                await self._memory.record_run(run.id, run.goal, completed)

            # ── 6. Synthesize ─────────────────────────────────────────────────
            await self._session.refresh(run)
            synthesis = ""
            if run.status not in _TERMINAL and self._cfg.synthesis.enabled:
                all_tasks_result = await self._session.execute(
                    select(Task).where(
                        Task.run_id == run.id,
                        Task.status == TaskStatus.completed,
                    )
                )
                completed_tasks = list(all_tasks_result.scalars().all())
                if completed_tasks:
                    synthesis = await self._synthesizer.synthesize(run.goal, completed_tasks)
                    if synthesis:
                        run.summary = synthesis
                        await self._log_event(run.id, EventKind.synthesis, "synthesizer", synthesis)

            # ── 7. Critic ─────────────────────────────────────────────────────
            if synthesis and self._cfg.critic.enabled:
                critique = await self._critic.evaluate(run.goal, synthesis)
                await self._log_event(
                    run.id, EventKind.critic, "critic",
                    f"Quality={critique['quality']}/100 passed={critique['passed']}",
                    critique,
                )

            # ── 8. Mark complete ──────────────────────────────────────────────
            await self._session.refresh(run)
            if run.status not in _TERMINAL:
                run.status = RunStatus.completed
                await self._log_event(run.id, EventKind.agent_message, "orchestrator",
                                      "All tasks completed successfully.")

            run.token_usage = self._router.token_stats()

        except Exception as exc:
            await self._session.refresh(run)
            if run.status not in _TERMINAL:
                run.status = RunStatus.failed
                await self._log_event(run.id, EventKind.error, "orchestrator",
                                      f"Run failed: {exc}")
        finally:
            self._session.add(run)
            await self._session.commit()

    # ── Wave execution ─────────────────────────────────────────────────────────

    async def _execute_wave(
        self,
        run_id: str,
        task_ids: list[str],
        completed: dict[str, Any],
        pool: AgentPool,
        bus: MessageBus,
        knowledge: KnowledgeStore,
    ) -> None:
        if len(task_ids) == 1:
            task = await self._session.get(Task, task_ids[0])
            run = await self._session.get(Run, run_id)
            if task and run:
                await self._execute_task(run, task, completed, pool, bus, knowledge, self._session)
                completed[task.id] = task.result
            return

        max_parallel = self._cfg.memory.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)
        factory = get_session_factory()

        async def _run_one(task_id: str) -> tuple[str, str | None]:
            async with semaphore:
                async with factory() as s:
                    task = await s.get(Task, task_id)
                    run = await s.get(Run, run_id)
                    if task is None or run is None:
                        return task_id, None
                    await self._execute_task(run, task, completed, pool, bus, knowledge, s)
                    return task_id, task.result

        results = await asyncio.gather(
            *[_run_one(tid) for tid in task_ids],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                continue
            t_id, t_result = result  # type: ignore[misc]
            completed[t_id] = t_result

    # ── Single task execution ──────────────────────────────────────────────────

    async def _execute_task(
        self,
        run: Run,
        task: Task,
        completed: dict[str, Any],
        pool: AgentPool,
        bus: MessageBus,
        knowledge: KnowledgeStore,
        session: AsyncSession,
    ) -> None:
        task.status = TaskStatus.running
        session.add(task)
        await session.commit()

        # ── Auction: find the best agent for this task ─────────────────────────
        agent = pool.auction(task)
        bids = pool.auction_with_scores(task)
        top3 = [(score, a.name, a.model) for score, a in bids[:3]]
        await self._log_event(
            run.id, EventKind.agent_bid, agent.name,
            f"[{agent.name}] won auction for '{task.title}' "
            f"(bid={bids[0][0]}, model={agent.model})",
            {"winner": agent.name, "model": agent.model, "bid": bids[0][0],
             "top3": top3},
            task_id=task.id, session=session,
        )

        await self._log_event(
            run.id, EventKind.task_started, agent.name,
            f"Starting: {task.title}", task_id=task.id, session=session,
        )

        # ── Build context (deps + shared knowledge) ────────────────────────────
        raw_dep_context = json.dumps(
            {dep: completed.get(dep) for dep in (task.depends_on or [])}
        )
        dep_context = await self._budget_context(raw_dep_context)
        knowledge_context = await knowledge.context_block()

        context = dep_context
        if knowledge_context:
            context += f"\n\n{knowledge_context}"

        try:
            await self._log_event(
                run.id, EventKind.thinking, agent.name,
                f"[{agent.name}] working on: {task.title}"
                + (f" — {task.description}" if task.description else ""),
                task_id=task.id, session=session,
            )

            # ── Execute via a per-agent ModelRouter pinned to agent's model ────
            outcome = await self._execute_with_agent(agent, task, context, run.id)

            # Surface reasoning
            if outcome.get("reasoning"):
                await self._log_event(
                    run.id, EventKind.thinking, agent.name,
                    outcome["reasoning"], task_id=task.id, session=session,
                )

            # Log tool events
            if outcome.get("tool_call"):
                await self._log_event(
                    run.id, EventKind.tool_call, "tool_operator",
                    f"Calling {outcome['tool_call'].get('tool')}",
                    outcome["tool_call"], task_id=task.id, session=session,
                )
            if outcome.get("tool_result"):
                tr = outcome["tool_result"]
                await self._log_event(
                    run.id, EventKind.tool_result, "tool_operator",
                    f"Tool result (success={tr.get('success')})",
                    tr, task_id=task.id, session=session,
                )
                await self._store_artifact(run.id, task.id, tr, session=session)

            # ── Verify ────────────────────────────────────────────────────────
            verification = await self._verifier.verify(
                task, str(outcome.get("result", "")), outcome.get("tool_result"),
            )
            await self._log_event(
                run.id, EventKind.verification, "verifier",
                f"Verified={verification.get('verified')} "
                f"confidence={verification.get('confidence')}",
                verification, task_id=task.id, session=session,
            )

            # ── Reflexion + cooperative help loop ─────────────────────────────
            cfg = self._cfg
            for attempt in range(cfg.reflexion.max_retries):
                verified_ok = (
                    verification.get("verified") is True
                    and int(verification.get("confidence", 0)) >= cfg.reflexion.min_confidence
                )
                if verified_ok or not cfg.reflexion.enabled:
                    break

                # First: standard Reflexion correction
                correction = await self._reflexion.reflect(
                    task, str(outcome.get("result", "")), verification,
                )
                await self._log_event(
                    run.id, EventKind.reflexion, "reflexion",
                    f"Reflexion attempt {attempt + 1}/{cfg.reflexion.max_retries}: "
                    f"{correction[:200]}",
                    {"attempt": attempt + 1, "correction": correction},
                    task_id=task.id, session=session,
                )

                # Second: cooperative help from peers (if correction alone isn't enough)
                peer_advice = None
                if cfg.cooperative.enabled and attempt >= 1:
                    peer_advice = await bus.request_help(
                        from_agent=agent.name,
                        topic=task.title,
                        context=(
                            f"Task: {task.title}\n"
                            f"Failed result: {str(outcome.get('result', ''))[:500]}\n"
                            f"Verifier notes: {verification.get('notes', '')}"
                        ),
                        timeout=cfg.cooperative.help_timeout_seconds,
                    )
                    if peer_advice:
                        await self._log_event(
                            run.id, EventKind.help_response, "peer",
                            f"Peer advice received: {peer_advice[:200]}",
                            task_id=task.id, session=session,
                        )
                        correction = f"{correction}\n\nPeer advice: {peer_advice}"

                outcome = await self._execute_with_agent(
                    agent, task, context, run.id, reflection=correction,
                )

                if outcome.get("reasoning"):
                    await self._log_event(
                        run.id, EventKind.thinking, agent.name,
                        f"Revised: {outcome['reasoning']}",
                        task_id=task.id, session=session,
                    )
                if outcome.get("tool_result"):
                    tr = outcome["tool_result"]
                    await self._log_event(
                        run.id, EventKind.tool_result, "tool_operator",
                        f"Tool result (success={tr.get('success')})", tr,
                        task_id=task.id, session=session,
                    )
                    await self._store_artifact(run.id, task.id, tr, session=session)

                verification = await self._verifier.verify(
                    task, str(outcome.get("result", "")), outcome.get("tool_result"),
                )
                await self._log_event(
                    run.id, EventKind.verification, "verifier",
                    f"Re-verified={verification.get('verified')} "
                    f"confidence={verification.get('confidence')}",
                    verification, task_id=task.id, session=session,
                )

            # ── Write result + share knowledge ────────────────────────────────
            task.result = str(outcome.get("result", ""))
            task.status = TaskStatus.completed

            # Share a condensed fact to the knowledge store
            if task.result and cfg.cooperative.enabled:
                await knowledge.add(
                    key=task.title[:80],
                    value=task.result[:500],
                    source_agent=agent.name,
                )
                await bus.share_knowledge(agent.name, task.title[:80], task.result[:300])

            if outcome.get("kind") == "direct" and task.result:
                await self._store_text_artifact(run.id, task.id, task.result, session=session)

            await self._log_event(
                run.id, EventKind.task_completed, agent.name,
                f"Task completed: {task.title}",
                {"result_preview": task.result[:200] if task.result else "",
                 "agent": agent.name, "model": agent.model},
                task_id=task.id, session=session,
            )
        except Exception as exc:
            task.status = TaskStatus.failed
            task.result = str(exc)
            await self._log_event(
                run.id, EventKind.error, "orchestrator",
                f"Task failed: {exc}", task_id=task.id, session=session,
            )
        finally:
            session.add(task)
            await session.commit()

    # ── Per-agent model routing ────────────────────────────────────────────────

    async def _execute_with_agent(
        self,
        agent: AgentSpec,
        task: Task,
        context: str,
        run_id: str,
        reflection: str = "",
    ) -> dict[str, Any]:
        """Execute *task* using *agent*'s model via ToolOperator."""
        from backend.agents.tool_operator import ToolOperator

        # Pin the router to the agent's specific model for this execution.
        if agent.model:
            all_slots = ["fast", "reasoning", "code", "search", "math", "vision"]
            agent_router = ModelRouter(
                run_models={slot: agent.model for slot in all_slots}
            )
        else:
            agent_router = self._router

        tool_op = ToolOperator(agent_router, self._tool_bus)
        return await tool_op.execute_task(task, context, reflection=reflection, run_id=run_id)

    # ── Topological wave decomposition (shared with Orchestrator) ─────────────

    @staticmethod
    def _topological_waves(tasks: list[Task]) -> list[list[Task]]:
        id_to_task = {t.id: t for t in tasks}
        completed_ids: set[str] = set()
        remaining = list(tasks)
        waves: list[list[Task]] = []

        while remaining:
            wave = [
                t for t in remaining
                if all(dep in completed_ids for dep in (t.depends_on or []))
            ]
            if not wave:
                # Cycle or unresolvable dependency — add all remaining as one wave
                waves.append(remaining)
                break
            waves.append(wave)
            completed_ids.update(t.id for t in wave)
            remaining = [t for t in remaining if t.id not in completed_ids]

        return waves

    # ── Context budget (shared with Orchestrator) ──────────────────────────────

    async def _budget_context(self, raw_context: str) -> str:
        budget = self._cfg.memory.context_budget_chars
        if len(raw_context) <= budget:
            return raw_context
        try:
            summary = await self._router.chat(
                TaskType.fast,
                [
                    {"role": "system", "content": (
                        "You are a context compressor. Summarize the following task results "
                        "into a concise JSON-like summary preserving key facts. Max 500 words."
                    )},
                    {"role": "user", "content": raw_context[:8000]},
                ],
            )
            return summary.strip() or raw_context[:budget]
        except Exception:
            return raw_context[:budget]

    # ── Event logging ──────────────────────────────────────────────────────────

    async def _log_event(
        self,
        run_id: str,
        kind: EventKind,
        agent_role: str,
        content: str,
        data: dict | None = None,
        task_id: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        s = session or self._session
        event = Event(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            kind=kind,
            agent_role=agent_role,
            content=content,
            data=data,
        )
        s.add(event)
        await s.commit()

    # ── Artifact storage (same helpers as Orchestrator) ────────────────────────

    async def _store_artifact(
        self,
        run_id: str,
        task_id: str | None,
        tool_result: dict,
        session: AsyncSession,
    ) -> None:
        if not tool_result.get("output"):
            return
        output = tool_result.get("output", "")
        content = json.dumps(output, default=str) if not isinstance(output, str) else output
        if not content.strip():
            return
        artifact = Artifact(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            name=f"{tool_result.get('tool_name', 'tool')}-output",
            content_type="application/json",
            sha256=tool_result.get("sha256"),
            provenance={
                "tool": tool_result.get("tool_name"),
                "success": tool_result.get("success"),
            },
        )
        session.add(artifact)
        await session.commit()

    async def _store_text_artifact(
        self,
        run_id: str,
        task_id: str | None,
        text: str,
        session: AsyncSession,
    ) -> None:
        if not text.strip():
            return
        import hashlib
        sha = hashlib.sha256(text.encode()).hexdigest()
        artifact = Artifact(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            name="llm-answer",
            content_type="text/plain",
            sha256=sha,
            provenance={"source": "direct_llm_answer"},
        )
        session.add(artifact)
        await session.commit()
