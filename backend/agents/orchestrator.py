from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.critic import CriticAgent
from backend.agents.debater import Debater
from backend.agents.memory import MemoryStore
from backend.agents.planner import Planner
from backend.agents.reflexion import ReflexionAgent
from backend.agents.synthesizer import Synthesizer
from backend.agents.tool_operator import ToolOperator
from backend.agents.verifier import Verifier
from backend.agents.voter import Voter
from backend.config import get_config
from backend.llm.ollama_client import OllamaClient
from backend.llm.router import ModelRouter
from backend.models.database import get_session_factory
from backend.models.db import Artifact, Event, EventKind, Run, RunStatus, Task, TaskStatus
from backend.tools.bus import ToolBus
from backend.tools.scratchpad import clear_run_scratchpad

_TERMINAL = {RunStatus.completed, RunStatus.failed, RunStatus.cancelled}


class Orchestrator:
    """Owns the top-level goal, coordinates planning, execution, and verification.

    Architectural improvements over the baseline:
    * **Parallel wave execution**: tasks whose dependencies are all satisfied
      run concurrently within each DAG wave, each in its own DB session.
    * **Memory integration**: past learnings are recalled before planning and
      recorded after a successful run so the agent improves over time.
    * **Tool-aware planning**: the planner receives the full list of available
      tools so it can design tasks that map cleanly onto real capabilities.
    * **Session isolation**: every concurrent task operates in its own
      SQLAlchemy session to avoid cross-task session conflicts.
    * **Context budget**: dependency results are truncated / summarized before
      being injected into the next task so the model's context window is never
      silently overflowed.
    * **Critic**: after synthesis a Critic agent evaluates the final answer
      quality and surfaces any gaps to the user.
    * **Scratchpad cleanup**: per-run in-memory scratchpad is cleared when the
      run terminates to avoid memory leaks.
    * **Run timeout**: if ``server.max_runtime_seconds > 0`` the entire run is
      forcibly failed after that many seconds.
    """

    def __init__(
        self,
        session: AsyncSession,
        tool_bus: ToolBus,
        run_models: dict[str, str] | None = None,
    ):
        self._session = session
        self._tool_bus = tool_bus
        self._router = ModelRouter(run_models=run_models)
        self._planner = Planner(self._router)
        self._debater = Debater(self._router)
        self._voter = Voter(self._router)
        self._tool_op = ToolOperator(self._router, tool_bus)
        self._verifier = Verifier(self._router)
        self._reflexion = ReflexionAgent(self._router)
        self._synthesizer = Synthesizer(self._router)
        self._critic = CriticAgent(self._router)
        self._memory = MemoryStore(self._router)
        self._cfg = get_config()

    async def run(self, run: Run) -> None:
        """Execute the full run lifecycle."""
        # Re-read the run from DB before touching it — a cancel request may have
        # already set status = cancelled between task creation and first execution.
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
                        f"Run exceeded maximum runtime of {max_runtime}s and was terminated.",
                    )
                    self._session.add(run)
                    await self._session.commit()
        else:
            await self._run_inner(run)

        # Clean up per-run scratchpad regardless of outcome.
        clear_run_scratchpad(run.id)

    async def _run_inner(self, run: Run) -> None:
        """Core run logic (extracted so we can wrap it with a timeout)."""
        await self._log_event(run.id, EventKind.agent_message, "orchestrator",
                              f"Starting run: {run.goal}")

        # ── Preflight: verify Ollama is reachable ──────────────────────────────
        ollama = OllamaClient()
        if not await ollama.is_available():
            run.status = RunStatus.failed
            await self._log_event(
                run.id, EventKind.agent_message, "orchestrator",
                "Ollama is not running. "
                "Start it with: `ollama serve`, "
                "then pull a model: `ollama pull llama3.2:3b`",
            )
            self._session.add(run)
            await self._session.commit()
            return

        try:
            # ── 1. Recall past learnings ───────────────────────────────────────
            memories: list[str] = []
            if self._cfg.memory.enabled:
                memories = await self._memory.recall(
                    run.goal, limit=self._cfg.memory.recall_limit
                )

            # ── 2. Plan (tool-aware + memory-aware, optionally via debate) ────
            tool_names = self._tool_bus.list_tools()
            if self._cfg.debate.enabled:
                await self._log_event(
                    run.id, EventKind.agent_message, "debater",
                    f"Generating {self._cfg.debate.num_candidates} candidate plans...",
                )
                candidates = await self._debater.generate_candidates(
                    run.goal,
                    n=self._cfg.debate.num_candidates,
                    temperature=self._cfg.debate.temperature,
                    available_tools=tool_names,
                    memories=memories if memories else None,
                )
                tasks = await self._voter.select_best(run.goal, run.id, candidates)
                await self._log_event(
                    run.id, EventKind.agent_message, "voter",
                    f"Voter selected best plan from {len(candidates)} candidates.",
                )
            else:
                tasks = await self._planner.create_plan(
                    run.goal, run.id,
                    available_tools=tool_names,
                    memories=memories if memories else None,
                )
                # Retry once if the planner returned only the fallback single task
                # whose description signals a parse failure (empty plan).
                if (
                    len(tasks) == 1
                    and tasks[0].agent_role == "tool_operator"
                    and tasks[0].title == "Execute goal"
                ):
                    await self._log_event(
                        run.id, EventKind.agent_message, "planner",
                        "Initial plan was empty; retrying with explicit JSON instruction.",
                    )
                    tasks = await self._planner.create_plan(
                        run.goal, run.id,
                        available_tools=tool_names,
                        memories=memories if memories else None,
                    )
            for task in tasks:
                self._session.add(task)
            await self._session.commit()

            await self._log_event(run.id, EventKind.plan_created, "planner",
                                  f"Plan created with {len(tasks)} tasks",
                                  {"task_ids": [t.id for t in tasks]})

            # ── 3. Execute tasks in DAG waves (parallel within each wave) ──────
            completed: dict[str, Any] = {}
            waves = self._topological_waves(tasks)

            for wave in waves:
                # Check if the run has been cancelled between waves
                await self._session.refresh(run)
                if run.status == RunStatus.cancelled:
                    for task in wave:
                        task.status = TaskStatus.skipped
                        self._session.add(task)
                    await self._session.commit()
                    continue

                await self._execute_wave(run.id, [t.id for t in wave], completed)

            # ── 4. Record learnings for future runs ────────────────────────────
            await self._session.refresh(run)
            if run.status not in _TERMINAL and self._cfg.memory.enabled:
                await self._memory.record_run(run.id, run.goal, completed)

            # ── 5. Synthesize a final answer from all task results ─────────────
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
                        await self._log_event(
                            run.id, EventKind.synthesis, "synthesizer",
                            synthesis,
                        )

            # ── 6. Critic: evaluate quality of the synthesis ───────────────────
            if synthesis and self._cfg.critic.enabled:
                critique = await self._critic.evaluate(run.goal, synthesis)
                await self._log_event(
                    run.id, EventKind.critic, "critic",
                    f"Quality={critique['quality']}/100 passed={critique['passed']}",
                    critique,
                )

            # ── 7. Mark run complete ───────────────────────────────────────────
            await self._session.refresh(run)
            if run.status not in _TERMINAL:
                run.status = RunStatus.completed
                await self._log_event(run.id, EventKind.agent_message, "orchestrator",
                                      "All tasks completed successfully.")

            # Store accumulated token usage regardless of terminal status so we
            # always have visibility into how much was consumed.
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
    ) -> None:
        """Execute a wave of tasks.

        A single-task wave reuses the orchestrator's own session (no overhead).
        A multi-task wave launches each task in a fresh session so they can
        progress concurrently without sharing session state.
        """
        if len(task_ids) == 1:
            task = await self._session.get(Task, task_ids[0])
            run = await self._session.get(Run, run_id)
            if task and run:
                await self._execute_task(run, task, completed, self._session)
                completed[task.id] = task.result
            return

        # Parallel path: each task owns its own session
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
                    await self._execute_task(run, task, completed, s)
                    return task_id, task.result

        results = await asyncio.gather(
            *[_run_one(tid) for tid in task_ids],
            return_exceptions=True,
        )
        for task_id, result in zip(task_ids, results):
            if isinstance(result, BaseException):
                completed[task_id] = None
            else:
                t_id, t_result = result  # type: ignore[misc]
                completed[t_id] = t_result

    async def _execute_task(
        self,
        run: Run,
        task: Task,
        completed: dict[str, Any],
        session: AsyncSession,
    ) -> None:
        task.status = TaskStatus.running
        session.add(task)
        await session.commit()

        await self._log_event(
            run.id, EventKind.task_started, task.agent_role or "agent",
            f"Task started: {task.title}", task_id=task.id, session=session,
        )

        # Build context from dependency results with budget enforcement.
        raw_context = json.dumps({dep: completed.get(dep) for dep in (task.depends_on or [])})
        context = await self._budget_context(raw_context)

        try:
            outcome = await self._tool_op.execute_task(task, context, run_id=run.id)

            # Log tool call + result
            if outcome.get("tool_call"):
                await self._log_event(run.id, EventKind.tool_call, "tool_operator",
                                      f"Calling {outcome['tool_call'].get('tool')}",
                                      outcome["tool_call"], task_id=task.id, session=session)
            if outcome.get("tool_result"):
                tr = outcome["tool_result"]
                await self._log_event(run.id, EventKind.tool_result, "tool_operator",
                                      f"Tool result (success={tr.get('success')})",
                                      tr, task_id=task.id, session=session)
                # Store as artifact
                await self._store_artifact(run.id, task.id, tr, session=session)

            # Verify
            verification = await self._verifier.verify(
                task,
                str(outcome.get("result", "")),
                outcome.get("tool_result"),
            )
            await self._log_event(run.id, EventKind.verification, "verifier",
                                  f"Verified={verification.get('verified')} "
                                  f"confidence={verification.get('confidence')}",
                                  verification, task_id=task.id, session=session)

            # ── Reflexion loop: retry if verification failed or confidence low ─
            cfg = self._cfg
            reflexion_enabled = cfg.reflexion.enabled
            min_confidence = cfg.reflexion.min_confidence
            max_retries = cfg.reflexion.max_retries

            for reflexion_attempt in range(max_retries):
                verified_ok = (
                    verification.get("verified") is True
                    and int(verification.get("confidence", 0)) >= min_confidence
                )
                if verified_ok or not reflexion_enabled:
                    break

                # Generate a corrective prompt from the Reflexion agent
                correction = await self._reflexion.reflect(
                    task,
                    str(outcome.get("result", "")),
                    verification,
                )
                await self._log_event(
                    run.id, EventKind.reflexion, "reflexion",
                    f"Reflexion attempt {reflexion_attempt + 1}/{max_retries}: {correction[:200]}",
                    {
                        "attempt": reflexion_attempt + 1,
                        "max_retries": max_retries,
                        "correction": correction,
                        "prev_confidence": verification.get("confidence"),
                    },
                    task_id=task.id,
                    session=session,
                )

                # Re-execute with correction injected as additional context
                outcome = await self._tool_op.execute_task(task, context, reflection=correction, run_id=run.id)

                if outcome.get("tool_call"):
                    await self._log_event(run.id, EventKind.tool_call, "tool_operator",
                                          f"Calling {outcome['tool_call'].get('tool')}",
                                          outcome["tool_call"], task_id=task.id, session=session)
                if outcome.get("tool_result"):
                    tr = outcome["tool_result"]
                    await self._log_event(run.id, EventKind.tool_result, "tool_operator",
                                          f"Tool result (success={tr.get('success')})",
                                          tr, task_id=task.id, session=session)
                    await self._store_artifact(run.id, task.id, tr, session=session)

                # Re-verify after correction
                verification = await self._verifier.verify(
                    task,
                    str(outcome.get("result", "")),
                    outcome.get("tool_result"),
                )
                await self._log_event(run.id, EventKind.verification, "verifier",
                                      f"Re-verified={verification.get('verified')} "
                                      f"confidence={verification.get('confidence')}",
                                      verification, task_id=task.id, session=session)

            task.result = str(outcome.get("result", ""))
            task.status = TaskStatus.completed

            # Store plain-text results (direct LLM answers) as text artifacts
            if outcome.get("kind") == "direct" and task.result:
                await self._store_text_artifact(run.id, task.id, task.result, session=session)

            await self._log_event(
                run.id, EventKind.task_completed, task.agent_role or "agent",
                f"Task completed: {task.title}",
                {"result_preview": task.result[:200] if task.result else ""},
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

    # ── Context budget management ──────────────────────────────────────────────

    async def _budget_context(self, raw_context: str) -> str:
        """Truncate or summarize dependency context to stay within budget.

        If the raw context JSON string exceeds ``memory.context_budget_chars``
        we ask the LLM to produce a concise summary of the dependency results
        so the next task receives a dense but compact context that doesn't
        overflow the model's context window.
        """
        budget = self._cfg.memory.context_budget_chars
        if len(raw_context) <= budget:
            return raw_context

        # Try LLM summarization; fall back to hard truncation on any error.
        try:
            from backend.llm.router import TaskType
            summary = await self._router.chat(
                TaskType.fast,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a context compressor. Summarize the following task results "
                            "into a concise JSON-like summary that preserves the key facts and "
                            "findings. Be dense and specific. Max 500 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": raw_context[:budget * 2],
                    },
                ],
            )
            return summary[:budget] if summary else raw_context[:budget]
        except Exception:
            return raw_context[:budget]

    # ── Logging & artifact helpers ─────────────────────────────────────────────

    async def _log_event(
        self,
        run_id: str,
        kind: EventKind,
        agent_role: str,
        content: str,
        data: dict[str, Any] | None = None,
        task_id: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        _session = session if session is not None else self._session
        event = Event(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            kind=kind,
            agent_role=agent_role,
            content=content,
            data=data,
        )
        _session.add(event)
        await _session.commit()

    async def _store_text_artifact(
        self, run_id: str, task_id: str, text: str, session: AsyncSession | None = None
    ) -> None:
        _session = session if session is not None else self._session
        cfg = self._cfg
        artifact_dir = Path(cfg.artifacts.dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        sha = hashlib.sha256(text.encode()).hexdigest()
        file_name = f"{sha[:16]}.txt"
        file_path = artifact_dir / file_name
        file_path.write_text(text)

        artifact = Artifact(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            name=file_name,
            content_type="text/plain",
            file_path=str(file_path),
            sha256=sha,
            provenance={"kind": "llm_result"},
        )
        _session.add(artifact)
        await _session.commit()

    async def _store_artifact(
        self,
        run_id: str,
        task_id: str,
        tool_result: dict[str, Any],
        session: AsyncSession | None = None,
    ) -> None:
        _session = session if session is not None else self._session
        cfg = self._cfg
        artifact_dir = Path(cfg.artifacts.dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        content = json.dumps(tool_result, default=str)
        sha = hashlib.sha256(content.encode()).hexdigest()
        file_name = f"{sha[:16]}.json"
        file_path = artifact_dir / file_name
        file_path.write_text(content)

        artifact = Artifact(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            name=file_name,
            content_type="application/json",
            file_path=str(file_path),
            sha256=sha,
            provenance={"tool": tool_result.get("tool_name")},
        )
        _session.add(artifact)
        await _session.commit()

    # ── DAG helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _topological_waves(tasks: list[Task]) -> list[list[Task]]:
        """Group tasks into dependency waves for parallel execution.

        Tasks within the same wave have no dependencies on each other and can
        be executed concurrently.  Tasks in wave N depend only on tasks in
        waves 0..N-1.

        Example:
            t1(deps=[])  →  wave 0
            t2(deps=[t1]), t3(deps=[t1])  →  wave 1  (run in parallel)
            t4(deps=[t2, t3])  →  wave 2
        """
        id_to_task = {t.id: t for t in tasks}
        wave_map: dict[str, int] = {}

        def get_wave(task_id: str, visiting: set[str]) -> int:
            if task_id in wave_map:
                return wave_map[task_id]
            if task_id in visiting:  # cycle — treat as independent
                return 0
            task = id_to_task.get(task_id)
            if task is None or not task.depends_on:
                wave_map[task_id] = 0
                return 0
            visiting.add(task_id)
            max_dep_wave = max(
                (get_wave(dep, visiting) for dep in task.depends_on if dep in id_to_task),
                default=-1,
            )
            visiting.discard(task_id)
            wave_map[task_id] = max_dep_wave + 1
            return wave_map[task_id]

        for task in tasks:
            get_wave(task.id, set())

        if not wave_map:
            return []

        num_waves = max(wave_map.values()) + 1
        waves: list[list[Task]] = [[] for _ in range(num_waves)]
        for task in tasks:
            waves[wave_map.get(task.id, 0)].append(task)
        return [w for w in waves if w]

    @staticmethod
    def _topological_sort(tasks: list[Task]) -> list[Task]:
        """Kahn's algorithm for DAG topological sort (kept for backward compat)."""
        id_to_task = {t.id: t for t in tasks}
        in_degree: dict[str, int] = {t.id: 0 for t in tasks}
        dependents: dict[str, list[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in (task.depends_on or []):
                if dep in in_degree:
                    in_degree[task.id] += 1
                    dependents[dep].append(task.id)

        queue = [t for t in tasks if in_degree[t.id] == 0]
        result: list[Task] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dep_id in dependents[node.id]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(id_to_task[dep_id])

        # If we didn't process all tasks, there's a cycle - just return remaining
        remaining = [t for t in tasks if t not in result]
        return result + remaining
