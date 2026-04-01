from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.planner import Planner
from backend.agents.tool_operator import ToolOperator
from backend.agents.verifier import Verifier
from backend.config import get_config
from backend.llm.ollama_client import OllamaClient
from backend.llm.router import ModelRouter, TaskType
from backend.models.db import Artifact, Event, EventKind, Run, RunStatus, Task, TaskStatus
from backend.tools.bus import ToolBus

_TERMINAL = {RunStatus.completed, RunStatus.failed, RunStatus.cancelled}


class Orchestrator:
    """Owns the top-level goal, coordinates planning, execution, and verification."""

    def __init__(self, session: AsyncSession, tool_bus: ToolBus):
        self._session = session
        self._router = ModelRouter()
        self._planner = Planner(self._router)
        self._tool_op = ToolOperator(self._router, tool_bus)
        self._verifier = Verifier(self._router)
        self._cfg = get_config()

    async def run(self, run: Run) -> None:
        """Execute the full run lifecycle."""
        run.status = RunStatus.running
        self._session.add(run)
        await self._session.commit()

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
            # 1. Plan
            tasks = await self._planner.create_plan(run.goal, run.id)
            for task in tasks:
                self._session.add(task)
            await self._session.commit()

            await self._log_event(run.id, EventKind.plan_created, "planner",
                                  f"Plan created with {len(tasks)} tasks",
                                  {"task_ids": [t.id for t in tasks]})

            # 2. Execute tasks in DAG order
            completed: dict[str, Any] = {}
            for task in self._topological_sort(tasks):
                # Check if the run has been cancelled between tasks
                await self._session.refresh(run)
                if run.status == RunStatus.cancelled:
                    task.status = TaskStatus.skipped
                    self._session.add(task)
                    await self._session.commit()
                    continue
                await self._execute_task(run, task, completed)
                completed[task.id] = task.result

            # 3. Mark run complete (unless cancelled mid-run)
            await self._session.refresh(run)
            if run.status not in _TERMINAL:
                run.status = RunStatus.completed
                await self._log_event(run.id, EventKind.agent_message, "orchestrator",
                                      "All tasks completed successfully.")
        except Exception as exc:
            await self._session.refresh(run)
            if run.status not in _TERMINAL:
                run.status = RunStatus.failed
                await self._log_event(run.id, EventKind.agent_message, "orchestrator",
                                      f"Run failed: {exc}")
        finally:
            self._session.add(run)
            await self._session.commit()

    async def _execute_task(
        self, run: Run, task: Task, completed: dict[str, Any]
    ) -> None:
        task.status = TaskStatus.running
        self._session.add(task)
        await self._session.commit()

        await self._log_event(run.id, EventKind.agent_message, task.agent_role or "agent",
                              f"Starting task: {task.title}", task_id=task.id)

        context = json.dumps({dep: completed.get(dep) for dep in (task.depends_on or [])})
        try:
            outcome = await self._tool_op.execute_task(task, context)

            # Log tool call + result
            if outcome.get("tool_call"):
                await self._log_event(run.id, EventKind.tool_call, "tool_operator",
                                      f"Calling {outcome['tool_call'].get('tool')}",
                                      outcome["tool_call"], task_id=task.id)
            if outcome.get("tool_result"):
                tr = outcome["tool_result"]
                await self._log_event(run.id, EventKind.tool_result, "tool_operator",
                                      f"Tool result (success={tr.get('success')})",
                                      tr, task_id=task.id)
                # Store as artifact
                await self._store_artifact(run.id, task.id, tr)

            # Verify
            verification = await self._verifier.verify(
                task,
                str(outcome.get("result", "")),
                outcome.get("tool_result"),
            )
            await self._log_event(run.id, EventKind.verification, "verifier",
                                  f"Verified={verification.get('verified')} "
                                  f"confidence={verification.get('confidence')}",
                                  verification, task_id=task.id)

            task.result = str(outcome.get("result", ""))
            task.status = TaskStatus.completed

            # Store plain-text results (direct LLM answers) as text artifacts
            if outcome.get("kind") == "direct" and task.result:
                await self._store_text_artifact(run.id, task.id, task.result)
        except Exception as exc:
            task.status = TaskStatus.failed
            task.result = str(exc)
            await self._log_event(run.id, EventKind.agent_message, "orchestrator",
                                  f"Task failed: {exc}", task_id=task.id)
        finally:
            self._session.add(task)
            await self._session.commit()

    async def _log_event(
        self,
        run_id: str,
        kind: EventKind,
        agent_role: str,
        content: str,
        data: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> None:
        event = Event(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            kind=kind,
            agent_role=agent_role,
            content=content,
            data=data,
        )
        self._session.add(event)
        await self._session.commit()

    async def _store_text_artifact(
        self, run_id: str, task_id: str, text: str
    ) -> None:
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
        self._session.add(artifact)
        await self._session.commit()

    async def _store_artifact(
        self, run_id: str, task_id: str, tool_result: dict[str, Any]
    ) -> None:
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
        self._session.add(artifact)
        await self._session.commit()

    @staticmethod
    def _topological_sort(tasks: list[Task]) -> list[Task]:
        """Kahn's algorithm for DAG topological sort."""
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
