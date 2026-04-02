from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.orchestrator import Orchestrator
from backend.config import get_config
from backend.models.db import Artifact, Event, EventKind, Run, RunStatus, Task, TaskStatus
from backend.models.database import get_session_factory

router = APIRouter(prefix="/api/runs", tags=["runs"])

# ── Concurrency limiting ───────────────────────────────────────────────────────
_RUN_SEMAPHORE: asyncio.Semaphore | None = None

def _get_semaphore() -> asyncio.Semaphore:
    global _RUN_SEMAPHORE
    if _RUN_SEMAPHORE is None:
        _RUN_SEMAPHORE = asyncio.Semaphore(get_config().server.max_concurrent_runs)
    return _RUN_SEMAPHORE


# Track running asyncio.Tasks so we can cancel them
_run_tasks: dict[str, asyncio.Task] = {}

# Terminal statuses — stream ends when the run reaches one of these
_TERMINAL = {RunStatus.completed, RunStatus.failed, RunStatus.cancelled}


async def get_db() -> AsyncSession:
    factory = get_session_factory()
    async with factory() as session:
        yield session


# ── Request / response models ──────────────────────────────────────────────────

class CreateRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Goal for the agent run")

    @field_validator("goal")
    @classmethod
    def goal_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be blank or whitespace")
        return v


class RunResponse(BaseModel):
    id: str
    goal: str
    status: str
    created_at: str
    updated_at: str

    @classmethod
    def from_orm(cls, run: Run) -> "RunResponse":
        return cls(
            id=run.id,
            goal=run.goal,
            status=run.status,
            created_at=run.created_at.isoformat(),
            updated_at=run.updated_at.isoformat(),
        )


class TaskResponse(BaseModel):
    id: str
    run_id: str
    title: str
    description: str | None
    agent_role: str | None
    status: str
    depends_on: list[str]
    result: str | None
    artifact_ids: list[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_orm(cls, task: Task) -> "TaskResponse":
        return cls(
            id=task.id,
            run_id=task.run_id,
            title=task.title,
            description=task.description,
            agent_role=task.agent_role,
            status=task.status,
            depends_on=task.depends_on or [],
            result=task.result,
            artifact_ids=task.artifact_ids or [],
            created_at=task.created_at.isoformat(),
            updated_at=task.updated_at.isoformat(),
        )


class EventResponse(BaseModel):
    id: str
    run_id: str
    task_id: str | None
    kind: str
    agent_role: str | None
    content: str | None
    data: Any
    created_at: str

    @classmethod
    def from_orm(cls, event: Event) -> "EventResponse":
        return cls(
            id=event.id,
            run_id=event.run_id,
            task_id=event.task_id,
            kind=event.kind,
            agent_role=event.agent_role,
            content=event.content,
            data=event.data,
            created_at=event.created_at.isoformat(),
        )


class ArtifactResponse(BaseModel):
    id: str
    run_id: str
    task_id: str | None
    name: str
    content_type: str
    sha256: str | None
    provenance: Any
    created_at: str

    @classmethod
    def from_orm(cls, artifact: Artifact) -> "ArtifactResponse":
        return cls(
            id=artifact.id,
            run_id=artifact.run_id,
            task_id=artifact.task_id,
            name=artifact.name,
            content_type=artifact.content_type,
            sha256=artifact.sha256,
            provenance=artifact.provenance,
            created_at=artifact.created_at.isoformat(),
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", response_model=RunResponse)
async def create_run(body: CreateRunRequest, db: AsyncSession = Depends(get_db)):
    from backend.tools import create_default_tool_bus

    run = Run(id=str(uuid.uuid4()), goal=body.goal)
    db.add(run)
    await db.commit()
    await db.refresh(run)

    tool_bus = create_default_tool_bus()
    run_id = run.id

    async def _run_background():
        async with _get_semaphore():
            factory = get_session_factory()
            async with factory() as bg_session:
                bg_run = await bg_session.get(Run, run_id)
                if bg_run is None:
                    return
                orch = Orchestrator(bg_session, tool_bus)
                await orch.run(bg_run)

    task = asyncio.create_task(_run_background())
    _run_tasks[run_id] = task
    task.add_done_callback(lambda _: _run_tasks.pop(run_id, None))

    return RunResponse.from_orm(run)


@router.get("", response_model=list[RunResponse])
async def list_runs(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List runs with optional pagination via ``?limit=`` and ``?offset=``."""
    limit = max(1, min(limit, 200))   # clamp: 1 – 200
    offset = max(0, offset)
    result = await db.execute(
        select(Run).order_by(Run.created_at.desc()).limit(limit).offset(offset)
    )
    runs = result.scalars().all()
    return [RunResponse.from_orm(r) for r in runs]


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse.from_orm(run)


@router.delete("/{run_id}", status_code=204)
async def cancel_run(run_id: str, db: AsyncSession = Depends(get_db)):
    """Cancel a pending or running run."""
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status in _TERMINAL:
        raise HTTPException(status_code=409, detail=f"Run is already {run.status}")

    run.status = RunStatus.cancelled
    db.add(run)
    await db.commit()

    # Cancel the background asyncio.Task if it is still running
    bg_task = _run_tasks.pop(run_id, None)
    if bg_task and not bg_task.done():
        bg_task.cancel()


@router.get("/{run_id}/stream")
async def stream_run(run_id: str):
    """Server-Sent Events stream for a run.

    Each message is a JSON object:
    ``{"type": "run_update"|"task_update"|"artifact"|"event"|"done", "payload": {...}}``

    The stream closes automatically once the run reaches a terminal state.
    """
    # Verify the run exists before we open the generator
    factory = get_session_factory()
    async with factory() as db:
        run = await db.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

    async def _generate():
        sent_event_ids: set[str] = set()
        sent_artifact_ids: set[str] = set()
        last_run_status: str | None = None
        # key = task_id, value = "status:result" fingerprint
        last_task_fingerprints: dict[str, str] = {}

        while True:
            try:
                async with factory() as db:
                    run = await db.get(Run, run_id)
                    if run is None:
                        break

                    # ── Run status update ──────────────────────────────────────
                    if run.status != last_run_status:
                        last_run_status = run.status
                        msg = json.dumps({
                            "type": "run_update",
                            "payload": RunResponse.from_orm(run).model_dump(),
                        })
                        yield f"data: {msg}\n\n"

                    # ── New events ─────────────────────────────────────────────
                    ev_result = await db.execute(
                        select(Event).where(Event.run_id == run_id).order_by(Event.created_at)
                    )
                    for event in ev_result.scalars().all():
                        if event.id not in sent_event_ids:
                            sent_event_ids.add(event.id)
                            msg = json.dumps({
                                "type": "event",
                                "payload": EventResponse.from_orm(event).model_dump(),
                            })
                            yield f"data: {msg}\n\n"

                    # ── Task status updates ────────────────────────────────────
                    tk_result = await db.execute(
                        select(Task).where(Task.run_id == run_id).order_by(Task.created_at)
                    )
                    for task in tk_result.scalars().all():
                        fingerprint = f"{task.status}:{task.result}"
                        if last_task_fingerprints.get(task.id) != fingerprint:
                            last_task_fingerprints[task.id] = fingerprint
                            msg = json.dumps({
                                "type": "task_update",
                                "payload": TaskResponse.from_orm(task).model_dump(),
                            })
                            yield f"data: {msg}\n\n"

                    # ── New artifacts ──────────────────────────────────────────
                    ar_result = await db.execute(
                        select(Artifact).where(Artifact.run_id == run_id).order_by(Artifact.created_at)
                    )
                    for artifact in ar_result.scalars().all():
                        if artifact.id not in sent_artifact_ids:
                            sent_artifact_ids.add(artifact.id)
                            msg = json.dumps({
                                "type": "artifact",
                                "payload": ArtifactResponse.from_orm(artifact).model_dump(),
                            })
                            yield f"data: {msg}\n\n"

                    # ── Terminal check ─────────────────────────────────────────
                    if run.status in _TERMINAL:
                        yield 'data: {"type":"done"}\n\n'
                        break

            except asyncio.CancelledError:
                break
            except Exception:
                # Don't let transient DB errors kill the stream
                pass

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/{run_id}/tasks", response_model=list[TaskResponse])
async def get_tasks(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    result = await db.execute(
        select(Task).where(Task.run_id == run_id).order_by(Task.created_at)
    )
    tasks = result.scalars().all()
    return [TaskResponse.from_orm(t) for t in tasks]


@router.get("/{run_id}/events", response_model=list[EventResponse])
async def get_events(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    result = await db.execute(
        select(Event).where(Event.run_id == run_id).order_by(Event.created_at)
    )
    events = result.scalars().all()
    return [EventResponse.from_orm(e) for e in events]


@router.get("/{run_id}/artifacts", response_model=list[ArtifactResponse])
async def get_artifacts(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    result = await db.execute(
        select(Artifact).where(Artifact.run_id == run_id).order_by(Artifact.created_at)
    )
    artifacts = result.scalars().all()
    return [ArtifactResponse.from_orm(a) for a in artifacts]


@router.get("/{run_id}/artifacts/{artifact_id}/content")
async def get_artifact_content(
    run_id: str, artifact_id: str, db: AsyncSession = Depends(get_db)
):
    artifact = await db.get(Artifact, artifact_id)
    if artifact is None or artifact.run_id != run_id:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not artifact.file_path:
        raise HTTPException(status_code=404, detail="Artifact has no file content")
    from pathlib import Path

    path = Path(artifact.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing on disk")
    return {"content": path.read_text()}


# ── Training-data export ───────────────────────────────────────────────────────

@router.get("/export/training-data")
async def export_training_data(
    min_confidence: int = 70,
    format: str = "alpaca",
    db: AsyncSession = Depends(get_db),
):
    """Export verified run data as JSONL for fine-tuning.

    Query parameters
    ----------------
    min_confidence : int (default 70)
        Only include tasks whose verifier confidence is >= this value.
    format : "alpaca" | "sharegpt" (default "alpaca")
        Output format.

    Each line of the JSONL response is one training example.

    Alpaca format::

        {"instruction": "<goal>", "input": "", "output": "<plan+results JSON>"}

    ShareGPT format::

        {"conversations": [{"from": "human", "value": "<goal>"},
                           {"from": "gpt",   "value": "<plan+results JSON>"}]}
    """
    if format not in ("alpaca", "sharegpt"):
        raise HTTPException(status_code=400, detail="format must be 'alpaca' or 'sharegpt'")

    min_confidence = max(0, min(100, min_confidence))

    # ── Fetch all completed runs ───────────────────────────────────────────────
    run_result = await db.execute(
        select(Run).where(Run.status == RunStatus.completed).order_by(Run.created_at)
    )
    runs = run_result.scalars().all()

    lines: list[str] = []

    for run in runs:
        # Fetch completed tasks for this run
        task_result = await db.execute(
            select(Task)
            .where(Task.run_id == run.id, Task.status == TaskStatus.completed)
            .order_by(Task.created_at)
        )
        tasks = task_result.scalars().all()
        if not tasks:
            continue

        # Fetch verification events for this run
        ev_result = await db.execute(
            select(Event)
            .where(Event.run_id == run.id, Event.kind == EventKind.verification)
        )
        verif_events = ev_result.scalars().all()

        # Build task_id → verification data map
        verif_map: dict[str, dict[str, Any]] = {}
        for ev in verif_events:
            if ev.task_id and ev.data:
                verif_map[ev.task_id] = ev.data

        # Filter tasks that pass the quality threshold
        qualified_tasks = []
        for task in tasks:
            v = verif_map.get(task.id, {})
            if v.get("verified") is True and int(v.get("confidence", 0)) >= min_confidence:
                qualified_tasks.append(task)

        if not qualified_tasks:
            continue

        # Build the output representation (plan + results)
        output_tasks = [
            {
                "title": t.title,
                "description": t.description,
                "agent_role": t.agent_role,
                "result": t.result,
            }
            for t in qualified_tasks
        ]
        output_str = json.dumps({"tasks": output_tasks}, ensure_ascii=False)

        if format == "alpaca":
            example = {
                "instruction": run.goal,
                "input": "",
                "output": output_str,
            }
        else:  # sharegpt
            example = {
                "conversations": [
                    {"from": "human", "value": run.goal},
                    {"from": "gpt", "value": output_str},
                ]
            }

        lines.append(json.dumps(example, ensure_ascii=False))

    content = "\n".join(lines) + ("\n" if lines else "")
    return StreamingResponse(
        iter([content]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="training_data.jsonl"'},
    )
