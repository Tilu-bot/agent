from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.orchestrator import Orchestrator
from backend.models.db import Artifact, Event, Run, RunStatus, Task
from backend.models.database import get_session_factory

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _get_session():
    factory = get_session_factory()
    return factory()


async def get_db() -> AsyncSession:
    factory = get_session_factory()
    async with factory() as session:
        yield session


class CreateRunRequest(BaseModel):
    goal: str


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


@router.post("", response_model=RunResponse)
async def create_run(body: CreateRunRequest, db: AsyncSession = Depends(get_db)):
    from backend.tools import create_default_tool_bus

    run = Run(id=str(uuid.uuid4()), goal=body.goal)
    db.add(run)
    await db.commit()
    await db.refresh(run)

    # Launch orchestrator in background
    tool_bus = create_default_tool_bus()
    run_id = run.id

    async def _run_background():
        factory = get_session_factory()
        async with factory() as bg_session:
            bg_run = await bg_session.get(Run, run_id)
            if bg_run is None:
                return
            orch = Orchestrator(bg_session, tool_bus)
            await orch.run(bg_run)

    asyncio.create_task(_run_background())

    return RunResponse.from_orm(run)


@router.get("", response_model=list[RunResponse])
async def list_runs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Run).order_by(Run.created_at.desc()))
    runs = result.scalars().all()
    return [RunResponse.from_orm(r) for r in runs]


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    run = await db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse.from_orm(run)


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
