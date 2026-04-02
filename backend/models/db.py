from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class EventKind(str, Enum):
    agent_message = "agent_message"
    tool_call = "tool_call"
    tool_result = "tool_result"
    plan_created = "plan_created"
    verification = "verification"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[RunStatus] = mapped_column(String(20), default=RunStatus.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="run", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship("Event", back_populates="run", cascade="all, delete-orphan")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    agent_role: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[TaskStatus] = mapped_column(String(20), default=TaskStatus.pending)
    depends_on: Mapped[Optional[list[str]]] = mapped_column(JSON, default=list)
    result: Mapped[Optional[str]] = mapped_column(Text)
    artifact_ids: Mapped[Optional[list[str]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    run: Mapped["Run"] = relationship("Run", back_populates="tasks")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    task_id: Mapped[Optional[str]] = mapped_column(ForeignKey("tasks.id"))
    kind: Mapped[EventKind] = mapped_column(String(32), nullable=False)
    agent_role: Mapped[Optional[str]] = mapped_column(String(64))
    content: Mapped[Optional[str]] = mapped_column(Text)
    data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    run: Mapped["Run"] = relationship("Run", back_populates="events")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    task_id: Mapped[Optional[str]] = mapped_column(ForeignKey("tasks.id"))
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), default="text/plain")
    file_path: Mapped[Optional[str]] = mapped_column(String(512))
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    provenance: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Memory(Base):
    """Cross-run learnings that the planner uses as context for future runs."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[Optional[str]] = mapped_column(String(36))
    goal_summary: Mapped[str] = mapped_column(Text, nullable=False)
    learnings: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
