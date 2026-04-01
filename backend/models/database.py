from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config import get_config
from backend.models.db import Base

_engine = None
_session_factory = None


def _make_engine():
    cfg = get_config()
    db_url = cfg.database.url
    # Ensure directory exists for SQLite
    if "sqlite" in db_url:
        db_file = db_url.replace("sqlite+aiosqlite:///", "")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(db_url, echo=False, future=True)


async def init_db() -> None:
    global _engine, _session_factory
    _engine = _make_engine()
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session_factory() -> async_sessionmaker:
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
