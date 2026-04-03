from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.models import router as models_router
from backend.api.runs import router as runs_router
from backend.api.training import router as training_router
from backend.config import get_config
from backend.models.database import init_db

_UNPROTECTED_PATHS = {"/api/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(title="Agentic", version="0.2.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Optional API-key authentication ───────────────────────────────────────
    # Set AGENTIC_API_KEY in the environment to enable.  When unset every
    # request is allowed (useful for local-only deployments).
    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next) -> Response:
        required_key = os.environ.get("AGENTIC_API_KEY", "")
        if required_key and request.url.path not in _UNPROTECTED_PATHS:
            provided = request.headers.get("X-API-Key", "")
            if provided != required_key:
                return Response(
                    content='{"detail":"Invalid or missing X-API-Key header"}',
                    status_code=401,
                    media_type="application/json",
                )
        return await call_next(request)

    app.include_router(runs_router)
    app.include_router(models_router)
    app.include_router(training_router)
    app.include_router(chat_router)

    @app.get("/api/health")
    async def health():
        from backend.llm.ollama_client import OllamaClient
        ollama = OllamaClient()
        ollama_ok = await ollama.is_available()
        models: list[str] = []
        if ollama_ok:
            try:
                models = await ollama.list_models()
            except Exception:
                pass
        return {
            "status": "ok",
            "ollama": ollama_ok,
            "models_available": len(models),
            "models": models,
        }

    @app.get("/api/stats")
    async def global_stats():
        """Return aggregate statistics across all runs stored in the database."""
        from sqlalchemy import func as sqlfunc, select as sqselect
        from sqlalchemy.ext.asyncio import AsyncSession
        from backend.models.database import get_session_factory as _gsf
        from backend.models.db import Run as _Run, Task as _Task, RunStatus as _RS

        factory = _gsf()
        async with factory() as db:
            # Counts by status
            status_rows = await db.execute(
                sqselect(_Run.status, sqlfunc.count(_Run.id).label("count"))
                .group_by(_Run.status)
            )
            counts_by_status = {row.status: row.count for row in status_rows}

            # Total runs
            total_runs = sum(counts_by_status.values())

            # Total tasks
            task_count_row = await db.execute(
                sqselect(sqlfunc.count(_Task.id))
            )
            total_tasks = task_count_row.scalar() or 0

            # Token usage totals across all runs that have token_usage set
            all_runs_result = await db.execute(
                sqselect(_Run.token_usage).where(_Run.token_usage.isnot(None))
            )
            total_prompt = 0
            total_completion = 0
            token_run_count = 0
            for (usage,) in all_runs_result:
                if isinstance(usage, dict):
                    total_prompt += usage.get("prompt_tokens", 0)
                    total_completion += usage.get("completion_tokens", 0)
                    token_run_count += 1

        return {
            "total_runs": total_runs,
            "runs_by_status": counts_by_status,
            "total_tasks": total_tasks,
            "token_usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "tracked_runs": token_run_count,
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("backend.main:app", host=cfg.server.host, port=cfg.server.port, reload=True)
