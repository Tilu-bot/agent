from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.runs import router as runs_router
from backend.config import get_config
from backend.models.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(title="Agentic", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(runs_router)

    @app.get("/api/health")
    async def health():
        from backend.llm.ollama_client import OllamaClient
        ollama = OllamaClient()
        ollama_ok = await ollama.is_available()
        return {"status": "ok", "ollama": ollama_ok}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run("backend.main:app", host=cfg.server.host, port=cfg.server.port, reload=True)
