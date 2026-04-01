"""Basic smoke tests for agentic backend imports and logic."""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def use_temp_dir(tmp_path, monkeypatch):
    """Run each test in a temporary directory so .agentic data is isolated."""
    monkeypatch.chdir(tmp_path)


def test_config_loads():
    from backend.config import load_config, AppConfig

    cfg = load_config.__wrapped__() if hasattr(load_config, "__wrapped__") else AppConfig()
    assert cfg.models.fast
    assert cfg.ollama.base_url.startswith("http")


def test_tool_bus_registers_default_tools():
    from backend.tools import create_default_tool_bus

    bus = create_default_tool_bus()
    tools = bus.list_tools()
    assert "filesystem.read" in tools
    assert "filesystem.write" in tools
    assert "web.fetch" in tools
    assert "shell.exec" in tools


def test_model_router_selects_models():
    from backend.llm.router import ModelRouter, TaskType

    router = ModelRouter()
    assert router.select_model(TaskType.fast)
    assert router.select_model(TaskType.code)
    assert router.select_model(TaskType.reasoning)
    # Unknown type falls back to fast model
    assert router.select_model("unknown") == router.select_model(TaskType.fast)


@pytest.mark.asyncio
async def test_db_init_and_run_creation(tmp_path):
    import os
    os.chdir(tmp_path)

    from backend.models.database import init_db, session_scope
    from backend.models.db import Run

    await init_db()

    import uuid
    run_id = str(uuid.uuid4())
    async with session_scope() as session:
        run = Run(id=run_id, goal="Test goal")
        session.add(run)

    async with session_scope() as session:
        from sqlalchemy import select
        result = await session.execute(select(Run).where(Run.id == run_id))
        run = result.scalar_one_or_none()
        assert run is not None
        assert run.goal == "Test goal"
        assert run.status == "pending"


@pytest.mark.asyncio
async def test_filesystem_read_tool(tmp_path):
    from backend.tools.filesystem import FilesystemReadTool

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello from agentic")

    tool = FilesystemReadTool()
    result = await tool.execute({"path": str(test_file)})
    assert result.success
    assert result.output["content"] == "hello from agentic"


@pytest.mark.asyncio
async def test_filesystem_read_missing_file():
    from backend.tools.filesystem import FilesystemReadTool

    tool = FilesystemReadTool()
    result = await tool.execute({"path": "/nonexistent/path/file.txt"})
    assert not result.success
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_filesystem_write_disabled_by_default(tmp_path):
    from backend.tools.filesystem import FilesystemWriteTool

    tool = FilesystemWriteTool()
    result = await tool.execute({"path": str(tmp_path / "out.txt"), "content": "hi"})
    assert not result.success
    assert "disabled" in result.error.lower()


@pytest.mark.asyncio
async def test_tool_result_sha256():
    from backend.tools.bus import ToolResult

    result = ToolResult.from_success("test.tool", {"data": "hello"})
    assert result.sha256 is not None
    assert len(result.sha256) == 64  # SHA-256 hex


def test_planner_parse_fallback():
    """Planner._parse_plan falls back gracefully on invalid JSON."""
    from backend.agents.planner import Planner
    from backend.llm.router import ModelRouter

    router = ModelRouter()
    planner = Planner(router)
    tasks = planner._parse_plan("not json at all", "run-123")
    assert len(tasks) == 1
    assert tasks[0].run_id == "run-123"


def test_verifier_parse_fallback():
    """Verifier._parse handles non-JSON gracefully."""
    from backend.agents.verifier import Verifier
    from backend.llm.router import ModelRouter

    router = ModelRouter()
    verifier = Verifier(router)
    result = verifier._parse("not json")
    assert result["verified"] is False
    assert "confidence" in result


def test_fastapi_app_creates():
    from backend.main import create_app

    app = create_app()
    routes = [r.path for r in app.routes]
    assert "/api/runs" in routes
    assert "/api/health" in routes
