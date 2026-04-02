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


def test_config_max_concurrent_runs():
    from backend.config import AppConfig

    cfg = AppConfig()
    assert cfg.server.max_concurrent_runs == 5

    custom = AppConfig.model_validate({"server": {"max_concurrent_runs": 2}})
    assert custom.server.max_concurrent_runs == 2


def test_tool_bus_registers_default_tools():
    from backend.tools import create_default_tool_bus

    bus = create_default_tool_bus()
    tools = bus.list_tools()
    assert "filesystem.read" in tools
    assert "filesystem.write" in tools
    assert "web.fetch" in tools
    assert "web.search" in tools
    assert "shell.exec" in tools


def test_model_router_selects_models():
    from backend.llm.router import ModelRouter, TaskType

    router = ModelRouter()
    assert router.select_model(TaskType.fast)
    assert router.select_model(TaskType.code)
    assert router.select_model(TaskType.reasoning)
    # Unknown type falls back to fast model
    assert router.select_model("unknown") == router.select_model(TaskType.fast)


def test_run_status_has_cancelled():
    from backend.models.db import RunStatus

    assert RunStatus.cancelled == "cancelled"
    statuses = {s.value for s in RunStatus}
    assert "cancelled" in statuses


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


def test_fastapi_routes_include_new_endpoints():
    """Cancel + SSE stream routes must be registered."""
    from backend.main import create_app

    app = create_app()
    paths = [r.path for r in app.routes]
    assert "/api/runs/{run_id}" in paths          # GET + DELETE
    assert "/api/runs/{run_id}/stream" in paths   # SSE


@pytest.mark.asyncio
async def test_empty_goal_rejected(tmp_path):
    """POST /api/runs with an empty goal must return 422."""
    import os
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db

    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/runs", json={"goal": ""})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"

        # Whitespace-only should also be rejected
        r2 = await client.post("/api/runs", json={"goal": "   "})
        assert r2.status_code == 422, f"Expected 422 for whitespace goal, got {r2.status_code}"


@pytest.mark.asyncio
async def test_cancel_run(tmp_path):
    """DELETE /api/runs/{id} marks the run as cancelled."""
    import os
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db

    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create a run
        r = await client.post("/api/runs", json={"goal": "Do something long"})
        assert r.status_code == 200
        run_id = r.json()["id"]

        # Cancel it immediately
        r2 = await client.delete(f"/api/runs/{run_id}")
        assert r2.status_code == 204

        # Check status
        r3 = await client.get(f"/api/runs/{run_id}")
        assert r3.json()["status"] == "cancelled"

        # Cancelling again returns 409
        r4 = await client.delete(f"/api/runs/{run_id}")
        assert r4.status_code == 409


@pytest.mark.asyncio
async def test_cancel_nonexistent_run(tmp_path):
    """DELETE /api/runs/{id} for unknown id returns 404."""
    import os
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db

    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.delete("/api/runs/no-such-id")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_sse_stream_returns_streaming_response(tmp_path):
    """GET /api/runs/{id}/stream returns a text/event-stream response."""
    import os
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db

    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Non-existent run should 404
        r404 = await client.get("/api/runs/no-such/stream")
        assert r404.status_code == 404

        # Create a run then immediately check stream headers
        rc = await client.post("/api/runs", json={"goal": "Stream test goal"})
        run_id = rc.json()["id"]

        async with client.stream("GET", f"/api/runs/{run_id}/stream") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            # Read at least one chunk to confirm the stream sends data
            chunks = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
                if len(chunks) >= 1:
                    break
            assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_concurrency_semaphore_limits_runs():
    """_get_semaphore respects max_concurrent_runs from config."""
    import backend.api.runs as runs_module
    from backend.config import AppConfig

    # Save originals
    original_sem = runs_module._RUN_SEMAPHORE
    original_get = runs_module.get_config

    # Reset semaphore and patch the get_config reference *inside* the runs module
    runs_module._RUN_SEMAPHORE = None
    runs_module.get_config = lambda: AppConfig.model_validate(
        {"server": {"max_concurrent_runs": 2}}
    )

    try:
        sem = runs_module._get_semaphore()
        assert sem._value == 2  # asyncio.Semaphore._value holds initial count

        # Same semaphore returned on subsequent calls
        assert runs_module._get_semaphore() is sem
    finally:
        runs_module.get_config = original_get
        runs_module._RUN_SEMAPHORE = original_sem


def test_topological_sort_cycle_safety():
    """_topological_sort must not hang on a cyclic dependency graph."""
    from backend.agents.orchestrator import Orchestrator
    from backend.models.db import Task

    def make(tid, deps):
        t = Task()
        t.id = tid
        t.title = tid
        t.depends_on = deps
        return t

    # Cycle: t1 -> t2 -> t1
    t1 = make("t1", ["t2"])
    t2 = make("t2", ["t1"])
    result = Orchestrator._topological_sort([t1, t2])
    assert len(result) == 2   # all tasks returned, no hang


def test_topological_waves_linear_chain():
    """Linear dependency chain produces one task per wave."""
    from backend.agents.orchestrator import Orchestrator
    from backend.models.db import Task

    def make(tid, deps):
        t = Task()
        t.id = tid
        t.depends_on = deps
        return t

    t1 = make("t1", [])
    t2 = make("t2", ["t1"])
    t3 = make("t3", ["t2"])

    waves = Orchestrator._topological_waves([t1, t2, t3])
    assert len(waves) == 3
    assert waves[0][0].id == "t1"
    assert waves[1][0].id == "t2"
    assert waves[2][0].id == "t3"


def test_topological_waves_parallel_detection():
    """Tasks with a shared dependency but not depending on each other are in the same wave."""
    from backend.agents.orchestrator import Orchestrator
    from backend.models.db import Task

    def make(tid, deps):
        t = Task()
        t.id = tid
        t.depends_on = deps
        return t

    # t1 → {t2, t3} → t4
    t1 = make("t1", [])
    t2 = make("t2", ["t1"])
    t3 = make("t3", ["t1"])
    t4 = make("t4", ["t2", "t3"])

    waves = Orchestrator._topological_waves([t1, t2, t3, t4])
    assert len(waves) == 3
    assert len(waves[0]) == 1   # t1 alone
    assert len(waves[1]) == 2   # t2 and t3 in parallel
    assert len(waves[2]) == 1   # t4 alone


def test_topological_waves_no_deps():
    """All independent tasks go into a single wave."""
    from backend.agents.orchestrator import Orchestrator
    from backend.models.db import Task

    def make(tid):
        t = Task()
        t.id = tid
        t.depends_on = []
        return t

    tasks = [make(f"t{i}") for i in range(4)]
    waves = Orchestrator._topological_waves(tasks)
    assert len(waves) == 1
    assert len(waves[0]) == 4


def test_topological_waves_cycle_safety():
    """_topological_waves must not hang on a cyclic graph."""
    from backend.agents.orchestrator import Orchestrator
    from backend.models.db import Task

    def make(tid, deps):
        t = Task()
        t.id = tid
        t.depends_on = deps
        return t

    t1 = make("t1", ["t2"])
    t2 = make("t2", ["t1"])
    waves = Orchestrator._topological_waves([t1, t2])
    # Should not hang; all tasks should appear in some wave
    flat = [t for w in waves for t in w]
    assert len(flat) == 2


def test_role_aware_model_routing():
    """_ROLE_TASK_TYPE maps roles to the correct model tier."""
    from backend.agents.tool_operator import _ROLE_TASK_TYPE
    from backend.llm.router import TaskType

    assert _ROLE_TASK_TYPE["coder"] == TaskType.code
    assert _ROLE_TASK_TYPE["researcher"] == TaskType.reasoning
    assert _ROLE_TASK_TYPE["analyst"] == TaskType.reasoning
    assert _ROLE_TASK_TYPE["writer"] == TaskType.reasoning
    assert _ROLE_TASK_TYPE["tool_operator"] == TaskType.fast
    assert _ROLE_TASK_TYPE["verifier"] == TaskType.fast


@pytest.mark.asyncio
async def test_tool_operator_uses_role_model(monkeypatch):
    """ToolOperator selects the code model for a coder task."""
    from backend.agents.tool_operator import ToolOperator
    from backend.llm.router import ModelRouter, TaskType
    from backend.models.db import Task
    from backend.tools.bus import ToolBus

    used_task_types: list[TaskType] = []

    class CapturingRouter:
        async def chat(self, task_type, messages, options=None):
            used_task_types.append(task_type)
            return '{"tool": null, "result": "done"}'

    task = Task()
    task.title = "Write a Python script"
    task.description = "hello world"
    task.depends_on = []
    task.agent_role = "coder"

    op = ToolOperator(CapturingRouter(), ToolBus())
    await op.execute_task(task, "")

    assert used_task_types[0] == TaskType.code


def test_planner_build_system_prompt_includes_tools():
    """_build_system_prompt injects available tools into the system prompt."""
    from backend.agents.planner import Planner
    from backend.llm.router import ModelRouter

    planner = Planner(ModelRouter())
    prompt = planner._build_system_prompt(
        available_tools=["web.search", "web.fetch"],
        memories=None,
    )
    assert "web.search" in prompt
    assert "web.fetch" in prompt


def test_planner_build_system_prompt_includes_memories():
    """_build_system_prompt injects past learnings into the system prompt."""
    from backend.agents.planner import Planner
    from backend.llm.router import ModelRouter

    planner = Planner(ModelRouter())
    prompt = planner._build_system_prompt(
        available_tools=None,
        memories=["Use DuckDuckGo for news; arXiv for papers."],
    )
    assert "DuckDuckGo" in prompt
    assert "arXiv" in prompt


def test_planner_parse_fallback():
    """Planner._parse_plan falls back gracefully on invalid JSON."""
    from backend.agents.planner import Planner
    from backend.llm.router import ModelRouter

    router = ModelRouter()
    planner = Planner(router)
    tasks = planner._parse_plan("not json at all", "run-123")
    assert len(tasks) == 1
    assert tasks[0].run_id == "run-123"


@pytest.mark.asyncio
async def test_memory_store_recall_empty(tmp_path):
    """MemoryStore.recall returns [] when no memories exist."""
    import os
    os.chdir(tmp_path)

    from backend.models.database import init_db
    from backend.agents.memory import MemoryStore
    from backend.llm.router import ModelRouter

    await init_db()
    store = MemoryStore(ModelRouter())
    memories = await store.recall("test goal")
    assert memories == []


@pytest.mark.asyncio
async def test_memory_store_record_and_recall(tmp_path):
    """Directly stored memories are returned by recall()."""
    import os
    os.chdir(tmp_path)
    import uuid

    from backend.models.database import init_db, session_scope
    from backend.agents.memory import MemoryStore
    from backend.models.db import Memory
    from backend.llm.router import ModelRouter

    await init_db()

    # Insert a memory directly (bypassing LLM summarisation)
    async with session_scope() as session:
        m = Memory(
            id=str(uuid.uuid4()),
            run_id="test-run",
            goal_summary="research AI papers news",
            learnings="arXiv is the best source for AI papers. web.fetch reliably extracts content.",
        )
        session.add(m)

    store = MemoryStore(ModelRouter())
    memories = await store.recall("research AI papers")
    assert len(memories) == 1
    assert "arXiv" in memories[0]


@pytest.mark.asyncio
async def test_memory_store_relevance_ordering(tmp_path):
    """More relevant memories (higher word overlap) appear first."""
    import os
    os.chdir(tmp_path)
    import uuid

    from backend.models.database import init_db, session_scope
    from backend.agents.memory import MemoryStore
    from backend.models.db import Memory
    from backend.llm.router import ModelRouter

    await init_db()

    async with session_scope() as session:
        session.add(Memory(
            id=str(uuid.uuid4()),
            run_id="r1",
            goal_summary="write python code script",
            learnings="shell.exec with python works well for code tasks.",
        ))
        session.add(Memory(
            id=str(uuid.uuid4()),
            run_id="r2",
            goal_summary="summarise cooking recipes",
            learnings="web.fetch retrieves recipe pages effectively.",
        ))

    store = MemoryStore(ModelRouter())
    memories = await store.recall("write a python script", limit=2)
    # The python-related memory should come first
    assert "python" in memories[0].lower() or "code" in memories[0].lower()


def test_tool_bus_describe_tools():
    """ToolBus.describe_tools returns name→description for all registered tools."""
    from backend.tools import create_default_tool_bus

    bus = create_default_tool_bus()
    descriptions = bus.describe_tools()
    assert isinstance(descriptions, dict)
    for name in bus.list_tools():
        assert name in descriptions
        assert isinstance(descriptions[name], str)
        assert len(descriptions[name]) > 0


def test_memory_config_defaults():
    """MemoryConfig has the expected default values."""
    from backend.config import AppConfig

    cfg = AppConfig()
    assert cfg.memory.enabled is True
    assert cfg.memory.recall_limit == 3
    assert cfg.memory.max_parallel_tasks == 4


def test_memory_model_exists():
    """Memory table is declared in the ORM."""
    from backend.models.db import Memory

    m = Memory()
    m.id = "test-id"
    m.goal_summary = "test"
    m.learnings = "test learnings"
    assert m.id == "test-id"


# ── New tests for 10/10 upgrades ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_fetch_extracts_text(tmp_path, monkeypatch):
    """web.fetch should return extracted text, not raw HTML."""
    import httpx
    from backend.tools.web import WebFetchTool

    html = (
        "<html><body>"
        "<nav>nav junk</nav>"
        "<article><p>Hello clean world!</p></article>"
        "<footer>footer junk</footer>"
        "</body></html>"
    )

    # Patch httpx to avoid real network calls
    class FakeResp:
        status_code = 200
        text = html

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def get(self, *a, **kw):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

    tool = WebFetchTool()
    result = await tool.execute({"url": "https://example.com"})
    assert result.success
    content = result.output["content"]
    assert result.output["extraction"] == "text"
    # The extracted text should be much shorter than raw HTML
    assert len(content) < len(html) or "Hello clean world" in content


@pytest.mark.asyncio
async def test_web_fetch_raw_mode(tmp_path, monkeypatch):
    """web.fetch with raw=true should return full HTML."""
    import httpx
    from backend.tools.web import WebFetchTool

    html = "<html><body><p>raw</p></body></html>"

    class FakeResp:
        status_code = 200
        text = html

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def get(self, *a, **kw):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeClient())

    tool = WebFetchTool()
    result = await tool.execute({"url": "https://example.com", "raw": "true"})
    assert result.success
    assert result.output["extraction"] == "raw"
    assert "<html>" in result.output["content"]


@pytest.mark.asyncio
async def test_web_search_tool_registered():
    """web.search must be in the default tool bus."""
    from backend.tools import create_default_tool_bus

    bus = create_default_tool_bus()
    assert "web.search" in bus.list_tools()


@pytest.mark.asyncio
async def test_web_search_missing_query():
    """web.search returns an error when query is missing."""
    from backend.tools.web import WebSearchTool

    tool = WebSearchTool()
    result = await tool.execute({})
    assert not result.success
    assert "query" in result.error.lower()


@pytest.mark.asyncio
async def test_tool_operator_react_retry(tmp_path, monkeypatch):
    """ToolOperator retries up to MAX_RETRIES on tool failure."""
    from backend.agents.tool_operator import ToolOperator, _MAX_RETRIES
    from backend.llm.router import ModelRouter, TaskType
    from backend.models.db import Task
    from backend.tools.bus import ToolBus, ToolResult

    call_count = 0

    # Router always returns a call to a fictional tool
    class FakeRouter:
        async def chat(self, task_type, messages, options=None):
            return '{"tool": "fake.tool", "params": {}}'

    # Tool always fails
    class FailingBus(ToolBus):
        async def call(self, tool_input):
            nonlocal call_count
            call_count += 1
            return ToolResult.from_error("fake.tool", "always fails")

    bus = FailingBus()

    task = Task()
    task.title = "Test task"
    task.description = ""
    task.depends_on = []

    op = ToolOperator(FakeRouter(), bus)
    outcome = await op.execute_task(task, "")

    # Should have retried _MAX_RETRIES times
    assert call_count == _MAX_RETRIES
    assert outcome["attempts"] == _MAX_RETRIES


@pytest.mark.asyncio
async def test_tool_operator_react_succeeds_on_second_attempt():
    """ToolOperator returns on first successful tool call."""
    from backend.agents.tool_operator import ToolOperator
    from backend.models.db import Task
    from backend.tools.bus import ToolBus, ToolResult

    call_count = 0

    class FakeRouter:
        async def chat(self, task_type, messages, options=None):
            return '{"tool": "fake.tool", "params": {}}'

    class SometimesBus(ToolBus):
        async def call(self, tool_input):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ToolResult.from_error("fake.tool", "first failure")
            return ToolResult.from_success("fake.tool", {"data": "ok"})

    task = Task()
    task.title = "Test task"
    task.description = ""
    task.depends_on = []

    op = ToolOperator(FakeRouter(), SometimesBus())
    outcome = await op.execute_task(task, "")

    assert call_count == 2
    assert outcome["attempts"] == 2
    assert outcome["kind"] == "tool"


@pytest.mark.asyncio
async def test_list_runs_pagination(tmp_path):
    """GET /api/runs respects limit and offset query params."""
    import os
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db

    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create 5 runs
        for i in range(5):
            await client.post("/api/runs", json={"goal": f"Goal {i}"})

        # Default (no params) returns all 5
        r_all = await client.get("/api/runs")
        assert r_all.status_code == 200
        assert len(r_all.json()) == 5

        # limit=2 returns 2
        r_limited = await client.get("/api/runs?limit=2")
        assert len(r_limited.json()) == 2

        # limit=2&offset=4 returns the last 1
        r_paged = await client.get("/api/runs?limit=2&offset=4")
        assert len(r_paged.json()) == 1


@pytest.mark.asyncio
async def test_api_key_auth_blocks_when_set(tmp_path, monkeypatch):
    """When AGENTIC_API_KEY is set, requests without the header get 401."""
    import os
    os.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_API_KEY", "secret-key")

    from httpx import AsyncClient, ASGITransport
    from backend.models.database import init_db

    await init_db()

    # Import AFTER setting env var so middleware picks it up
    import importlib, backend.main as main_mod
    importlib.reload(main_mod)
    app = main_mod.create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # No key → 401
        r = await client.get("/api/runs")
        assert r.status_code == 401

        # Wrong key → 401
        r2 = await client.get("/api/runs", headers={"X-API-Key": "wrong"})
        assert r2.status_code == 401

        # Correct key → 200
        r3 = await client.get("/api/runs", headers={"X-API-Key": "secret-key"})
        assert r3.status_code == 200

        # /api/health is always accessible
        r4 = await client.get("/api/health")
        assert r4.status_code == 200


@pytest.mark.asyncio
async def test_api_key_auth_open_when_not_set(tmp_path, monkeypatch):
    """When AGENTIC_API_KEY is not set, all requests pass through."""
    import os
    os.chdir(tmp_path)
    monkeypatch.delenv("AGENTIC_API_KEY", raising=False)

    from httpx import AsyncClient, ASGITransport
    from backend.models.database import init_db
    import importlib, backend.main as main_mod
    importlib.reload(main_mod)
    app = main_mod.create_app()

    await init_db()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/runs")
        assert r.status_code == 200


# ── Debate / Vote tests ───────────────────────────────────────────────────────

def test_debate_config_defaults():
    from backend.config import AppConfig

    cfg = AppConfig()
    assert cfg.debate.enabled is False
    assert cfg.debate.num_candidates == 3
    assert cfg.debate.temperature == 0.8


def test_debate_config_enable():
    from backend.config import AppConfig

    cfg = AppConfig.model_validate({"debate": {"enabled": True, "num_candidates": 5}})
    assert cfg.debate.enabled is True
    assert cfg.debate.num_candidates == 5


@pytest.mark.asyncio
async def test_debater_returns_candidates():
    """Debater collects LLM responses as candidate strings."""
    from backend.agents.debater import Debater

    call_count = 0

    class FakeRouter:
        async def chat(self, task_type, messages, options=None):
            nonlocal call_count
            call_count += 1
            return '{"tasks": [{"id": "t1", "title": "Do something", "description": "", "agent_role": "researcher", "depends_on": []}]}'

    d = Debater(FakeRouter())
    candidates = await d.generate_candidates("my goal", n=3, temperature=0.8)
    assert call_count == 3
    assert len(candidates) == 3


@pytest.mark.asyncio
async def test_debater_skips_exceptions():
    """Debater silently drops failed LLM calls."""
    from backend.agents.debater import Debater

    call_count = 0

    class FlakyRouter:
        async def chat(self, task_type, messages, options=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            return '{"tasks": [{"id": "t1", "title": "T", "description": "", "agent_role": "researcher", "depends_on": []}]}'

    d = Debater(FlakyRouter())
    candidates = await d.generate_candidates("goal", n=3)
    # call_count should be 3, but only 2 non-exceptional results
    assert len(candidates) == 2


@pytest.mark.asyncio
async def test_voter_picks_winner():
    """Voter calls LLM and picks the plan at the returned index."""
    from backend.agents.voter import Voter

    class FakeRouter:
        async def chat(self, task_type, messages, options=None):
            return '{"winner": 1, "reason": "Plan 1 is best"}'

    candidates = [
        '{"tasks": [{"id": "t1", "title": "Plan A task", "description": "", "agent_role": "researcher", "depends_on": []}]}',
        '{"tasks": [{"id": "t1", "title": "Plan B task", "description": "", "agent_role": "analyst", "depends_on": []}]}',
    ]
    v = Voter(FakeRouter())
    tasks = await v.select_best("some goal", "run-1", candidates)
    assert len(tasks) == 1
    assert tasks[0].title == "Plan B task"
    assert tasks[0].agent_role == "analyst"


@pytest.mark.asyncio
async def test_voter_fallback_on_bad_json():
    """Voter falls back to index 0 when LLM returns bad JSON."""
    from backend.agents.voter import Voter

    class BadRouter:
        async def chat(self, task_type, messages, options=None):
            return "I cannot decide."

    candidates = [
        '{"tasks": [{"id": "t1", "title": "Fallback task", "description": "", "agent_role": "researcher", "depends_on": []}]}',
        '{"tasks": [{"id": "t1", "title": "Other task", "description": "", "agent_role": "analyst", "depends_on": []}]}',
    ]
    v = Voter(BadRouter())
    tasks = await v.select_best("goal", "run-1", candidates)
    assert tasks[0].title == "Fallback task"


@pytest.mark.asyncio
async def test_voter_single_candidate_skips_llm():
    """Voter skips the LLM call when only one candidate is provided."""
    from backend.agents.voter import Voter

    call_count = 0

    class TrackingRouter:
        async def chat(self, task_type, messages, options=None):
            nonlocal call_count
            call_count += 1
            return '{"winner": 0, "reason": "only one"}'

    candidates = [
        '{"tasks": [{"id": "t1", "title": "Solo", "description": "", "agent_role": "researcher", "depends_on": []}]}'
    ]
    v = Voter(TrackingRouter())
    tasks = await v.select_best("goal", "run-1", candidates)
    assert call_count == 0
    assert tasks[0].title == "Solo"


@pytest.mark.asyncio
async def test_voter_empty_candidates():
    """Voter returns empty list for empty candidates."""
    from backend.agents.voter import Voter

    class FakeRouter:
        async def chat(self, task_type, messages, options=None):
            return '{"winner": 0, "reason": ""}'

    v = Voter(FakeRouter())
    tasks = await v.select_best("goal", "run-1", [])
    assert tasks == []


# ── Training data export tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_training_data_empty(tmp_path):
    """Export endpoint returns empty JSONL when there are no completed runs."""
    import os
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db

    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/runs/export/training-data")
        assert r.status_code == 200
        assert r.text.strip() == ""


@pytest.mark.asyncio
async def test_export_training_data_bad_format(tmp_path):
    """Export endpoint rejects unknown format strings."""
    import os
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db

    await init_db()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/runs/export/training-data?format=invalid")
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_export_training_data_with_verified_run(tmp_path):
    """Export endpoint returns a JSONL line for a verified, completed run/task."""
    import json
    import os
    import uuid
    os.chdir(tmp_path)

    from backend.models.database import init_db, get_session_factory
    from backend.models.db import Run, RunStatus, Task, TaskStatus, Event, EventKind
    from backend.main import create_app
    from httpx import AsyncClient, ASGITransport

    await init_db()

    # Seed a completed run with a verified task
    factory = get_session_factory()
    run_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    async with factory() as session:
        run = Run(id=run_id, goal="Test goal for export", status=RunStatus.completed)
        session.add(run)
        task = Task(
            id=task_id,
            run_id=run_id,
            title="Do the test",
            description="desc",
            agent_role="researcher",
            status=TaskStatus.completed,
            result="great result",
            depends_on=[],
        )
        session.add(task)
        # Verification event with high confidence
        ev = Event(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            kind=EventKind.verification,
            agent_role="verifier",
            content="verified",
            data={"verified": True, "confidence": 90, "notes": "good"},
        )
        session.add(ev)
        await session.commit()

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Alpaca format
        r = await client.get("/api/runs/export/training-data?format=alpaca&min_confidence=70")
        assert r.status_code == 200
        lines = [l for l in r.text.strip().split("\n") if l]
        assert len(lines) == 1
        example = json.loads(lines[0])
        assert example["instruction"] == "Test goal for export"
        assert example["input"] == ""
        output = json.loads(example["output"])
        assert output["tasks"][0]["title"] == "Do the test"
        assert output["tasks"][0]["result"] == "great result"

        # ShareGPT format
        r2 = await client.get("/api/runs/export/training-data?format=sharegpt&min_confidence=70")
        assert r2.status_code == 200
        lines2 = [l for l in r2.text.strip().split("\n") if l]
        ex2 = json.loads(lines2[0])
        assert ex2["conversations"][0]["from"] == "human"
        assert ex2["conversations"][0]["value"] == "Test goal for export"
        assert ex2["conversations"][1]["from"] == "gpt"


@pytest.mark.asyncio
async def test_export_training_data_filters_low_confidence(tmp_path):
    """Tasks below min_confidence are excluded from the export."""
    import json
    import os
    import uuid
    os.chdir(tmp_path)

    from backend.models.database import init_db, get_session_factory
    from backend.models.db import Run, RunStatus, Task, TaskStatus, Event, EventKind
    from backend.main import create_app
    from httpx import AsyncClient, ASGITransport

    await init_db()

    factory = get_session_factory()
    run_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    async with factory() as session:
        run = Run(id=run_id, goal="Low confidence goal", status=RunStatus.completed)
        session.add(run)
        task = Task(
            id=task_id,
            run_id=run_id,
            title="Low conf task",
            description="",
            agent_role="analyst",
            status=TaskStatus.completed,
            result="meh",
            depends_on=[],
        )
        session.add(task)
        # Verification event with LOW confidence (40)
        ev = Event(
            id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task_id,
            kind=EventKind.verification,
            agent_role="verifier",
            content="low",
            data={"verified": True, "confidence": 40, "notes": "weak"},
        )
        session.add(ev)
        await session.commit()

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/runs/export/training-data?min_confidence=70")
        assert r.status_code == 200
        # Run has no tasks that meet the threshold → empty output
        assert r.text.strip() == ""


# ── Reflexion agent tests ─────────────────────────────────────────────────────

def test_reflexion_config_defaults():
    from backend.config import AppConfig

    cfg = AppConfig()
    assert cfg.reflexion.enabled is True
    assert cfg.reflexion.min_confidence == 60
    assert cfg.reflexion.max_retries == 2


def test_reflexion_config_custom():
    from backend.config import AppConfig

    cfg = AppConfig.model_validate({"reflexion": {"enabled": False, "min_confidence": 80, "max_retries": 3}})
    assert cfg.reflexion.enabled is False
    assert cfg.reflexion.min_confidence == 80
    assert cfg.reflexion.max_retries == 3


@pytest.mark.asyncio
async def test_reflexion_agent_returns_correction():
    """ReflexionAgent returns a non-empty correction string from the LLM."""
    from backend.agents.reflexion import ReflexionAgent
    from backend.models.db import Task

    class FakeRouter:
        async def chat(self, task_type, messages, options=None):
            return "Try using web.search with a more specific query."

    agent = ReflexionAgent(FakeRouter())
    task = Task(
        id="t1",
        run_id="r1",
        title="Research AI news",
        description="Find the latest AI news",
        agent_role="researcher",
        depends_on=[],
    )
    correction = await agent.reflect(
        task,
        result="Nothing found",
        verification={"verified": False, "confidence": 20, "notes": "No evidence found"},
    )
    assert correction == "Try using web.search with a more specific query."


@pytest.mark.asyncio
async def test_reflexion_agent_handles_error():
    """ReflexionAgent returns empty string when LLM raises."""
    from backend.agents.reflexion import ReflexionAgent
    from backend.models.db import Task

    class ErrorRouter:
        async def chat(self, task_type, messages, options=None):
            raise RuntimeError("LLM unavailable")

    agent = ReflexionAgent(ErrorRouter())
    task = Task(
        id="t1",
        run_id="r1",
        title="Do something",
        description="",
        agent_role="researcher",
        depends_on=[],
    )
    correction = await agent.reflect(task, result="bad", verification={"verified": False})
    assert correction == ""


# ── Synthesizer agent tests ───────────────────────────────────────────────────

def test_synthesis_config_defaults():
    from backend.config import AppConfig

    cfg = AppConfig()
    assert cfg.synthesis.enabled is True


def test_synthesis_config_disable():
    from backend.config import AppConfig

    cfg = AppConfig.model_validate({"synthesis": {"enabled": False}})
    assert cfg.synthesis.enabled is False


@pytest.mark.asyncio
async def test_synthesizer_returns_summary():
    """Synthesizer produces a coherent answer from task results."""
    from backend.agents.synthesizer import Synthesizer
    from backend.models.db import Task

    class FakeRouter:
        async def chat(self, task_type, messages, options=None):
            return "AI is advancing rapidly with new models released every month."

    synth = Synthesizer(FakeRouter())
    tasks = [
        Task(
            id="t1",
            run_id="r1",
            title="Research task",
            description="",
            agent_role="researcher",
            status="completed",
            result="New AI models released in March 2026.",
            depends_on=[],
        )
    ]
    summary = await synth.synthesize("What is new in AI?", tasks)
    assert "AI" in summary


@pytest.mark.asyncio
async def test_synthesizer_skips_empty_results():
    """Synthesizer returns empty string when all task results are empty."""
    from backend.agents.synthesizer import Synthesizer
    from backend.models.db import Task

    called = False

    class TrackingRouter:
        async def chat(self, task_type, messages, options=None):
            nonlocal called
            called = True
            return ""

    synth = Synthesizer(TrackingRouter())
    tasks = [
        Task(
            id="t1", run_id="r1", title="Empty task", description="",
            agent_role="researcher", status="completed", result="", depends_on=[],
        )
    ]
    summary = await synth.synthesize("goal", tasks)
    # LLM should not be called if there are no results to synthesize
    assert not called
    assert summary == ""


@pytest.mark.asyncio
async def test_synthesizer_handles_error():
    """Synthesizer returns empty string on LLM error."""
    from backend.agents.synthesizer import Synthesizer
    from backend.models.db import Task

    class ErrorRouter:
        async def chat(self, task_type, messages, options=None):
            raise RuntimeError("LLM down")

    synth = Synthesizer(ErrorRouter())
    tasks = [
        Task(
            id="t1", run_id="r1", title="T", description="",
            agent_role="researcher", status="completed", result="some result", depends_on=[],
        )
    ]
    result = await synth.synthesize("goal", tasks)
    assert result == ""


# ── ToolOperator dynamic tool descriptions ────────────────────────────────────

def test_tool_operator_dynamic_system_prompt():
    """ToolOperator builds system prompt from live tool registry."""
    from backend.agents.tool_operator import ToolOperator
    from backend.tools.bus import BaseTool, ToolBus, ToolResult
    from backend.llm.router import ModelRouter

    class FakeTool(BaseTool):
        name = "my.special_tool"
        description = "Does something special with data"

        async def execute(self, params):
            return ToolResult.from_success(self.name, {})

    bus = ToolBus()
    bus.register(FakeTool())
    op = ToolOperator(ModelRouter(), bus)
    prompt = op._build_system_prompt()
    assert "my.special_tool" in prompt
    assert "Does something special with data" in prompt


def test_tool_operator_includes_reflection_in_prompt():
    """execute_task injects reflection text into the user message."""
    from backend.agents.tool_operator import ToolOperator
    from backend.tools.bus import ToolBus
    from backend.llm.router import ModelRouter
    from backend.models.db import Task

    captured_messages = []

    class CapturingRouter:
        def select_model(self, task_type):
            return "test-model"

        async def chat(self, task_type, messages, options=None):
            captured_messages.extend(messages)
            return '{"tool": null, "result": "done"}'

    bus = ToolBus()
    op = ToolOperator(CapturingRouter(), bus)
    task = Task(
        id="t1", run_id="r1", title="Task", description="desc",
        agent_role="researcher", depends_on=[],
    )

    import asyncio
    asyncio.get_event_loop().run_until_complete(
        op.execute_task(task, context="{}", reflection="Use a different URL next time")
    )

    user_msg = next(m for m in captured_messages if m["role"] == "user")
    assert "Use a different URL next time" in user_msg["content"]


# ── Run summary field ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_summary_field(tmp_path):
    """Run model stores and retrieves summary correctly."""
    import os, uuid
    os.chdir(tmp_path)

    from backend.models.database import init_db, session_scope
    from backend.models.db import Run, RunStatus

    await init_db()

    run_id = str(uuid.uuid4())
    async with session_scope() as session:
        run = Run(id=run_id, goal="A goal", status=RunStatus.completed, summary="Final answer here.")
        session.add(run)

    async with session_scope() as session:
        from sqlalchemy import select
        result = await session.execute(select(Run).where(Run.id == run_id))
        fetched = result.scalar_one_or_none()
        assert fetched is not None
        assert fetched.summary == "Final answer here."


@pytest.mark.asyncio
async def test_run_response_includes_summary(tmp_path):
    """GET /api/runs/{id} returns summary field."""
    import os, uuid
    os.chdir(tmp_path)

    from httpx import AsyncClient, ASGITransport
    from backend.main import create_app
    from backend.models.database import init_db, get_session_factory
    from backend.models.db import Run, RunStatus

    await init_db()
    factory = get_session_factory()
    run_id = str(uuid.uuid4())
    async with factory() as session:
        run = Run(id=run_id, goal="A test goal", status=RunStatus.completed, summary="The final answer.")
        session.add(run)
        await session.commit()

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(f"/api/runs/{run_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["summary"] == "The final answer."


# ── Embedding-based memory recall ─────────────────────────────────────────────

def test_cosine_similarity():
    """Cosine similarity function is correct for known vectors."""
    from backend.agents.memory import _cosine

    # Identical vectors → similarity 1.0
    assert abs(_cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6

    # Orthogonal vectors → similarity 0.0
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6

    # Opposite vectors → similarity -1.0
    assert abs(_cosine([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-6


def test_cosine_zero_vector():
    """Cosine similarity returns 0.0 for zero-length vectors."""
    from backend.agents.memory import _cosine

    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine([], []) == 0.0


# ── EventKind includes new kinds ─────────────────────────────────────────────

def test_event_kind_includes_reflexion_and_synthesis():
    from backend.models.db import EventKind

    kinds = {k.value for k in EventKind}
    assert "reflexion" in kinds
    assert "synthesis" in kinds
