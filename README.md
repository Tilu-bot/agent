# ⬡ agentic

**Local-first, Docker-sandboxed Multi-Agent AI Runtime** — v0.1 (MVP)

A minimal but cleanly architected system where multiple AI agents collaborate to achieve a goal. Designed for local execution on Windows with 4 GB VRAM via Ollama.

---

## Features (v0.1)

| Feature | Status |
|---|---|
| Session / run creation | ✅ |
| Task DAG (planning → execution) | ✅ |
| Event log (persisted in SQLite) | ✅ |
| Agent roles: Orchestrator, Planner, ToolOperator, Verifier | ✅ |
| Tool bus + tools: filesystem.read/write, web.fetch, shell.exec | ✅ |
| shell.exec via Docker sandbox (no direct host execution) | ✅ |
| Ollama integration + model router (fast/reasoning/code) | ✅ |
| Artifact store with SHA-256 provenance | ✅ |
| Web UI (Next.js): run list, timeline, task list, evidence panel | ✅ |
| YAML configuration | ✅ |

---

## Prerequisites (Windows)

### 1. Install Docker Desktop
- Download from <https://www.docker.com/products/docker-desktop/>
- Enable WSL 2 backend during installation
- After install, verify: `docker --version`

### 2. Install Ollama
- Download from <https://ollama.com/download>
- After install, pull the default models:

```powershell
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
```

Ollama will automatically use your GPU if available (4 GB VRAM is sufficient for 3B models).

### 3. Install Python 3.11+
- Download from <https://www.python.org/downloads/>
- Check "Add to PATH" during install
- Verify: `python --version`

### 4. Install Node.js 20+
- Download from <https://nodejs.org/>
- Verify: `node --version`

---

## Setup

### Clone the repo

```powershell
git clone https://github.com/Tilu-bot/agent.git
cd agent
```

### Build the Docker sandbox image

```powershell
docker build -t agentic-sandbox:latest ./sandbox
```

### Backend setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate      # On Windows
pip install -r requirements.txt
cd ..
```

### Frontend setup

```powershell
cd frontend
npm install
cd ..
```

---

## Running

Open two PowerShell/terminal windows:

**Terminal 1 — Backend**

```powershell
cd backend
.venv\Scripts\activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend**

```powershell
cd frontend
npm run dev
```

Then open your browser at **<http://localhost:3000>**

---

## Configuration

Edit `backend/config.yaml` to customise models, tools, and sandbox settings:

```yaml
models:
  fast: "llama3.2:3b"        # Used for quick/routing tasks
  reasoning: "llama3.2:3b"   # Used for planning
  code: "qwen2.5-coder:3b"   # Used for code tasks

ollama:
  base_url: "http://localhost:11434"

tools:
  filesystem:
    allow_write: false        # Set true to enable file writes
  shell:
    enabled: true
    docker_image: "agentic-sandbox:latest"
    timeout_seconds: 30
    memory_limit: "256m"
```

---

## How it works

```
Goal
 │
 ▼
Orchestrator ──► Planner (creates Task DAG)
 │
 ▼ (for each task in topological order)
ToolOperator ──► Tool Bus ──► [filesystem.read / web.fetch / shell.exec / ...]
 │
 ▼
Verifier (checks evidence, marks verified/unverified)
 │
 ▼
Event log + Artifact store (SQLite + disk)
```

All events are stored in `.agentic/agentic.db` (SQLite).  
Artifacts (tool outputs) are stored in `.agentic/artifacts/` with SHA-256 hashes.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/runs` | Create a new run |
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/{id}` | Get run details |
| GET | `/api/runs/{id}/tasks` | Get tasks for a run |
| GET | `/api/runs/{id}/events` | Get event timeline |
| GET | `/api/runs/{id}/artifacts` | Get artifacts |
| GET | `/api/runs/{id}/artifacts/{aid}/content` | Get artifact content |
| GET | `/api/health` | Health check (includes Ollama status) |

Interactive API docs are available at <http://localhost:8000/docs>

---

## Docker Compose (optional, full stack)

```powershell
docker compose up --build
```

Note: Requires Ollama running on the host at port 11434.

---

## Architecture (designed to grow)

```
backend/
├── agents/
│   ├── orchestrator.py   # Goal coordination, run lifecycle
│   ├── planner.py        # Goal → Task DAG
│   ├── tool_operator.py  # Tool selection + execution
│   └── verifier.py       # Evidence checking
├── llm/
│   ├── ollama_client.py  # Ollama HTTP client
│   └── router.py         # Task-type → model selection
├── tools/
│   ├── bus.py            # Tool bus interface
│   ├── filesystem.py     # Read/write tools
│   ├── web.py            # HTTP fetch tool
│   └── shell.py          # Docker-sandboxed shell
├── models/
│   ├── db.py             # SQLAlchemy ORM models
│   └── database.py       # Async session management
├── api/
│   └── runs.py           # FastAPI routes
├── config.py             # Pydantic config from YAML
├── config.yaml           # User configuration
└── main.py               # FastAPI app entry point
frontend/
└── src/
    ├── app/
    │   ├── page.tsx          # Home: run list + create form
    │   └── runs/[id]/page.tsx # Run detail: tasks + timeline + artifacts
    └── lib/
        └── api.ts            # API client
sandbox/
└── Dockerfile            # Minimal Alpine image for shell.exec
```

Future phases (not in MVP):
- Long-term memory with vector DB (Qdrant/Chroma)
- Multi-agent spawning (Researcher, Coder, Critic)
- Browser automation (Playwright)
- GitHub tool integration
- Model router learned from logs

---

## Troubleshooting

**Ollama shows offline in UI**  
Make sure Ollama is running: `ollama serve`

**shell.exec fails with "Docker not found"**  
Make sure Docker Desktop is running and `docker` is in your PATH.

**"agentic-sandbox:latest" not found**  
Build it: `docker build -t agentic-sandbox:latest ./sandbox`

**Models responding slowly**  
With 4 GB VRAM, 3B models should run at reasonable speed. If too slow, try `llama3.2:1b` (edit config.yaml).
