# ⬡ agentic

**Local-first, open-source Multi-Agent AI Runtime** — powered by [Ollama](https://ollama.com) (free, runs on your GPU/CPU, no API key needed).

---

## ⚡ Quick Start — One Click

### Windows

1. **Install prerequisites** (one-time):
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) — for the secure sandbox
   - [Ollama](https://ollama.com/download) — free local LLM server
   - [Python 3.11+](https://www.python.org/downloads/) ← check "Add to PATH"
   - [Node.js 20+](https://nodejs.org/)

2. **Double-click `install.bat`** → sets up everything automatically (venv, deps, models)

3. **Double-click `start.bat`** → starts backend + frontend, opens your browser

4. **Double-click `stop.bat`** → cleanly stops everything when you're done

---

### Linux / macOS

```bash
# 1. Install: (one time, ~5 minutes on first run to pull models)
chmod +x install.sh start.sh stop.sh
./install.sh

# 2. Start (opens browser automatically):
./start.sh

# 3. Stop (Ctrl+C in the terminal, or in a new terminal):
./stop.sh
```

---

### Docker Compose (advanced — truly one command)

```bash
docker compose up --build
```

This starts Ollama, pulls the models, builds the sandbox, backend and frontend — everything in one shot.  
Then open **http://localhost:3000**.

> **GPU (NVIDIA):** uncomment the `deploy: resources:` block in `docker-compose.yml`.

---

## What it does

A minimal, cleanly-architected system where AI agents collaborate to achieve a goal:

```
Goal
 │
 ▼
Orchestrator ──► Planner (creates Task DAG)
 │
 ▼ (for each task in topological order)
ToolOperator ──► Tool Bus ──► [filesystem.read / web.fetch / web.search / shell.exec / ...]
 │   (ReAct retry loop — self-corrects on tool failure, up to 3 retries)
 ▼
Verifier (checks evidence, marks verified/unverified)
 │
 ▼
Event log + Artifact store (SQLite + disk)
```

All data lives in `.agentic/` (SQLite DB + artifact files).

---

## Features

| Feature | Status |
|---|---|
| Session / run creation | ✅ |
| Task DAG (planning → execution) | ✅ |
| Event log (persisted in SQLite) | ✅ |
| Agent roles: Orchestrator, Planner, ToolOperator, Verifier | ✅ |
| Tool bus: filesystem.read/write, web.fetch, web.search, shell.exec | ✅ |
| **ReAct retry loop** — self-corrects on tool failure | ✅ |
| **HTML extraction** via trafilatura (clean text, not raw HTML) | ✅ |
| **Web search** via DuckDuckGo (no API key) | ✅ |
| shell.exec via Docker sandbox (no direct host execution) | ✅ |
| Ollama integration + model router (fast / reasoning / code) | ✅ |
| Artifact store with SHA-256 provenance | ✅ |
| Run list pagination (`?limit=&offset=`) | ✅ |
| Optional API key auth (`AGENTIC_API_KEY` env var) | ✅ |
| Web UI (Next.js): run list, timeline, task list, evidence panel | ✅ |
| YAML configuration | ✅ |
| **One-click install + start scripts** (Windows + Linux/macOS) | ✅ |
| **Docker Compose** with Ollama + auto model pull | ✅ |

---

## Configuration

Edit `backend/config.yaml` to change models or tool settings:

```yaml
models:
  fast: "llama3.2:3b"        # Quick tasks
  reasoning: "llama3.2:3b"   # Planning
  code: "qwen2.5-coder:3b"   # Code tasks

ollama:
  base_url: "http://localhost:11434"

tools:
  filesystem:
    allow_write: false        # Set true to enable file writes
  shell:
    enabled: true
    docker_image: "agentic-sandbox:latest"
    memory_limit: "256m"
```

Slower GPU / CPU only? Switch to 1B models:
```yaml
models:
  fast: "llama3.2:1b"
  reasoning: "llama3.2:1b"
  code: "qwen2.5-coder:1.5b"
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/runs` | Create a new run |
| GET | `/api/runs?limit=50&offset=0` | List runs (paginated) |
| GET | `/api/runs/{id}` | Get run details |
| DELETE | `/api/runs/{id}` | Cancel a run |
| GET | `/api/runs/{id}/stream` | Server-Sent Events stream |
| GET | `/api/runs/{id}/tasks` | Get tasks |
| GET | `/api/runs/{id}/events` | Get event timeline |
| GET | `/api/runs/{id}/artifacts` | Get artifacts |
| GET | `/api/health` | Health check (includes Ollama status) |

Interactive docs: **http://localhost:8000/docs**

---

## Architecture

```
backend/
├── agents/
│   ├── orchestrator.py   # Goal coordination, run lifecycle
│   ├── planner.py        # Goal → Task DAG
│   ├── tool_operator.py  # Tool selection + ReAct retry loop
│   └── verifier.py       # Evidence checking
├── llm/
│   ├── ollama_client.py  # Ollama HTTP client
│   └── router.py         # Task-type → model selection
├── tools/
│   ├── bus.py            # Tool registry & dispatcher
│   ├── filesystem.py     # Read/write tools
│   ├── web.py            # HTTP fetch (trafilatura) + DuckDuckGo search
│   └── shell.py          # Docker-sandboxed shell exec
├── models/
│   ├── db.py             # SQLAlchemy ORM models
│   └── database.py       # Async session management
├── api/
│   └── runs.py           # FastAPI routes
├── config.py             # Pydantic config from YAML
├── config.yaml           # Local config (edit this)
├── config.docker.yaml    # Docker Compose config (Ollama via service name)
└── main.py               # FastAPI app + API key middleware
frontend/
└── src/app/              # Next.js UI
sandbox/
└── Dockerfile            # Alpine image for shell.exec isolation
install.bat / install.sh  # One-click installer
start.bat   / start.sh    # One-click launcher
stop.bat    / stop.sh     # One-click stopper
docker-compose.yml        # Full stack (Ollama included)
```

---

## Troubleshooting

**`install.bat` says Ollama not found**  
Download and install from https://ollama.com/download — it's free and runs on CPU too.

**Ollama shows offline in UI**  
Run `ollama serve` in a terminal, or restart Ollama from the system tray.

**shell.exec fails with "Docker not found"**  
Start Docker Desktop. If you don't have it, disable the tool in `config.yaml`:
```yaml
tools:
  shell:
    enabled: false
```

**"agentic-sandbox:latest" not found**  
Run `docker build -t agentic-sandbox:latest ./sandbox` once, or re-run `install.bat` / `install.sh`.

**Models responding slowly**  
4 GB VRAM is enough for 3B models. On CPU-only machines try 1B models — edit `config.yaml` (see above).

**Port already in use**  
Run `stop.bat` / `stop.sh` to free ports, then `start.bat` / `start.sh` again.

