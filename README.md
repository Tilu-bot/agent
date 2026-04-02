# ⬡ agentic

**Local-first, open-source Multi-Agent AI Runtime** — powered by [Ollama](https://ollama.com) (free, runs on your GPU/CPU, no API key needed).

---

## ⚡ Quick Start — One Click, Zero Prerequisites

Download the repository as a ZIP from GitHub (**Code → Download ZIP**), unzip it, then:

### Windows

**Double-click `install.bat`**

That's it. The installer automatically downloads and installs:
- Python 3.11
- Node.js 20
- Ollama (local AI engine)
- Docker Desktop (optional — enables the secure code sandbox)
- All Python and Node packages
- AI models (~4 GB one-time download: `llama3.2:3b`, `qwen2.5-coder:3b`, `nomic-embed-text`)

It also creates an **"Agentic AI"** shortcut on your Desktop.

> **Note:** `install.bat` asks for administrator access once (needed to install software). Click **Yes** when prompted.

After installation:
- **Double-click `start.bat`** → starts everything, opens your browser automatically
- **Double-click `stop.bat`** → cleanly stops everything when you're done

---

### Linux / macOS

```bash
# 1. Install + download everything (one time, ~5 minutes on first run):
chmod +x install.sh start.sh stop.sh
./install.sh

# 2. Start (opens browser automatically):
./start.sh

# 3. Stop:
./stop.sh
```

The script auto-installs Python, Node.js, Ollama, and all dependencies — no manual steps needed.

---

### Docker Compose (advanced — truly one command)

```bash
docker compose up --build
```

Starts Ollama, pulls the models, builds the sandbox, backend and frontend — everything in one shot.  
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
| Agent roles: Orchestrator, Planner, ToolOperator, Verifier, Critic, Voter | ✅ |
| Tool bus: filesystem.read/write, web.fetch, web.search, shell.exec | ✅ |
| **ReAct retry loop** — self-corrects on tool failure | ✅ |
| **HTML extraction** via trafilatura (clean text, not raw HTML) | ✅ |
| **Web search** via DuckDuckGo (no API key) | ✅ |
| shell.exec via Docker sandbox (no direct host execution) | ✅ |
| Ollama integration + model router (fast / reasoning / code / search / math / vision / embedding) | ✅ |
| **In-app model pull** with live download progress bar | ✅ |
| Per-run model overrides via API | ✅ |
| Artifact store with SHA-256 provenance + file download | ✅ |
| Run list pagination (`?limit=&offset=`) | ✅ |
| Run cancellation | ✅ |
| Token usage tracking per run | ✅ |
| Global stats API | ✅ |
| Optional API key auth (`AGENTIC_API_KEY` env var) | ✅ |
| Training data export (JSONL — alpaca / sharegpt format) | ✅ |
| One-click fine-tuning from verified run data | ✅ |
| Web UI (Next.js): run list, timeline, task list, evidence panel | ✅ |
| YAML configuration | ✅ |
| **One-click install + start scripts** (Windows + Linux/macOS) — zero prerequisites | ✅ |
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

### Runs

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/runs` | Create a new run (body: `{"goal": "...", "models": {...}}`) |
| GET | `/api/runs?limit=50&offset=0` | List runs (paginated) |
| GET | `/api/runs/{id}` | Get run details |
| DELETE | `/api/runs/{id}` | Cancel a run |
| GET | `/api/runs/{id}/stream` | Server-Sent Events stream |
| GET | `/api/runs/{id}/tasks` | Get tasks |
| GET | `/api/runs/{id}/events` | Get event timeline |
| GET | `/api/runs/{id}/artifacts` | List artifacts |
| GET | `/api/runs/{id}/artifacts/{artifact_id}/content` | Get artifact content (add `?as_download=true` to save as file) |
| GET | `/api/runs/{id}/stats` | Per-run statistics |
| GET | `/api/runs/export/training-data` | Export verified runs as JSONL (alpaca / sharegpt) |

### Models

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/models` | List available models and current slot assignments |
| PUT | `/api/models/slots` | Update model slot assignments |
| DELETE | `/api/models/slots` | Reset slots to config defaults |
| GET | `/api/models/pull?model=<tag>` | Pull a model from Ollama (SSE progress stream) |

### System

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check (includes Ollama status) |
| GET | `/api/stats` | Global aggregate statistics |
| POST | `/api/training/start` | Start a fine-tuning job from verified run data |
| GET | `/api/training/status` | Status of the current / most recent training job |
| GET | `/api/training/jobs` | List all training jobs |

Interactive docs: **http://localhost:8000/docs**

---

## Architecture

```
backend/
├── agents/
│   ├── orchestrator.py   # Goal coordination, run lifecycle
│   ├── planner.py        # Goal → Task DAG
│   ├── tool_operator.py  # Tool selection + ReAct retry loop
│   ├── verifier.py       # Evidence checking
│   ├── critic.py         # Synthesis quality evaluation
│   └── voter.py          # Multi-agent voting
├── llm/
│   ├── ollama_client.py  # Ollama HTTP client + model pull streaming
│   ├── router.py         # Task-type → model selection
│   └── json_utils.py     # Robust LLM JSON extraction
├── tools/
│   ├── bus.py            # Tool registry & dispatcher
│   ├── filesystem.py     # Read/write tools
│   ├── web.py            # HTTP fetch (trafilatura) + DuckDuckGo search
│   └── shell.py          # Docker-sandboxed shell exec
├── models/
│   ├── db.py             # SQLAlchemy ORM models
│   └── database.py       # Async session management
├── api/
│   ├── runs.py           # FastAPI routes — runs, tasks, events, artifacts
│   ├── models.py         # FastAPI routes — model management + pull
│   └── training.py       # FastAPI routes — fine-tuning
├── config.py             # Pydantic config from YAML
├── config.yaml           # Local config (edit this)
├── config.docker.yaml    # Docker Compose config (Ollama via service name)
└── main.py               # FastAPI app + API key middleware
frontend/
└── src/app/              # Next.js 15 UI (App Router)
sandbox/
└── Dockerfile            # Alpine image for shell.exec isolation
scripts/
└── fine_tune.py          # LoRA fine-tuning script (optional)
install.bat / install.sh  # One-click installer (installs all prerequisites)
start.bat   / start.sh    # One-click launcher
stop.bat    / stop.sh     # One-click stopper
docker-compose.yml        # Full stack (Ollama included)
```

---

## Troubleshooting

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

**install.bat closes immediately after "administrator" prompt**  
Right-click `install.bat` → **Run as administrator**.

