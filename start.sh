#!/usr/bin/env bash
# Agentic — One-Click Launcher (Linux / macOS)
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

ok()   { echo -e " ${GREEN}[OK]${RESET}   $*"; }
warn() { echo -e " ${YELLOW}[WARN]${RESET}  $*"; }
fail() { echo -e " ${RED}[ERROR]${RESET} $*"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$REPO_ROOT/.agentic/pids"

echo -e "\n${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  Agentic  |  Starting...${RESET}"
echo -e "${BOLD} =====================================================${RESET}\n"

# ── Guard ─────────────────────────────────────────────────────────────────────
[ -d "$REPO_ROOT/backend/.venv" ] || fail "Backend not set up. Run ./install.sh first."
[ -d "$REPO_ROOT/frontend/node_modules" ] || fail "Frontend not set up. Run ./install.sh first."

mkdir -p "$REPO_ROOT/.agentic"

# ── 1. Ollama ─────────────────────────────────────────────────────────────────
if ! curl -s http://localhost:11434 &>/dev/null; then
    echo " Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
    ok "Ollama started."
else
    ok "Ollama already running."
fi

# ── 2. Backend ────────────────────────────────────────────────────────────────
echo " Starting backend on http://localhost:8000 ..."
cd "$REPO_ROOT"
PYTHONPATH=. backend/.venv/bin/python -m uvicorn backend.main:app \
    --host 0.0.0.0 --port 8000 \
    > .agentic/backend.log 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$PIDFILE.backend"

# ── 3. Wait for backend ───────────────────────────────────────────────────────
echo -n " Waiting for backend"
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/health &>/dev/null; then
        echo ""
        ok "Backend ready (PID $BACKEND_PID)."
        break
    fi
    echo -n "."
    sleep 1
    if [ "$i" -eq 30 ]; then
        echo ""
        fail "Backend did not start. Check .agentic/backend.log"
    fi
done

# ── 4. Frontend ───────────────────────────────────────────────────────────────
echo " Starting frontend on http://localhost:3000 ..."
cd "$REPO_ROOT/frontend"
npm run dev > "$REPO_ROOT/.agentic/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$PIDFILE.frontend"
cd "$REPO_ROOT"

# ── 5. Wait for frontend ──────────────────────────────────────────────────────
echo -n " Waiting for frontend"
for i in $(seq 1 30); do
    if curl -s http://localhost:3000 &>/dev/null; then
        echo ""
        ok "Frontend ready (PID $FRONTEND_PID)."
        break
    fi
    echo -n "."
    sleep 1
    if [ "$i" -eq 30 ]; then
        echo ""
        warn "Frontend taking longer than expected. Opening browser anyway — refresh if blank."
        break
    fi
done

# ── 6. Open browser ───────────────────────────────────────────────────────────
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:3000 &>/dev/null &
elif command -v open &>/dev/null; then
    open http://localhost:3000
fi

echo ""
echo -e "${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  Agentic is running!${RESET}"
echo ""
echo -e "  Browser:  ${BOLD}http://localhost:3000${RESET}"
echo -e "  API docs: ${BOLD}http://localhost:8000/docs${RESET}"
echo ""
echo -e "  Logs:     .agentic/backend.log"
echo -e "            .agentic/frontend.log"
echo ""
echo -e "  To stop:  ${BOLD}./stop.sh${RESET}"
echo -e "${BOLD} =====================================================${RESET}\n"

# Keep script alive so Ctrl+C stops everything
echo " Press Ctrl+C to stop Agentic."
trap 'echo ""; echo " Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo " Done."; exit 0' INT TERM
wait
