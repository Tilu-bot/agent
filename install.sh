#!/usr/bin/env bash
# Agentic — One-Click Installer (Linux / macOS)
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

ok()   { echo -e " ${GREEN}[OK]${RESET}   $*"; }
warn() { echo -e " ${YELLOW}[WARN]${RESET}  $*"; }
fail() { echo -e " ${RED}[ERROR]${RESET} $*"; exit 1; }

echo -e "\n${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  Agentic  |  One-Click Installer${RESET}"
echo -e "${BOLD} =====================================================${RESET}\n"

# ── 1. Python ────────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    fail "Python 3 not found. Install from https://www.python.org/downloads/"
fi
ok "Python $(python3 --version 2>&1 | awk '{print $2}')"

# ── 2. Node.js ───────────────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    fail "Node.js not found. Install from https://nodejs.org/"
fi
ok "Node.js $(node --version)"

# ── 3. Docker ─────────────────────────────────────────────────────────────────
DOCKER_OK=0
if command -v docker &>/dev/null; then
    ok "Docker $(docker --version | awk '{print $3}' | tr -d ',')"
    DOCKER_OK=1
else
    warn "Docker not found — shell.exec tool will be disabled."
    warn "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
fi

# ── 4. Ollama ────────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    fail "Ollama not found. Install from https://ollama.com/download (free, runs locally)"
fi
ok "Ollama $(ollama --version 2>&1 | head -1)"

echo ""
echo -e "${BOLD} --- Setting up Python backend ---${RESET}\n"

# ── 5. Python venv ────────────────────────────────────────────────────────────
if [ ! -d "backend/.venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv backend/.venv
    ok "Virtual environment created."
else
    ok "Virtual environment already exists."
fi

# ── 6. Python deps ────────────────────────────────────────────────────────────
echo " Installing Python dependencies..."
backend/.venv/bin/pip install --upgrade pip --quiet
backend/.venv/bin/pip install -r backend/requirements.txt --quiet
ok "Python dependencies installed."

echo ""
echo -e "${BOLD} --- Setting up Node.js frontend ---${RESET}\n"

# ── 7. Node deps ──────────────────────────────────────────────────────────────
echo " Installing Node.js dependencies..."
(cd frontend && npm install --silent)
ok "Node.js dependencies installed."

echo ""
echo -e "${BOLD} --- Building Docker sandbox image ---${RESET}\n"

# ── 8. Sandbox Docker image ───────────────────────────────────────────────────
if [ "$DOCKER_OK" = "1" ]; then
    echo " Building agentic-sandbox:latest (first run only)..."
    docker build -t agentic-sandbox:latest ./sandbox -q
    ok "agentic-sandbox:latest built."
else
    warn "Skipping sandbox build (Docker not available)."
fi

echo ""
echo -e "${BOLD} --- Pulling Ollama models ---${RESET}\n"

# ── 9. Ensure Ollama is running ───────────────────────────────────────────────
if ! curl -s http://localhost:11434 &>/dev/null; then
    echo " Starting Ollama server..."
    ollama serve &>/dev/null &
    sleep 3
fi

echo " Pulling llama3.2:3b  (may take a few minutes on first run)..."
ollama pull llama3.2:3b
echo " Pulling qwen2.5-coder:3b..."
ollama pull qwen2.5-coder:3b
echo " Pulling nomic-embed-text..."
ollama pull nomic-embed-text
ok "Models ready."

echo ""
echo -e "${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  Installation complete!${RESET}"
echo ""
echo -e "  Next step:  ${BOLD}./start.sh${RESET}"
echo -e "${BOLD} =====================================================${RESET}\n"
