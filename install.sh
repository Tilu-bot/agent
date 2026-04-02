#!/usr/bin/env bash
# Agentic — Automatic Setup (Linux / macOS)
# Downloads and installs everything: Ollama, Python, Node.js, models, and the app itself.
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"

ok()      { echo -e " ${GREEN}[OK]${RESET}    $*"; }
info()    { echo -e " ${CYAN}[>>]${RESET}    $*"; }
warn()    { echo -e " ${YELLOW}[WARN]${RESET}   $*"; }
fail()    { echo -e " ${RED}[ERROR]${RESET}  $*"; exit 1; }
section() { echo -e "\n${BOLD} ---  $*  ---${RESET}\n"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"

echo -e "\n${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  Agentic  |  Automatic Setup${RESET}"
echo -e "${BOLD}  Everything will be downloaded and installed now.${RESET}"
echo -e "${BOLD} =====================================================${RESET}\n"

# ════════════════════════════════════════════════════════════════════════════
# 1. PYTHON 3
# ════════════════════════════════════════════════════════════════════════════
section "Python 3"
if command -v python3 &>/dev/null; then
    ok "Python $(python3 --version 2>&1 | awk '{print $2}') already installed."
else
    info "Python 3 not found — installing automatically..."
    case "$OS" in
        Darwin)
            if ! command -v brew &>/dev/null; then
                info "Installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python@3.11
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq && sudo apt-get install -y python3 python3-pip python3-venv
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y python3 python3-pip
            elif command -v pacman &>/dev/null; then
                sudo pacman -Sy --noconfirm python python-pip
            else
                fail "Cannot auto-install Python on this system. Install python3 manually then re-run."
            fi
            ;;
    esac
    command -v python3 &>/dev/null || fail "Python install failed. Install python3 manually then re-run."
    ok "Python $(python3 --version 2>&1 | awk '{print $2}') installed."
fi

# ════════════════════════════════════════════════════════════════════════════
# 2. NODE.JS
# ════════════════════════════════════════════════════════════════════════════
section "Node.js"
if command -v node &>/dev/null; then
    ok "Node.js $(node --version) already installed."
else
    info "Node.js not found — installing automatically..."
    case "$OS" in
        Darwin)
            brew install node@20
            brew link --overwrite node@20 || true
            ;;
        Linux)
            # NodeSource official installer (works on Debian/Ubuntu/RHEL/Fedora)
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - 2>/dev/null || \
            curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash - 2>/dev/null || true
            if command -v apt-get &>/dev/null; then
                sudo apt-get install -y nodejs
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y nodejs
            fi
            ;;
    esac
    command -v node &>/dev/null || fail "Node.js install failed. Install node 20+ manually then re-run."
    ok "Node.js $(node --version) installed."
fi

# ════════════════════════════════════════════════════════════════════════════
# 3. OLLAMA
# ════════════════════════════════════════════════════════════════════════════
section "Ollama (local AI engine)"
if command -v ollama &>/dev/null; then
    ok "Ollama $(ollama --version 2>&1 | head -1) already installed."
else
    info "Ollama not found — installing automatically..."
    # Official Ollama installer — works on macOS and Linux
    curl -fsSL https://ollama.com/install.sh | sh
    command -v ollama &>/dev/null || fail "Ollama install failed. Visit https://ollama.com/download"
    ok "Ollama installed."
fi

# ════════════════════════════════════════════════════════════════════════════
# 4. DOCKER  (optional)
# ════════════════════════════════════════════════════════════════════════════
section "Docker (optional - for code sandbox)"
DOCKER_OK=0
if command -v docker &>/dev/null; then
    ok "Docker $(docker --version | awk '{print $3}' | tr -d ',') already installed."
    DOCKER_OK=1
else
    info "Docker not found — installing automatically..."
    case "$OS" in
        Darwin)
            warn "On macOS, Docker Desktop must be installed manually."
            warn "Download from https://www.docker.com/products/docker-desktop/"
            warn "Sandbox (shell.exec) tool will be disabled until Docker is installed."
            ;;
        Linux)
            if command -v apt-get &>/dev/null; then
                curl -fsSL https://get.docker.com | sudo sh
                sudo usermod -aG docker "$USER" || true
                DOCKER_OK=1
                ok "Docker installed. You may need to log out and back in for group permissions."
            else
                warn "Cannot auto-install Docker on this distro. Install manually if needed."
            fi
            ;;
    esac
fi

# ════════════════════════════════════════════════════════════════════════════
# 5. PYTHON VIRTUAL ENVIRONMENT + DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
section "Python backend packages"
if [ ! -d "$REPO/backend/.venv" ]; then
    info "Creating Python environment..."
    python3 -m venv "$REPO/backend/.venv"
fi
ok "Python environment ready."
info "Installing Python packages (may take ~1 min on first run)..."
"$REPO/backend/.venv/bin/pip" install --upgrade pip --quiet
"$REPO/backend/.venv/bin/pip" install -r "$REPO/backend/requirements.txt" --quiet
ok "Python packages installed."

# ════════════════════════════════════════════════════════════════════════════
# 6. NODE.JS FRONTEND DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
section "Web frontend packages"
info "Installing web packages (may take ~1 min on first run)..."
(cd "$REPO/frontend" && npm install --silent)
ok "Web packages installed."

# ════════════════════════════════════════════════════════════════════════════
# 7. DOCKER SANDBOX IMAGE
# ════════════════════════════════════════════════════════════════════════════
section "Docker sandbox image"
if [ "$DOCKER_OK" = "1" ] && docker info &>/dev/null 2>&1; then
    info "Building sandbox image..."
    docker build -t agentic-sandbox:latest "$REPO/sandbox" -q
    ok "Sandbox image built."
else
    warn "Docker not available — sandbox skipped (non-fatal)."
fi

# ════════════════════════════════════════════════════════════════════════════
# 8. OLLAMA MODELS  (~4 GB total, one-time download)
# ════════════════════════════════════════════════════════════════════════════
section "AI models (one-time download, ~4 GB total)"
if ! curl -s http://localhost:11434 &>/dev/null; then
    info "Starting Ollama service..."
    ollama serve &>/dev/null &
    sleep 4
fi
info "Downloading llama3.2:3b  (main model, ~2 GB) ..."
ollama pull llama3.2:3b
info "Downloading qwen2.5-coder:3b  (code model, ~2 GB) ..."
ollama pull qwen2.5-coder:3b
info "Downloading nomic-embed-text  (tiny embedding model) ..."
ollama pull nomic-embed-text
ok "All AI models downloaded."

# ════════════════════════════════════════════════════════════════════════════
# 9. MAKE SCRIPTS EXECUTABLE
# ════════════════════════════════════════════════════════════════════════════
chmod +x "$REPO/start.sh" "$REPO/stop.sh" "$REPO/install.sh"

# ════════════════════════════════════════════════════════════════════════════
# 10. DESKTOP SHORTCUT / LAUNCHER
# ════════════════════════════════════════════════════════════════════════════
section "Desktop shortcut"
case "$OS" in
    Darwin)
        # macOS: create a clickable .command file on the Desktop
        SHORTCUT="$HOME/Desktop/Agentic AI.command"
        printf '#!/bin/bash\ncd "%s"\n./start.sh\n' "$REPO" > "$SHORTCUT"
        chmod +x "$SHORTCUT"
        ok "Desktop shortcut created: ~/Desktop/Agentic AI.command"
        ;;
    Linux)
        # Linux: create a .desktop file
        SHORTCUT="$HOME/Desktop/agentic-ai.desktop"
        mkdir -p "$HOME/Desktop"
        cat > "$SHORTCUT" << EOF
[Desktop Entry]
Name=Agentic AI
Comment=Start Agentic AI
Exec=bash -c 'cd "$REPO" && ./start.sh'
Icon=utilities-terminal
Terminal=true
Type=Application
Categories=Utility;
EOF
        chmod +x "$SHORTCUT"
        ok "Desktop shortcut created: ~/Desktop/agentic-ai.desktop"
        ;;
esac

echo ""
echo -e "${BOLD} =====================================================${RESET}"
echo -e "${BOLD}  SETUP COMPLETE!${RESET}"
echo ""
echo -e "  To launch Agentic:"
echo -e "    ${BOLD}• Double-click \"Agentic AI\" on your Desktop${RESET}"
echo -e "    ${BOLD}• Or run:  ./start.sh${RESET}"
echo ""
echo -e "  The app opens in your browser automatically."
echo -e "${BOLD} =====================================================${RESET}"
echo ""
