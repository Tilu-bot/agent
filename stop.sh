#!/usr/bin/env bash
# Agentic — Stop (Linux / macOS)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$REPO_ROOT/.agentic/pids"

echo ""
echo " Stopping Agentic..."

stopped=0

for svc in backend frontend; do
    pf="$PIDFILE.$svc"
    if [ -f "$pf" ]; then
        pid=$(cat "$pf")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && echo " [OK]   $svc stopped (PID $pid)."
        fi
        rm -f "$pf"
        stopped=$((stopped + 1))
    fi
done

# Fallback: also kill by port
for port in 8000 3000; do
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill 2>/dev/null || true
    fi
done

if [ "$stopped" -eq 0 ]; then
    echo " Nothing was running (no PID files found)."
fi

echo " (Ollama keeps running — stop it separately if needed.)"
echo ""
