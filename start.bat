@echo off
setlocal enabledelayedexpansion
title Agentic — Launcher

echo.
echo  =====================================================
echo   Agentic  ^|  Starting...
echo  =====================================================
echo.

:: ── Guard: check install was run ───────────────────────────────────────────
if not exist "backend\.venv" (
    echo [ERROR] Backend not set up. Please run install.bat first.
    pause & exit /b 1
)
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend not set up. Please run install.bat first.
    pause & exit /b 1
)

:: ── 1. Start Ollama (if not already running) ───────────────────────────────
curl -s http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama...
    start /B ollama serve
    timeout /t 4 /nobreak >nul
    echo [OK] Ollama started.
) else (
    echo [OK] Ollama already running.
)

:: ── 2. Start backend ───────────────────────────────────────────────────────
echo Starting backend on http://localhost:8000 ...
start "Agentic Backend" /MIN cmd /c "backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 2>&1 | tee .agentic\backend.log"

:: Give backend a moment to initialise
timeout /t 3 /nobreak >nul

:: ── 3. Wait for backend to be ready (up to 30 s) ──────────────────────────
set /a TRIES=0
:WAIT_BACKEND
set /a TRIES+=1
if %TRIES% gtr 30 (
    echo [ERROR] Backend did not start within 30 seconds.
    echo         Check .agentic\backend.log for details.
    pause & exit /b 1
)
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto WAIT_BACKEND
)
echo [OK] Backend ready.

:: ── 4. Start frontend ──────────────────────────────────────────────────────
echo Starting frontend on http://localhost:3000 ...
start "Agentic Frontend" /MIN cmd /c "cd frontend && npm run dev 2>&1 | tee ..\.agentic\frontend.log"

:: ── 5. Wait for frontend (up to 30 s) ─────────────────────────────────────
set /a TRIES=0
:WAIT_FRONTEND
set /a TRIES+=1
if %TRIES% gtr 30 (
    echo [WARN] Frontend is taking longer than expected to start.
    echo        Opening browser anyway — refresh in a few seconds if needed.
    goto OPEN_BROWSER
)
curl -s http://localhost:3000 >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto WAIT_FRONTEND
)
echo [OK] Frontend ready.

:OPEN_BROWSER
echo.
echo  =====================================================
echo   Agentic is running!
echo.
echo   Browser:  http://localhost:3000
echo   API docs: http://localhost:8000/docs
echo.
echo   To stop everything:  run stop.bat
echo  =====================================================
echo.
start http://localhost:3000

pause
endlocal
