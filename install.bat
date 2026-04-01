@echo off
setlocal enabledelayedexpansion
title Agentic — Installer

echo.
echo  =====================================================
echo   Agentic  ^|  One-Click Installer
echo  =====================================================
echo.

:: ── 1. Check Python ────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download from https://www.python.org/downloads/
    echo         Make sure to check "Add to PATH" during install.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER%

:: ── 2. Check Node.js ───────────────────────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found.
    echo         Download from https://nodejs.org/
    pause & exit /b 1
)
for /f %%v in ('node --version 2^>^&1') do set NODE_VER=%%v
echo [OK] Node.js %NODE_VER%

:: ── 3. Check Docker ────────────────────────────────────────────────────────
docker --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Docker not found — shell.exec tool will be disabled.
    echo        Install Docker Desktop from https://www.docker.com/products/docker-desktop/
    set DOCKER_OK=0
) else (
    for /f "tokens=1-3" %%a in ('docker --version 2^>^&1') do set DOCKER_VER=%%a %%b %%c
    echo [OK] %DOCKER_VER%
    set DOCKER_OK=1
)

:: ── 4. Check Ollama ────────────────────────────────────────────────────────
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not found.
    echo         Download from https://ollama.com/download  (free, runs locally)
    echo         After install, re-run this script.
    pause & exit /b 1
)
for /f "tokens=1-2" %%a in ('ollama --version 2^>^&1') do set OLLAMA_VER=%%a %%b
echo [OK] Ollama %OLLAMA_VER%

echo.
echo  --- Setting up Python backend ---
echo.

:: ── 5. Create Python venv ──────────────────────────────────────────────────
if not exist "backend\.venv" (
    echo Creating virtual environment...
    python -m venv backend\.venv
    if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: ── 6. Install Python deps ─────────────────────────────────────────────────
echo Installing Python dependencies...
backend\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt --quiet
if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
echo [OK] Python dependencies installed.

echo.
echo  --- Setting up Node.js frontend ---
echo.

:: ── 7. Install Node deps ───────────────────────────────────────────────────
echo Installing Node.js dependencies...
cd frontend
call npm install --silent
if errorlevel 1 ( cd .. & echo [ERROR] npm install failed. & pause & exit /b 1 )
cd ..
echo [OK] Node.js dependencies installed.

echo.
echo  --- Building Docker sandbox image ---
echo.

:: ── 8. Build sandbox ──────────────────────────────────────────────────────
if "%DOCKER_OK%"=="1" (
    echo Building agentic-sandbox Docker image (first run only)...
    docker build -t agentic-sandbox:latest ./sandbox -q
    if errorlevel 1 (
        echo [WARN] Sandbox build failed — shell.exec will not work.
    ) else (
        echo [OK] agentic-sandbox:latest built.
    )
) else (
    echo [SKIP] Docker not available — skipping sandbox build.
)

echo.
echo  --- Pulling Ollama models ---
echo.

:: ── 9. Pull models (start Ollama serve in background if needed) ────────────
ollama list >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama server...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)

echo Pulling llama3.2:3b  (this may take a few minutes on first run)...
ollama pull llama3.2:3b
echo Pulling qwen2.5-coder:3b...
ollama pull qwen2.5-coder:3b
echo Pulling nomic-embed-text...
ollama pull nomic-embed-text
echo [OK] Models ready.

echo.
echo  =====================================================
echo   Installation complete!
echo.
echo   Next step:  double-click  start.bat
echo  =====================================================
echo.
pause
endlocal
