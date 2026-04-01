@echo off
title Agentic — Stop

echo.
echo  Stopping Agentic...
echo.

:: Kill processes listening on port 8000 (backend)
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":8000 "') do (
    if not "%%p"=="0" (
        taskkill /PID %%p /F >nul 2>&1
    )
)

:: Kill processes listening on port 3000 (frontend)
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":3000 "') do (
    if not "%%p"=="0" (
        taskkill /PID %%p /F >nul 2>&1
    )
)

echo [OK] Backend and frontend stopped.
echo      (Ollama keeps running — stop it from the system tray if needed.)
echo.
pause
