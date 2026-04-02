@echo off
setlocal enabledelayedexpansion
title Agentic - Setup (installing everything automatically...)

echo.
echo  =====================================================
echo   Agentic  ^|  Automatic Setup
echo   Everything will be downloaded and installed now.
echo  =====================================================
echo.

:: ── Require Administrator (needed to install software) ─────────────────────
net session >nul 2>&1
if errorlevel 1 (
    echo  [!] Administrator rights are needed to install software.
    echo.
    echo      A Windows security prompt (UAC) will appear asking you to allow
    echo      changes.  Click YES / Allow.
    echo.
    echo      A NEW installer window will then open and continue automatically.
    echo      This window will close -- please watch the new one.
    echo.
    pause
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: ── Helper: check winget ───────────────────────────────────────────────────
winget --version >nul 2>&1
set WINGET_OK=0
if not errorlevel 1 set WINGET_OK=1

:: ════════════════════════════════════════════════════════════════════════════
:: 1. PYTHON
:: ════════════════════════════════════════════════════════════════════════════
python --version >nul 2>&1
if errorlevel 1 (
    echo  [>>] Python not found - installing automatically...
    if "%WINGET_OK%"=="1" (
        winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
    ) else (
        echo      Downloading Python installer...
        powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python-installer.exe'"
        "%TEMP%\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1
        del "%TEMP%\python-installer.exe" >nul 2>&1
    )
    for /f "tokens=*" %%p in ('powershell -Command "[System.Environment]::GetEnvironmentVariable(\"PATH\",\"Machine\")"') do set PATH=%%p;%PATH%
    python --version >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] Python install failed. Please restart this script and try again.
        pause & exit /b 1
    )
    echo  [OK] Python installed.
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [OK] Python %%v already installed.
)

:: ════════════════════════════════════════════════════════════════════════════
:: 2. NODE.JS
:: ════════════════════════════════════════════════════════════════════════════
node --version >nul 2>&1
if errorlevel 1 (
    echo  [>>] Node.js not found - installing automatically...
    if "%WINGET_OK%"=="1" (
        winget install --id OpenJS.NodeJS.LTS -e --silent --accept-package-agreements --accept-source-agreements
    ) else (
        echo      Downloading Node.js installer...
        powershell -Command "Invoke-WebRequest -Uri 'https://nodejs.org/dist/v20.18.0/node-v20.18.0-x64.msi' -OutFile '%TEMP%\nodejs-installer.msi'"
        msiexec /i "%TEMP%\nodejs-installer.msi" /qn ADDLOCAL=ALL
        del "%TEMP%\nodejs-installer.msi" >nul 2>&1
    )
    for /f "tokens=*" %%p in ('powershell -Command "[System.Environment]::GetEnvironmentVariable(\"PATH\",\"Machine\")"') do set PATH=%%p;%PATH%
    node --version >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] Node.js install failed. Please restart this script and try again.
        pause & exit /b 1
    )
    echo  [OK] Node.js installed.
) else (
    for /f %%v in ('node --version 2^>^&1') do echo  [OK] Node.js %%v already installed.
)

:: ════════════════════════════════════════════════════════════════════════════
:: 3. OLLAMA
:: ════════════════════════════════════════════════════════════════════════════
ollama --version >nul 2>&1
if errorlevel 1 (
    echo  [>>] Ollama not found - installing automatically...
    if "%WINGET_OK%"=="1" (
        winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
    ) else (
        echo      Downloading Ollama installer...
        powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile '%TEMP%\OllamaSetup.exe'"
        "%TEMP%\OllamaSetup.exe" /S
        del "%TEMP%\OllamaSetup.exe" >nul 2>&1
    )
    for /f "tokens=*" %%p in ('powershell -Command "[System.Environment]::GetEnvironmentVariable(\"PATH\",\"Machine\")"') do set PATH=%%p;%PATH%
    ollama --version >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] Ollama install failed. Please restart this script and try again.
        pause & exit /b 1
    )
    echo  [OK] Ollama installed.
) else (
    for /f "tokens=1-2" %%a in ('ollama --version 2^>^&1') do echo  [OK] Ollama %%a %%b already installed.
)

:: ════════════════════════════════════════════════════════════════════════════
:: 4. DOCKER DESKTOP  (optional - only needed for shell.exec tool)
:: ════════════════════════════════════════════════════════════════════════════
docker --version >nul 2>&1
set DOCKER_OK=0
if errorlevel 1 (
    echo  [>>] Docker Desktop not found - installing automatically...
    echo      (Enables the secure code-execution sandbox.)
    if "%WINGET_OK%"=="1" (
        winget install --id Docker.DockerDesktop -e --silent --accept-package-agreements --accept-source-agreements
    ) else (
        echo      Downloading Docker Desktop installer...
        powershell -Command "Invoke-WebRequest -Uri 'https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe' -OutFile '%TEMP%\DockerInstaller.exe'"
        "%TEMP%\DockerInstaller.exe" install --quiet --accept-license
        del "%TEMP%\DockerInstaller.exe" >nul 2>&1
    )
    docker --version >nul 2>&1
    if not errorlevel 1 set DOCKER_OK=1
    if "%DOCKER_OK%"=="0" echo  [WARN] Docker needs a reboot to finish - sandbox will be enabled after restart.
) else (
    set DOCKER_OK=1
    for /f "tokens=1-3" %%a in ('docker --version 2^>^&1') do echo  [OK] Docker %%a %%b %%c already installed.
)

echo.
echo  ---  Setting up Python backend  ---
echo.

:: ════════════════════════════════════════════════════════════════════════════
:: 5. PYTHON VIRTUAL ENVIRONMENT + DEPENDENCIES
:: ════════════════════════════════════════════════════════════════════════════
if not exist "backend\.venv" (
    echo  Creating Python environment...
    python -m venv backend\.venv
    if errorlevel 1 ( echo  [ERROR] Could not create Python environment. & pause & exit /b 1 )
) else (
    echo  [OK] Python environment already exists.
)
echo  Installing Python packages (may take a few minutes on first run)...
echo  (You will see package names scroll by - this is normal.)
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt
if errorlevel 1 ( echo  [ERROR] Package install failed. & pause & exit /b 1 )
echo  [OK] Python packages installed.

echo.
echo  ---  Setting up web frontend  ---
echo.

:: ════════════════════════════════════════════════════════════════════════════
:: 6. NODE.JS FRONTEND DEPENDENCIES
:: ════════════════════════════════════════════════════════════════════════════
echo  Installing web packages (may take a few minutes on first run)...
echo  (You will see package names scroll by - this is normal.)
cd frontend
call npm install
if errorlevel 1 ( cd .. & echo  [ERROR] npm install failed. & pause & exit /b 1 )
cd ..
echo  [OK] Web packages installed.

echo.
echo  ---  Building Docker sandbox  ---
echo.

:: ════════════════════════════════════════════════════════════════════════════
:: 7. BUILD DOCKER SANDBOX
:: ════════════════════════════════════════════════════════════════════════════
if "%DOCKER_OK%"=="1" (
    docker info >nul 2>&1
    if errorlevel 1 (
        echo  [WARN] Docker installed but not running yet.
        echo         Start Docker Desktop once, then re-run this installer to finish.
    ) else (
        echo  Building sandbox image...
        docker build -t agentic-sandbox:latest ./sandbox -q
        if not errorlevel 1 echo  [OK] Sandbox image built.
    )
) else (
    echo  [SKIP] Docker not available - sandbox skipped.
)

echo.
echo  ---  Downloading AI models (one-time, ~4 GB total)  ---
echo.

:: ════════════════════════════════════════════════════════════════════════════
:: 8. START OLLAMA + PULL MODELS
:: ════════════════════════════════════════════════════════════════════════════
ollama list >nul 2>&1
if errorlevel 1 (
    echo  Starting Ollama service...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)
echo  Downloading llama3.2:3b  (main model, ~2 GB) ...
ollama pull llama3.2:3b
echo  Downloading qwen2.5-coder:3b  (code model, ~2 GB) ...
ollama pull qwen2.5-coder:3b
echo  Downloading nomic-embed-text  (tiny embedding model) ...
ollama pull nomic-embed-text
echo  [OK] All AI models downloaded.

echo.
echo  ---  Creating desktop shortcut  ---
echo.

:: ════════════════════════════════════════════════════════════════════════════
:: 9. DESKTOP SHORTCUT  (so you never need to find this folder again)
:: ════════════════════════════════════════════════════════════════════════════
set "SHORTCUT=%USERPROFILE%\Desktop\Agentic AI.lnk"
set "TARGET=%~dp0start.bat"
set "WORKDIR=%~dp0"
powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%WORKDIR%'; $s.Description = 'Start Agentic AI'; $s.Save()"
if exist "%SHORTCUT%" (
    echo  [OK] Desktop shortcut created: "Agentic AI"
) else (
    echo  [WARN] Could not create desktop shortcut ^(non-fatal^).
)

echo.
echo  =====================================================
echo.
echo   SETUP COMPLETE!
echo.
echo   To launch Agentic:
echo     * Double-click "Agentic AI" on your Desktop
echo       (or double-click start.bat in this folder)
echo.
echo   The app opens in your browser automatically.
echo  =====================================================
echo.
pause
endlocal
