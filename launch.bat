@echo off
title OpenClaw — System Launcher
cd /d "%~dp0"
setlocal enabledelayedexpansion

:: ─────────────────────────────────────────────────────────────────────────────
::  CONFIGURATION  (edit here or in Mini Openclaw\.env)
:: ─────────────────────────────────────────────────────────────────────────────
set FLASK_URL=http://127.0.0.1:8765
set OLLAMA_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=qwen3.5:4b
set BOT_DIR=%~dp0Mini Openclaw
set SERVER_LOG=%~dp0logs\server.log
set BOT_LOG=%~dp0logs\bot.log
set MAX_WAIT=120
set POLL_INTERVAL=2

:: ─────────────────────────────────────────────────────────────────────────────
::  SETUP
:: ─────────────────────────────────────────────────────────────────────────────
if not exist "%~dp0logs" mkdir "%~dp0logs"

cls
echo.
echo   ___  ____  ____  _  _  ___  __     __   _    _
echo  / _ \(  _ \( ___)( \( )/ __)(  )   /__\ ( \/\/ )
echo ( (_) ))|) / )__)  )  (( (__  )(__  /(__)\  \  /
echo  \___/(____/(____)(_)\_)\___)(____)(__)(__) \/\/
echo.
echo  SO-100 Robotic Arm  ^|  MuJoCo Simulation  ^|  Telegram Bot
echo  ════════════════════════════════════════════════════════════
echo.

:: ─────────────────────────────────────────────────────────────────────────────
::  CLEAN UP any previous instances
:: ─────────────────────────────────────────────────────────────────────────────
echo  [*] Cleaning up previous processes (if any)...
wmic process where "Name='python.exe' and CommandLine like '%%server.py%%'" delete >nul 2>&1
wmic process where "Name='python.exe' and CommandLine like '%%bot.py%%'" delete >nul 2>&1
timeout /t 1 /nobreak >nul

:: Clear old log files
echo. > "%SERVER_LOG%"
echo. > "%BOT_LOG%"

:: ─────────────────────────────────────────────────────────────────────────────
::  LAUNCH  (background, no visible window, output -> log files)
:: ─────────────────────────────────────────────────────────────────────────────
echo  [1/2] Starting MuJoCo Flask server in background...
start "" /B cmd /c "cd /d "%~dp0" && python server.py >> "%SERVER_LOG%" 2>&1"

echo  [2/2] Starting Telegram bot in background...
start "" /B cmd /c "cd /d "%BOT_DIR%" && python bot.py >> "%BOT_LOG%" 2>&1"

echo.
echo  ════════════════════════════════════════════════════════════
echo  Polling services — please wait (up to %MAX_WAIT%s)...
echo  Logs: logs\server.log  ^|  logs\bot.log
echo  ════════════════════════════════════════════════════════════
echo.

:: ─────────────────────────────────────────────────────────────────────────────
::  POLL LOOP
:: ─────────────────────────────────────────────────────────────────────────────
set OLLAMA_OK=0
set FLASK_OK=0
set BOT_OK=0
set ELAPSED=0

:POLL_LOOP

:: ── 1. Ollama: check tags endpoint lists our model ──────────────────────────
if !OLLAMA_OK!==0 (
    curl -s --max-time 2 "%OLLAMA_URL%/api/tags" 2>nul | find /i "%OLLAMA_MODEL%" >nul 2>&1
    if not errorlevel 1 set OLLAMA_OK=1
)

:: ── 2. Flask: check /state returns valid JSON ────────────────────────────────
if !FLASK_OK!==0 (
    curl -s --max-time 3 "%FLASK_URL%/state" 2>nul | find /i "eef" >nul 2>&1
    if not errorlevel 1 set FLASK_OK=1
)

:: ── 3. Bot: look for "Starting long-polling" in bot.log ─────────────────────
if !BOT_OK!==0 (
    if exist "%BOT_LOG%" (
        findstr /i "Starting long-polling" "%BOT_LOG%" >nul 2>&1
        if not errorlevel 1 set BOT_OK=1
    )
)

:: ── Status icons ─────────────────────────────────────────────────────────────
if !OLLAMA_OK!==1 (set OL=[waiting]) else (set OL=[  OK   ])
if !FLASK_OK!==1  (set FL=[waiting]) else (set FL=[  OK   ])
if !BOT_OK!==1    (set BO=[waiting]) else (set BO=[  OK   ])

echo   !OL!  Ollama  (%OLLAMA_MODEL%)
echo   !FL!  Flask simulation server  (%FLASK_URL%)
echo   !BO!  Telegram bot
echo   -----------------  elapsed: !ELAPSED!s
echo.

:: ── All up? ──────────────────────────────────────────────────────────────────
if !OLLAMA_OK!==1 if !FLASK_OK!==1 if !BOT_OK!==1 goto :ALL_READY

:: ── Timeout guard ────────────────────────────────────────────────────────────
if !ELAPSED! GEQ %MAX_WAIT% goto :TIMEOUT

timeout /t %POLL_INTERVAL% /nobreak >nul
set /a ELAPSED+=%POLL_INTERVAL%
goto :POLL_LOOP


:: ─────────────────────────────────────────────────────────────────────────────
::  ALL READY
:: ─────────────────────────────────────────────────────────────────────────────
:ALL_READY
cls
echo.
echo  ============================================================
echo.
echo    ALL SYSTEMS READY  (took !ELAPSED!s)
echo.
echo    OK   Ollama %OLLAMA_MODEL% is loaded
echo    OK   MuJoCo Flask server is responding  (%FLASK_URL%)
echo    OK   Telegram bot is polling Telegram
echo.
echo  ============================================================
echo.
echo  Logs:   logs\server.log   ^|   logs\bot.log
echo  Press any key to shut down all background services.
echo.
goto :IDLE


:: ─────────────────────────────────────────────────────────────────────────────
::  TIMEOUT
:: ─────────────────────────────────────────────────────────────────────────────
:TIMEOUT
echo.
echo  ============================================================
echo   WARNING — not all services came online within %MAX_WAIT% seconds.
echo  ============================================================
echo.
if !OLLAMA_OK!==0 echo    FAIL   Ollama — is "ollama serve" running?
if !FLASK_OK!==0  echo    FAIL   Flask  — check logs\server.log for errors
if !BOT_OK!==0    echo    FAIL   Bot    — check logs\bot.log for errors
echo.
echo  Services that did start are still running in the background.
echo  Press any key to exit this launcher (services keep running).
echo.
goto :IDLE


:: ─────────────────────────────────────────────────────────────────────────────
::  IDLE — keep window open so user can read the status
:: ─────────────────────────────────────────────────────────────────────────────
:IDLE
pause >nul
