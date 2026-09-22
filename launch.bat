@echo off
title XavierClaw - System Launcher
cd /d "%~dp0"
setlocal enabledelayedexpansion

rem ============================================================
rem  CONFIGURATION
rem ============================================================
set FLASK_URL=http://127.0.0.1:8765
set OLLAMA_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=qwen3.5:4b
set BOT_DIR=%~dp0Mini Openclaw
set SERVER_LOG=%~dp0logs\server.log
set BOT_LOG=%~dp0logs\bot.log
set MAX_WAIT=180
set POLL_INTERVAL=2

rem ============================================================
rem  SETUP
rem ============================================================
if not exist "%~dp0logs" mkdir "%~dp0logs"

cls
echo.
echo  ##  ##  ##   ##  ##  ####  ####  ####   ####  ##     ##   ##
echo  ##  ##  ###  ##  ##  ##    ##    ##  ##  ##    ##     ##   ##
echo  ##  ##  ## # ##  ##  ###   ###   ####    ###   ##     ## # ##
echo   ####   ##  ###  ##  ##    ##    ## ##   ##    ##     ## # ##
echo    ##    ##   ##  ##  ####  ####  ##  ##  ####  #####   ## ##
echo.
echo  ##  ##   ##   ##  ##  ####  ####   ##  ##         ##   ##
echo   ####   ###  ##   ##  ##    ##  ##  ####   ##     ##   ##
echo    ##    ## ###    ##  ###   ####     ##    ##     ##   ##
echo    ##    ##   ##   ##  ##    ## ##    ##    ##     ##   ##
echo    ##    ##    ##  ##  ####  ##  ##   ##    #####   #####
echo.
echo  SO-100 Robotic Arm  ^|  MuJoCo Simulation  ^|  Telegram Bot
echo  ============================================================
echo.

rem ============================================================
rem  STEP 1 - Stop any previous instances
rem ============================================================
echo  [*] Stopping previous processes...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'server.py' -or $_.CommandLine -match 'bot.py' } | Invoke-CimMethod -MethodName Terminate" >nul 2>&1
ping 127.0.0.1 -n 2 >nul

rem Clear old log files
copy /y nul "%SERVER_LOG%" >nul
copy /y nul "%BOT_LOG%"   >nul

rem ============================================================
rem  STEP 2 - Start Ollama & wait until model is awake in VRAM
rem ============================================================
echo  [1/3] Checking Ollama server...
curl -s --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1
if errorlevel 1 (
    echo  [Ollama] Not running - starting Ollama in background...
    start "" /B ollama serve >nul 2>&1
    echo  [Ollama] Waiting for server process to start...
    ping 127.0.0.1 -n 5 >nul
) else (
    echo  [Ollama] Server process is running.
)

rem Write the Ollama inference ping body to a temp file
echo {"model":"%OLLAMA_MODEL%","messages":[{"role":"user","content":"hi"}],"stream":false} > "%TEMP%\ollama_ping.json"

echo  [Ollama] Waking up model %OLLAMA_MODEL% (loading into VRAM, please wait)...
set OLLAMA_WAIT=0

:WAIT_OLLAMA
curl -s --max-time 90 -X POST "%OLLAMA_URL%/api/chat" -H "Content-Type: application/json" --data-binary "@%TEMP%\ollama_ping.json" 2>nul | find "done" >nul 2>&1
if not errorlevel 1 (
    echo  [Ollama] Model %OLLAMA_MODEL% is awake and verified!
    goto :START_BOT
)
set /a OLLAMA_WAIT+=3
if !OLLAMA_WAIT! GEQ %MAX_WAIT% (
    echo   FAIL  Ollama timed out loading model %OLLAMA_MODEL%
    goto :TIMEOUT
)
ping 127.0.0.1 -n 3 >nul
goto :WAIT_OLLAMA


rem ============================================================
rem  STEP 3 - Start Telegram bot & wait until connected
rem ============================================================
:START_BOT
echo.
echo  [2/3] Starting Telegram bot in background...
start "" /B python "%BOT_DIR%\bot.py" 1>>"%BOT_LOG%" 2>&1

set BOT_WAIT=0
:WAIT_BOT
if exist "%BOT_LOG%" (
    findstr /i "Starting long-polling" "%BOT_LOG%" >nul 2>&1
    if not errorlevel 1 (
        echo  [Telegram] Bot is connected and long-polling!
        goto :START_SERVER
    )
)
ping 127.0.0.1 -n 2 >nul
set /a BOT_WAIT+=2
if !BOT_WAIT! GEQ 30 (
    echo   FAIL  Telegram bot failed to connect - check logs\bot.log
    goto :TIMEOUT
)
goto :WAIT_BOT


rem ============================================================
rem  STEP 4 - Launch MuJoCo Flask server LAST
rem ============================================================
:START_SERVER
echo.
echo  [3/3] Starting MuJoCo Flask server in background...
start "" /B python "%~dp0server.py" 1>>"%SERVER_LOG%" 2>&1

set FLASK_WAIT=0
:WAIT_FLASK
curl -s --max-time 3 "%FLASK_URL%/state" 2>nul | find "eef" >nul 2>&1
if not errorlevel 1 (
    echo  [Flask] MuJoCo simulation and REST server ready!
    goto :ALL_READY
)
ping 127.0.0.1 -n 2 >nul
set /a FLASK_WAIT+=2
if !FLASK_WAIT! GEQ 30 (
    echo   FAIL  Flask server failed to start - check logs\server.log
    goto :TIMEOUT
)
goto :WAIT_FLASK


rem ============================================================
rem  ALL READY
rem ============================================================
:ALL_READY
set /a TOTAL_TIME=OLLAMA_WAIT+BOT_WAIT+FLASK_WAIT
cls
echo.
echo  ============================================================
echo.
echo    ALL SYSTEMS READY  ^(took !TOTAL_TIME!s^)
echo.
echo    READY  Ollama %OLLAMA_MODEL%  ^(inference verified^)
echo    READY  Telegram bot is live
echo    READY  MuJoCo Flask server  at %FLASK_URL%
echo.
echo  ============================================================
echo.
echo  Logs:  logs\server.log   logs\bot.log
echo.
echo  Press any key to exit this window.
echo  ^(background services will keep running^)
echo.
goto :IDLE


rem ============================================================
rem  TIMEOUT
rem ============================================================
:TIMEOUT
echo.
echo  ============================================================
echo   WARNING - startup was not completed
echo  ============================================================
echo.
echo   Logs: logs\server.log  |  logs\bot.log
echo.
echo  Running services are still active in the background.
echo  Press any key to exit.
echo.
goto :IDLE


rem ============================================================
rem  IDLE - keep window open
rem ============================================================
:IDLE
pause >nul
