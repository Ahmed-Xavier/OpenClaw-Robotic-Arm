@echo off
title Launcher — Robot Arm & Mini Openclaw
cd /d "%~dp0"

echo ======================================================
echo   Launching Robot Arm Simulation ^& Telegram Bot
echo ======================================================
echo.

:: 1. Launch server.py in a dedicated terminal window
echo [1/2] Starting MuJoCo Flask Server (server.py)...
start "Robot Arm Simulation Server" cmd /k "title Robot Arm Server && cd /d "%~dp0" && python server.py"

:: 2. Give the simulation server 3 seconds to initialize and bind port 8765
echo Waiting 3 seconds for server to initialize...
timeout /t 3 /nobreak >nul

:: 3. Launch bot.py in a dedicated terminal window
echo [2/2] Starting Mini Openclaw Telegram Bot (bot.py)...
start "Mini Openclaw Bot" cmd /k "title Mini Openclaw Bot && cd /d "%~dp0Mini Openclaw" && python bot.py"

echo.
echo ======================================================
echo   Both services launched in separate windows!
echo   You can close this launcher window now.
echo ======================================================
timeout /t 4 >nul
