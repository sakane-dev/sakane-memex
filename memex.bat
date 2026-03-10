@echo off
setlocal enabledelayedexpansion

:MENU
cls
echo.
echo  =========================================
echo   sakane-memex v2026.1
echo  =========================================
echo   [1] Start   - Ollama + uvicorn
echo   [2] Stop    - Ollama + uvicorn
echo   [3] Restart - Ollama + uvicorn
echo   [4] Status
echo   [5] Browser - http://localhost:8000
echo   [0] Exit
echo  =========================================
echo.
set /p CHOICE= Select [0-5]: 

if "%CHOICE%"=="1" goto START
if "%CHOICE%"=="2" goto STOP
if "%CHOICE%"=="3" goto RESTART
if "%CHOICE%"=="4" goto STATUS
if "%CHOICE%"=="5" goto BROWSER
if "%CHOICE%"=="0" goto EXIT
goto MENU

:START
echo.
echo [*] Starting Ollama...
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if not errorlevel 1 (
    echo [!] Ollama already running. Skip.
) else (
    start "Ollama" /min cmd /c "ollama serve"
    timeout /t 3 /nobreak >nul
    echo [+] Ollama started.
)

echo [*] Starting uvicorn...
netstat -ano 2>nul | find ":8000 " | find "LISTENING" >nul
if not errorlevel 1 (
    echo [!] Port 8000 already in use. Skip.
) else (
    start "uvicorn" /min cmd /c "cd /d D:\sakane-memex && .venv\Scripts\activate && uvicorn src.api.main:app --port 8000"
    timeout /t 3 /nobreak >nul
    echo [+] uvicorn started.
)

echo.
echo [+] Done. Web UI: http://localhost:8000
echo.
pause
goto MENU

:STOP
echo.
echo [*] Stopping uvicorn (port 8000)...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| find ":8000 " ^| find "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
    echo [+] Killed PID %%a
)

echo [*] Stopping Ollama...
taskkill /f /im ollama.exe >nul 2>&1
if not errorlevel 1 (
    echo [+] Ollama stopped.
) else (
    echo [!] Ollama was not running.
)

echo.
echo [+] All services stopped.
echo.
pause
goto MENU

:RESTART
echo.
echo [*] Stopping all services...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| find ":8000 " ^| find "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)
taskkill /f /im ollama.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo [+] Stopped. Restarting...
goto START

:STATUS
echo.
echo  -----------------------------------------
echo   Process Status
echo  -----------------------------------------

tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if not errorlevel 1 (
    echo   Ollama  : RUNNING
) else (
    echo   Ollama  : STOPPED
)

netstat -ano 2>nul | find ":8000 " | find "LISTENING" >nul
if not errorlevel 1 (
    echo   uvicorn : RUNNING  (http://localhost:8000)
) else (
    echo   uvicorn : STOPPED
)

echo  -----------------------------------------
echo.
pause
goto MENU

:BROWSER
start http://localhost:8000
goto MENU

:EXIT
exit /b 0
