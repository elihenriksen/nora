@echo off
REM Nora launcher for Windows.
REM Double-click in Explorer or run from cmd / PowerShell:
REM   .\Nora.bat              normal launch
REM   .\Nora.bat --debug      with NORA_DEBUG=1 propagated to subprocesses
REM
REM Linux/macOS users: use Nora.command (macOS) or Nora.sh (Linux).
REM The cross-platform fallback is `python -m nora` from the project root.

setlocal

REM Move to the directory containing this script, regardless of where it was launched from.
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo No virtualenv found at .venv\Scripts\python.exe
    echo.
    echo Set it up with:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -e .
    echo.
    pause
    exit /b 1
)

"%PY%" -c "import nora" >nul 2>&1
if errorlevel 1 (
    echo The 'nora' package isn't installed in .venv
    echo.
    echo Install it with:
    echo   .venv\Scripts\pip install -e .
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo No .env file found at .env
    echo.
    echo Create one with:
    echo   copy .env.example .env
    echo   notepad .env
    echo.
    pause
    exit /b 1
)

cls
"%PY%" -m nora %*
endlocal
