@echo off
REM ===================================================================
REM  Pipeline supervisor launcher (runs automatically at Windows boot
REM  when registered with Task Scheduler). After a power failure, a
REM  crash or a closed window it resumes where it left off.
REM
REM  Can also be started by hand: double-click.
REM  For a different job: auto_run.bat "run_external.py --experiments benchmark,proposed"
REM ===================================================================

REM %~dp0 = the folder of this .bat file (tools\). Move to the project root.
cd /d "%~dp0.."

REM Python of the project virtual environment (.venv); fall back to the PATH.
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

REM Keep non-ASCII characters from crashing the console
set PYTHONIOENCODING=utf-8

echo ============================================================
echo  Supervisor starting
echo  Python  : %PY%
echo  Project : %CD%
echo ============================================================

if "%~1"=="" (
    "%PY%" "tools\auto_run.py"
) else (
    "%PY%" "tools\auto_run.py" --cmd "%~1"
)

echo.
echo Supervisor finished. You can close this window.
timeout /t 20 >nul
exit /b %ERRORLEVEL%
