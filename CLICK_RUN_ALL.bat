@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=python"

if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON=%ROOT%.venv\Scripts\python.exe"

"%PYTHON%" "%ROOT%easy_run.py" compare --open-images --open-report
set "EXIT_CODE=%ERRORLEVEL%"

echo(
if "%EXIT_CODE%"=="0" (
  echo Benchmark, charts, and report completed.
) else (
  echo Run failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
