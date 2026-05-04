@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=python"

if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON=%ROOT%.venv\Scripts\python.exe"

"%PYTHON%" "%ROOT%easy_run.py" plot --open-images
set "EXIT_CODE=%ERRORLEVEL%"

echo(
if "%EXIT_CODE%"=="0" (
  echo Plot completed.
) else (
  echo Plot failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
