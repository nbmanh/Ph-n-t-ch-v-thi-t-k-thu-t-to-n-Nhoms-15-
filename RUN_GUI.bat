@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=python"

if exist "%ROOT%.venv\Scripts\pythonw.exe" (
  start "" "%ROOT%.venv\Scripts\pythonw.exe" "%ROOT%easy_run.py" gui
  exit /b 0
)

if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON=%ROOT%.venv\Scripts\python.exe"

"%PYTHON%" "%ROOT%easy_run.py" gui
set "EXIT_CODE=%ERRORLEVEL%"

echo(
if "%EXIT_CODE%"=="0" (
  echo GUI closed successfully.
) else (
  echo GUI exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
