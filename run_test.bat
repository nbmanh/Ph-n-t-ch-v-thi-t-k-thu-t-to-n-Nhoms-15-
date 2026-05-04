@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=python"

if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON=%ROOT%.venv\Scripts\python.exe"

"%PYTHON%" "%ROOT%easy_run.py" test
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" if exist "%ROOT%result\test_tour_seq5.png" start "" "%ROOT%result\test_tour_seq5.png"

echo(
if "%EXIT_CODE%"=="0" (
  echo Test completed.
) else (
  echo Test failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
