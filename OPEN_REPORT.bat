@echo off
setlocal
set "ROOT=%~dp0"
set "REPORT=%ROOT%result\algorithm_report.html"
set "EXIT_CODE=0"

if not exist "%REPORT%" goto missing_report
start "" "%REPORT%"
goto done

:missing_report
echo Report file not found: "%REPORT%"
echo Run "python easy_run.py compare" first.
set "EXIT_CODE=1"

:done
echo(
if "%EXIT_CODE%"=="0" (
  echo Report opened.
) else (
  echo Open report failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
