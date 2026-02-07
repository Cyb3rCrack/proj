@echo off
REM Zypherus launcher

setlocal enabledelayedexpansion
cd /d "%~dp0..\.."

if "%1"=="" goto help
if "%1"=="repl" goto repl
if "%1"=="test" goto test
if "%1"=="api" goto api
if "%1"=="version" goto version
if "%1"=="check" goto check
goto help

:repl
echo Starting REPL...
call .venv\Scripts\python.exe -m Zypherus.cli.repl %2 %3 %4 %5
goto end

:test
echo Running Zypherus tests...
call .venv\Scripts\python.exe test_quick_integration.py
goto end

:api
echo Starting API server...
call .venv\Scripts\python.exe wsgi.py
goto end

:check
echo Checking installation...
call .venv\Scripts\python.exe -c "import Zypherus; print('Status: OK')"
goto end

:version
call .venv\Scripts\python.exe -c "import Zypherus; print('Zypherus')"
goto end

:help
echo.
echo Zypherus
echo.
echo Usage: zypherus [command]
echo.
echo Commands:
echo   repl        Start REPL
echo   test        Run tests
echo   api         Start API server
echo   check       Health check
echo.
echo Examples:
echo   zypherus repl
echo   zypherus test
echo   zypherus api
echo.
goto end

:end
endlocal
