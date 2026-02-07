#Requires -Version 5.1
[System.Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseApprovedVerbs', '', Justification='External Python entry points')]
param()

function Open-ZypherusREPL {
    Write-Host "Starting REPL..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe main.py @args
}

function Invoke-ZypherusTest {
    Write-Host "Running tests..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe test_quick_integration.py @args
}

function Start-ZypherusAPI {
    Write-Host "Starting API server..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe wsgi.py
}

function Get-ZypherusVersion {
    & .\.venv\Scripts\python.exe -c "import ace; print(f'Zypherus v{ace.__version__}')"
}

function Test-ZypherusInstallation {
    Write-Host "Checking installation..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe -c "import Zypherus; print('Status: OK')"
}

Write-Host "Zypherus functions loaded:" -ForegroundColor Green
Write-Host "  Open-ZypherusREPL, Invoke-ZypherusTest, Start-ZypherusAPI, Get-ZypherusVersion, Test-ZypherusInstallation" -ForegroundColor Cyan
