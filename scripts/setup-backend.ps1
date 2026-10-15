$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $ProjectRoot 'backend'
Set-Location $Backend

Write-Host 'Checking installed Python versions...' -ForegroundColor Cyan
$PythonSelector = $null
$pyVersion = $null
foreach ($candidate in @('3.14', '3.12')) {
    try {
        $candidateVersion = & py ("-" + $candidate) --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $PythonSelector = $candidate
            $pyVersion = $candidateVersion
            break
        }
    } catch { }
}
if (-not $PythonSelector) {
    Write-Host 'Python 3.14 or 3.12 was not found. Install Python 3.12 (64-bit) or use your existing Python 3.14, then reopen PowerShell.' -ForegroundColor Yellow
    Write-Host 'Download Python from https://www.python.org/downloads/' -ForegroundColor Yellow
    exit 1
}
Write-Host "Using $pyVersion" -ForegroundColor Green
if ($PythonSelector -eq '3.14') {
    Write-Host 'Python 3.14 is supported by current PyTorch releases, but if any dependency fails to install, use Python 3.12 for the most conservative setup.' -ForegroundColor Yellow
}

if (Test-Path '.venv') {
    $venvPython = Join-Path $Backend '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        $existingVersion = & $venvPython --version 2>&1
        if ($existingVersion -notmatch '3\.(12|14)\.') {
            Write-Host "Existing .venv uses $existingVersion, not a supported project Python version (3.12/3.14)." -ForegroundColor Yellow
            Write-Host 'To recreate only this virtual environment, run: Remove-Item -Recurse -Force .venv' -ForegroundColor Yellow
            Write-Host 'Then rerun .\scripts\setup-backend.ps1 from the project root.' -ForegroundColor Yellow
            exit 1
        }
    }
} else {
    & py ("-" + $PythonSelector) -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment.' }
}

$VenvPython = Join-Path $Backend '.venv\Scripts\python.exe'
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
& $VenvPython -m pip install -r (Join-Path $Backend 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

if (-not (Test-Path (Join-Path $Backend '.env'))) {
    Copy-Item (Join-Path $Backend '.env.example') (Join-Path $Backend '.env')
}
Write-Host ''
Write-Host 'Backend setup finished.' -ForegroundColor Green
Write-Host 'Next, from this backend folder, activate the environment and index documents:' -ForegroundColor Cyan
Write-Host '.\.venv\Scripts\Activate.ps1'
Write-Host 'python -m app.ingest'
Write-Host 'uvicorn app.main:app --reload --host 127.0.0.1 --port 8000'
