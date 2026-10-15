$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $ProjectRoot 'backend'
Set-Location $Backend
if (-not (Test-Path '.venv\Scripts\Activate.ps1')) {
    Write-Host 'Backend environment not found. Run .\scripts\setup-backend.ps1 from the project root first.' -ForegroundColor Red
    exit 1
}
if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host 'Created backend\.env from .env.example. Chat model: qwen2.5:7b; embedding model: bge-m3.' -ForegroundColor Yellow
}
.\.venv\Scripts\Activate.ps1
Write-Host 'Starting API at http://127.0.0.1:8000. Keep this window open.' -ForegroundColor Cyan
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
