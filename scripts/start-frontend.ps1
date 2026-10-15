$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Frontend = Join-Path $ProjectRoot 'frontend'
Set-Location $Frontend
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host 'Node.js/npm was not found. Install Node.js LTS from https://nodejs.org/ and reopen PowerShell.' -ForegroundColor Red
    exit 1
}
if (-not (Test-Path 'node_modules')) {
    Write-Host 'Installing frontend packages (first run only)...' -ForegroundColor Cyan
    npm install
    if ($LASTEXITCODE -ne 0) { throw 'npm install failed.' }
}
Write-Host 'Starting website at http://localhost:5173. Keep this window open.' -ForegroundColor Green
npm run dev -- --host 127.0.0.1
