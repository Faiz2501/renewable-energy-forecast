$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$python = Join-Path $backend ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.11 -m venv (Join-Path $backend ".venv")
    } else {
        $systemPython = Get-Command python -ErrorAction SilentlyContinue
        if (-not $systemPython) { throw "Python 3.11 is required. Install it from python.org, then run this script again." }
        & python -m venv (Join-Path $backend ".venv")
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the backend virtual environment." }
}

$imports = "import fastapi, uvicorn, pandas, numpy, sklearn, torch, multipart"
& $python -c $imports 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the forecast backend dependencies (first run only)..." -ForegroundColor Yellow
    & $python -m pip install -r (Join-Path $backend "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Check your internet connection and try again." }
}

try { Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 2 | Out-Null }
catch {
    $apiOut = Join-Path $root "backend-server.log"
    $apiErr = Join-Path $root "backend-server-error.log"
    Start-Process -FilePath $python -ArgumentList @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000") -WorkingDirectory $backend -WindowStyle Hidden -RedirectStandardOutput $apiOut -RedirectStandardError $apiErr | Out-Null
}

try { Invoke-WebRequest "http://127.0.0.1:3000" -TimeoutSec 2 | Out-Null }
catch {
    $webOut = Join-Path $root "frontend-server.log"
    $webErr = Join-Path $root "frontend-server-error.log"
    Start-Process -FilePath $python -ArgumentList @("-m", "http.server", "3000", "--bind", "127.0.0.1") -WorkingDirectory $frontend -WindowStyle Hidden -RedirectStandardOutput $webOut -RedirectStandardError $webErr | Out-Null
}

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try { $health = Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 2; $ready = $true; break }
    catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { throw "The API did not start. See backend-server-error.log in this folder." }
Start-Process "http://localhost:3000"
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "Backend:  http://localhost:8000 ($($health.status))" -ForegroundColor Green
Write-Host "Upload intermittent-renewables-production-france.csv and click Generate forecast. The first request trains the LSTM; later requests with the same file reuse it." -ForegroundColor Cyan
