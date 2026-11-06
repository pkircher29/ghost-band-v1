# Starts the local Cover Studio app; all AI runs on Modal (deploy cloud/cover_ai.py first).
$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Create the app environment first (see README.md)." }
$Port = if ($env:COVER_STUDIO_PORT) { $env:COVER_STUDIO_PORT } else { "8420" }
Start-Process "http://127.0.0.1:$Port/"
& $Python -m uvicorn server:app --app-dir (Join-Path $PSScriptRoot "app") --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
