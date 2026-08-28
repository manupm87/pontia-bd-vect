$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Instalando uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv python install 3.12
uv sync --all-extras --group dev

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Creado .env a partir de .env.example."
}

uv run python -m ipykernel install --user --name aurum-market-eval --display-name "Python (Aurum Market · Actividad)"
uv run python scripts/build_notebook.py

Write-Host ""
Write-Host "Entorno preparado. Próximos pasos:"
Write-Host "  1. make up          # arranca Qdrant en Docker"
Write-Host "  2. make embeddings  # genera los embeddings"
Write-Host "  3. make pipeline    # ejecuta el recorrido completo"
