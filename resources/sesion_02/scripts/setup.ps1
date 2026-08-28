$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
}

$PythonVersion = (Get-Content .python-version -Raw).Trim()
uv python install $PythonVersion
uv sync --all-extras --python $PythonVersion

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}

uv run python -m ipykernel install --user `
    --name faiss-ann-search `
    --display-name "Python (BBDD Vectoriales · Sesión 2)"

uv run python scripts/validate_content.py --quick
Write-Host "Entorno preparado. Ejecuta 'uv run jupyter lab notebooks'."
