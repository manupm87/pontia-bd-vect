$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

uv python install 3.12
uv sync --all-extras --group dev

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

uv run python -m ipykernel install --user `
    --name bbdd-vectoriales-s03 `
    --display-name "Python (BBDD Vectoriales · Sesión 3)"
uv run python scripts/build_notebooks.py
uv run python scripts/validate_content.py --quick

Write-Host "Entorno preparado. Inicia JupyterLab con: uv run jupyter lab notebooks"
