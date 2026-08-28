#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv python install 3.12
uv sync --all-extras --group dev

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

uv run python -m ipykernel install --user \
  --name bbdd-vectoriales-s03 \
  --display-name "Python (BBDD Vectoriales · Sesión 3)"
uv run python scripts/build_notebooks.py
uv run python scripts/validate_content.py --quick

echo "Entorno preparado. Inicia JupyterLab con: uv run jupyter lab notebooks"
