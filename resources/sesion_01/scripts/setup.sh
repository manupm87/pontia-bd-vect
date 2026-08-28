#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

PYTHON_VERSION="$(tr -d '[:space:]' < .python-version)"
uv python install "$PYTHON_VERSION"
uv sync --all-extras --python "$PYTHON_VERSION"

if [[ ! -f .env ]]; then
    cp .env.example .env
fi

uv run python -m ipykernel install --user \
    --name ecommerce-semantic-search \
    --display-name "Python (BBDD Vectoriales · Sesión 1)"

uv run python scripts/validate_content.py --quick

echo "Entorno preparado. Ejecuta 'make lab' o 'uv run jupyter lab notebooks'."
