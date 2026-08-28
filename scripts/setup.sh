#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12
uv sync --all-extras --group dev

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Creado .env a partir de .env.example."
fi

uv run python -m ipykernel install --user --name aurum-market-eval \
    --display-name "Python (Aurum Market · Actividad)"
uv run python scripts/build_notebook.py

echo ""
echo "Entorno preparado. Próximos pasos:"
echo "  1. make up          # arranca Qdrant en Docker"
echo "  2. make embeddings  # genera los embeddings (usa GPU si está disponible)"
echo "  3. make pipeline    # ejecuta el recorrido completo"
