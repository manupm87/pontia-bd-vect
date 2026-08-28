.PHONY: setup build validate test lint execute-conceptual smoke-embedded pdf render-pdf clean

setup:
	uv sync --all-extras --group dev
	uv run python -m ipykernel install --user --name bbdd-vectoriales-s03 --display-name "Python (BBDD Vectoriales · Sesión 3)"
	@test -f .env || cp .env.example .env

build:
	uv run python scripts/build_notebooks.py

validate: build
	uv run python scripts/validate_content.py

test:
	uv run pytest

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

execute-conceptual:
	uv run python scripts/execute_notebooks.py --provider conceptual --require-provider

smoke-embedded:
	uv run pytest -m embedded

pdf:
	uv run python scripts/build_memory_pdf.py

render-pdf: pdf
	bash scripts/render_memory_pdf.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find notebooks -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
