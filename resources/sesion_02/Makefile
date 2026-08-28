SHELL := /bin/bash

.PHONY: setup lab data embeddings notebook test lint format validate verify clean

setup:
	bash scripts/setup.sh

lab:
	uv run jupyter lab notebooks

data:
	@test -n "$(ESCI_PRODUCTS)" || (echo "Falta ESCI_PRODUCTS=/ruta/products.parquet" && exit 1)
	@test -n "$(ESCI_JUDGMENTS)" || (echo "Falta ESCI_JUDGMENTS=/ruta/judgments.csv" && exit 1)
	@test -n "$(SEMANTIC_QUERIES)" || (echo "Falta SEMANTIC_QUERIES=/ruta/semantic_queries.csv" && exit 1)
	uv run python scripts/prepare_catalog.py --products "$(ESCI_PRODUCTS)" --judgments "$(ESCI_JUDGMENTS)" --semantic-queries "$(SEMANTIC_QUERIES)"

embeddings:
	uv run python scripts/build_embeddings.py

notebook:
	uv run python scripts/build_notebook.py

test:
	uv run pytest

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

format:
	uv run ruff check --fix src scripts tests
	uv run ruff format src scripts tests

validate:
	uv run python scripts/validate_content.py

verify: lint test validate
	uv run python scripts/execute_notebook.py

clean:
	rm -rf .artifacts .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
