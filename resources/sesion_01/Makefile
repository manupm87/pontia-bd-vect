SHELL := /bin/bash

.PHONY: setup lab data model notebooks test lint format validate verify clean

setup:
	bash scripts/setup.sh

lab:
	uv run jupyter lab notebooks

data:
	@test -n "$(ESCI_EXAMPLES)" || (echo "Falta ESCI_EXAMPLES=/ruta/examples.parquet" && exit 1)
	@test -n "$(ESCI_PRODUCTS)" || (echo "Falta ESCI_PRODUCTS=/ruta/products.parquet" && exit 1)
	uv run --with duckdb python scripts/prepare_esci_sample.py --examples "$(ESCI_EXAMPLES)" --products "$(ESCI_PRODUCTS)"

model:
	uv run python scripts/build_reference_embeddings.py

notebooks:
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
	uv run python scripts/execute_notebooks.py

clean:
	rm -rf .artifacts .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
