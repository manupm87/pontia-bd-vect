SHELL := /bin/bash
.PHONY: setup up down down-volumes embeddings ingest experiments evaluate sweep-ef search-results duplicates events evidence pipeline metrics search notebook execute-notebook lab test test-integration lint format verify informe clean

setup:
	bash scripts/setup.sh

up:
	docker compose -f deploy/qdrant/compose.yaml up -d --wait

down:
	docker compose -f deploy/qdrant/compose.yaml down

down-volumes:
	docker compose -f deploy/qdrant/compose.yaml down --volumes

embeddings:
	uv run python scripts/build_embeddings.py

ingest:
	uv run python scripts/ingest_catalog.py

experiments:
	uv run python scripts/run_experiments.py

validate-challengers:
	uv run --extra validation python scripts/validate_challengers.py

evaluate:
	uv run python scripts/evaluate_system.py

sweep-ef:
	uv run python scripts/sweep_ef_search.py

search-results:
	uv run python scripts/generate_search_results.py

duplicates:
	uv run python scripts/calibrate_duplicates.py
	uv run python scripts/generate_duplicate_results.py

events:
	uv run python scripts/apply_events.py

evidence:
	uv run python scripts/collect_evidence.py

pipeline: embeddings ingest experiments duplicates evaluate sweep-ef search-results events evidence

metrics: evaluate

search:
	@test -n "$(q)" || (echo 'Falta la consulta: make search q="taladro 24v"' && exit 1)
	uv run python scripts/search_cli.py "$(q)" --top-k $(or $(k),10) $(if $(brand),--brand "$(brand)")

notebook:
	uv run python scripts/build_notebook.py

execute-notebook: notebook
	uv run python scripts/execute_notebook.py

lab:
	uv run jupyter lab notebooks

test:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

format:
	uv run ruff check --fix src scripts tests
	uv run ruff format src scripts tests

verify: lint test

informe:
	uv run --extra pdf python scripts/render_informe.py

clean:
	rm -rf .artifacts .pytest_cache .ruff_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
