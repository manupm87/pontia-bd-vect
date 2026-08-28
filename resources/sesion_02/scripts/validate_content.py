"""Validate the self-contained FAISS session and its teaching notebook."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"
EXPECTED_NOTEBOOK = "sesion_02_faiss_indices_ann.ipynb"


def parse_arguments() -> argparse.Namespace:
    """Parse the installation-time quick flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def validate_notebook() -> dict[str, int]:
    """Check narrative depth, code size, syntax, and required concepts."""
    paths = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))
    if [path.name for path in paths] != [EXPECTED_NOTEBOOK]:
        raise AssertionError(f"Expected one notebook, found: {paths}")
    notebook = nbformat.read(paths[0], as_version=4)
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown_words = sum(len(cell.source.split()) for cell in markdown_cells)
    maximum_code_lines = max(
        sum(bool(line.strip()) for line in cell.source.splitlines())
        for cell in code_cells
    )
    if markdown_words < 5_000:
        raise AssertionError(f"Only {markdown_words} Markdown words")
    if maximum_code_lines > 24:
        raise AssertionError(f"A code cell has {maximum_code_lines} nonblank lines")
    single_letter_names = []
    for position, cell in enumerate(code_cells):
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and len(node.id) == 1 and node.id != "_":
                single_letter_names.append((position, node.id))
    if single_letter_names:
        raise AssertionError(f"Single-letter variables: {single_letter_names}")
    notebook_text = "\n".join(cell.source for cell in notebook.cells).lower()
    required_terms = [
        "indexflatip",
        "indexivfflat",
        "indexhnswflat",
        "indexivfpq",
        "recall@10",
        "nprobe",
        "efsearch",
        "product quantization",
        "persistencia",
        "metadatos",
    ]
    missing_terms = [term for term in required_terms if term not in notebook_text]
    if missing_terms:
        raise AssertionError(f"Missing concepts: {missing_terms}")
    if "matplotlib" in notebook_text or "seaborn" in notebook_text:
        raise AssertionError("Notebook charts must use Plotly")
    forbidden_meta = ["mapa de la sesión", "minutos de clase", "como te pedí"]
    if any(term in notebook_text for term in forbidden_meta):
        raise AssertionError("Student notebook contains instructor meta-content")
    return {
        "cells": len(notebook.cells),
        "markdown_cells": len(markdown_cells),
        "code_cells": len(code_cells),
        "markdown_words": markdown_words,
        "maximum_code_lines": maximum_code_lines,
    }


def validate_data() -> dict[str, int]:
    """Check committed counts and required binary artifacts."""
    data_directory = PROJECT_ROOT / "data" / "esci"
    manifest = json.loads(
        (data_directory / "manifest.json").read_text(encoding="utf-8")
    )
    expected_counts = {
        "products": 50_000,
        "judgments": 336,
        "business_queries": 20,
        "probe_queries": 256,
    }
    if manifest["counts"] != expected_counts:
        raise AssertionError(f"Unexpected counts: {manifest['counts']}")
    for filename in ["product_embeddings.npy", "query_embeddings.npy"]:
        if not (data_directory / filename).exists():
            raise AssertionError(f"Missing {filename}")
    return expected_counts


def validate_environment() -> None:
    """Ensure no credential is committed in the local template."""
    for filename in [".env", ".env.example"]:
        path = PROJECT_ROOT / filename
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"')
        if values.get("HF_TOKEN"):
            raise AssertionError(f"HF_TOKEN must remain blank in {filename}")


def main() -> None:
    """Run every static validation."""
    arguments = parse_arguments()
    notebook_stats = validate_notebook()
    data_stats = validate_data()
    validate_environment()
    mode = "quick" if arguments.quick else "full"
    print(f"Validation OK ({mode}): {notebook_stats}; data={data_stats}")


if __name__ == "__main__":
    main()
