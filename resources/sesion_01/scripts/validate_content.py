"""Validate the teaching artifact, data contract, and offline safety."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"
EXPECTED_NOTEBOOK = "sesion_01_buscador_semantico_ecommerce.ipynb"
FORBIDDEN_CASE_TERMS = (
    "na" + "sa",
    "luna" + "net",
    "rego" + "lito",
    "lunar " + "engineering",
)


def parse_arguments() -> argparse.Namespace:
    """Parse the optional quick mode used during installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def validate_notebook() -> dict[str, int]:
    """Validate count, narrative volume, code size, and visualization policy."""
    notebook_paths = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))
    names = [path.name for path in notebook_paths]
    if names != [EXPECTED_NOTEBOOK]:
        raise AssertionError(f"Expected one notebook, found: {names}")

    notebook = nbformat.read(notebook_paths[0], as_version=4)
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown_words = sum(len(cell.source.split()) for cell in markdown_cells)
    maximum_code_lines = max(
        sum(bool(line.strip()) for line in cell.source.splitlines())
        for cell in code_cells
    )

    if len(notebook.cells) < 70:
        raise AssertionError("The narrative is unexpectedly short")
    if markdown_words < 3_500:
        raise AssertionError(f"Only {markdown_words} Markdown words")
    if maximum_code_lines > 24:
        raise AssertionError(f"A code cell has {maximum_code_lines} nonblank lines")

    notebook_text = "\n".join(cell.source for cell in notebook.cells).lower()
    if "plotly" not in notebook_text:
        raise AssertionError("Plotly is required for notebook charts")
    if "matplotlib" in notebook_text or "seaborn" in notebook_text:
        raise AssertionError("Only Plotly may be used for charts")
    if "run_provider_examples" in notebook_text:
        raise AssertionError("The notebook must not hide APIs behind a run flag")
    api_cells = [
        cell
        for cell in code_cells
        if "requires-api-key" in cell.metadata.get("tags", [])
    ]
    if len(api_cells) < 10:
        raise AssertionError("Live provider cells are missing")

    return {
        "cells": len(notebook.cells),
        "markdown_cells": len(markdown_cells),
        "code_cells": len(code_cells),
        "markdown_words": markdown_words,
        "maximum_code_lines": maximum_code_lines,
    }


def validate_data() -> dict[str, int]:
    """Validate the committed ESCI snapshot metadata."""
    manifest_path = PROJECT_ROOT / "data" / "esci" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_counts = {"products": 336, "queries": 12, "judgments": 336}
    actual_counts = manifest["counts"]
    if any(actual_counts.get(key) != value for key, value in expected_counts.items()):
        raise AssertionError(f"Unexpected ESCI counts: {actual_counts}")
    if "Apache" not in manifest["source_license"]:
        raise AssertionError("Dataset license is not documented")
    return expected_counts


def validate_environment_template() -> None:
    """Ensure local credentials remain blank and remote calls remain disabled."""
    environment_path = PROJECT_ROOT / ".env"
    for path in [environment_path, PROJECT_ROOT / ".env.example"]:
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"')
        for key in ["OPENAI_API_KEY", "COHERE_API_KEY", "GEMINI_API_KEY", "HF_TOKEN"]:
            if values.get(key):
                raise AssertionError(f"{key} must be blank in {path.name}")


def validate_removed_case() -> None:
    """Ensure no prose or code from the discarded case remains."""
    checked_suffixes = {".md", ".py", ".ipynb", ".json", ".toml"}
    offenders = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in checked_suffixes:
            continue
        if any(part.startswith(".") for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(term in content for term in FORBIDDEN_CASE_TERMS):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    if offenders:
        raise AssertionError(f"Discarded case remains in: {offenders}")


def main() -> None:
    """Run all static validations and print a compact report."""
    arguments = parse_arguments()
    notebook_stats = validate_notebook()
    data_stats = validate_data()
    validate_environment_template()
    validate_removed_case()
    mode = "quick" if arguments.quick else "full"
    print(f"Validation OK ({mode}): {notebook_stats}; data={data_stats}")


if __name__ == "__main__":
    main()
