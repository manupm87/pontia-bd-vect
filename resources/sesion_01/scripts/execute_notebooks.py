"""Execute the local path of every notebook without consuming API credits."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"
ARTIFACT_DIRECTORY = PROJECT_ROOT / ".artifacts" / "executed"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        action="append",
        help="Nombre de notebook concreto. Se puede repetir.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Tiempo máximo por celda, en segundos.",
    )
    return parser.parse_args()


def discover_notebooks(selected_names: list[str] | None) -> list[Path]:
    """Return notebooks in teaching order."""
    available_paths = sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))
    if not selected_names:
        return available_paths

    selected_set = set(selected_names)
    selected_paths = [
        notebook_path
        for notebook_path in available_paths
        if notebook_path.name in selected_set
    ]
    missing_names = selected_set - {path.name for path in selected_paths}
    if missing_names:
        missing_display = ", ".join(sorted(missing_names))
        raise FileNotFoundError(f"No se encontraron: {missing_display}")
    return selected_paths


def execute_notebook(notebook_path: Path, timeout_seconds: int) -> float:
    """Execute one notebook, replacing only cells tagged as requiring API keys."""
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if "requires-api-key" in cell.metadata.get("tags", []):
            cell.source = "print('Celda API omitida en la validación local.')"
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )

    started_at = time.perf_counter()
    executed_notebook = client.execute()
    elapsed_seconds = time.perf_counter() - started_at

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACT_DIRECTORY / notebook_path.name
    nbformat.write(executed_notebook, output_path)
    return elapsed_seconds


def main() -> None:
    """Execute the requested notebooks and report concise timings."""
    arguments = parse_arguments()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    notebook_paths = discover_notebooks(arguments.notebook)
    if not notebook_paths:
        raise FileNotFoundError(f"No hay notebooks en {NOTEBOOK_DIRECTORY}")

    total_started_at = time.perf_counter()
    for notebook_path in notebook_paths:
        elapsed_seconds = execute_notebook(notebook_path, arguments.timeout)
        print(f"OK {notebook_path.name}: {elapsed_seconds:.1f} s")

    total_seconds = time.perf_counter() - total_started_at
    print(f"{len(notebook_paths)} notebooks ejecutados en {total_seconds:.1f} s")


if __name__ == "__main__":
    main()
