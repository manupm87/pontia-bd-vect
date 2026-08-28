"""Execute the complete teaching notebook and save a reviewable artifact."""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "sesion_02_faiss_indices_ann.ipynb"
OUTPUT_DIRECTORY = PROJECT_ROOT / ".artifacts" / "executed"


def main() -> None:
    """Execute the notebook from the project root."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    started_at = time.perf_counter()
    executed_notebook = client.execute()
    elapsed_seconds = time.perf_counter() - started_at
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / NOTEBOOK_PATH.name
    nbformat.write(executed_notebook, output_path)
    print(f"OK {NOTEBOOK_PATH.name}: {elapsed_seconds:.1f} s")


if __name__ == "__main__":
    main()
