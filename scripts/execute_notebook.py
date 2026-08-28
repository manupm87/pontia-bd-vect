"""Execute the demo notebook headlessly and save the run under .artifacts/."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "actividad_aurum_market.ipynb"
EXECUTED_DIRECTORY = PROJECT_ROOT / ".artifacts" / "executed"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Escribe los outputs también en notebooks/, para entregar el "
        "notebook ya ejecutado.",
    )
    return parser.parse_args()


def main() -> None:
    """Run every cell against the live system and persist the executed copy."""
    arguments = parse_arguments()
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"No existe {NOTEBOOK_PATH}. Genera el notebook con `make notebook`."
        )
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=arguments.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    started_at = perf_counter()
    client.execute()
    elapsed = perf_counter() - started_at
    EXECUTED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    executed_path = EXECUTED_DIRECTORY / NOTEBOOK_PATH.name
    nbformat.write(notebook, executed_path)
    print(f"OK {NOTEBOOK_PATH.name}: {elapsed:.1f}s")
    print(f"Copia ejecutada en {executed_path.relative_to(PROJECT_ROOT)}")
    if arguments.in_place:
        nbformat.write(notebook, NOTEBOOK_PATH)
        print(f"Outputs escritos en {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
