"""Execute the notebook series headlessly and save the runs under .artifacts/."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIRECTORY = PROJECT_ROOT / "notebooks"
EXECUTED_DIRECTORY = PROJECT_ROOT / ".artifacts" / "executed"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        action="append",
        help="Nombre de fichero a ejecutar; repetible. Por defecto, toda la serie.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Escribe los outputs también en notebooks/, para entregar la serie "
        "ya ejecutada.",
    )
    return parser.parse_args()


def execute_one(path: Path, *, timeout: int, in_place: bool) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    started_at = perf_counter()
    client.execute()
    elapsed = perf_counter() - started_at
    EXECUTED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, EXECUTED_DIRECTORY / path.name)
    if in_place:
        nbformat.write(notebook, path)
    print(f"OK {path.name}: {elapsed:.1f}s")


def main() -> None:
    """Run the requested notebooks in series order against the live system."""
    arguments = parse_arguments()
    available = sorted(NOTEBOOKS_DIRECTORY.glob("actividad_*.ipynb"))
    if not available:
        raise FileNotFoundError(
            "No hay notebooks en notebooks/. Genera la serie con `make notebook`."
        )
    if arguments.notebook:
        selected = [NOTEBOOKS_DIRECTORY / name for name in arguments.notebook]
        missing = [path.name for path in selected if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Notebooks inexistentes: {missing}.")
    else:
        selected = available
    for path in selected:
        execute_one(path, timeout=arguments.timeout, in_place=arguments.in_place)
    destination = (
        "notebooks/ y .artifacts/executed/"
        if arguments.in_place
        else ".artifacts/executed/"
    )
    print(f"{len(selected)} notebooks ejecutados; outputs en {destination}")


if __name__ == "__main__":
    main()
