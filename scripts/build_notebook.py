"""Build the R&D notebook series cell by cell, mirroring the course sessions."""

from __future__ import annotations

from pathlib import Path

import nbformat
from notebooks_src import (
    nb_00_caso_datos_baseline,
    nb_01_representacion,
    nb_02_indice_bbdd,
    nb_03_recuperacion_filtros,
    nb_04_operaciones_duplicados,
    nb_05_evaluacion_analisis,
)
from notebooks_src.common import build_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIRECTORY = PROJECT_ROOT / "notebooks"

NOTEBOOK_MODULES = (
    nb_00_caso_datos_baseline,
    nb_01_representacion,
    nb_02_indice_bbdd,
    nb_03_recuperacion_filtros,
    nb_04_operaciones_duplicados,
    nb_05_evaluacion_analisis,
)


def main() -> None:
    """Assemble every notebook in the series with the shared kernel metadata."""
    NOTEBOOKS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for module in NOTEBOOK_MODULES:
        notebook = build_notebook(module.build_cells())
        path = NOTEBOOKS_DIRECTORY / module.FILENAME
        nbformat.write(notebook, path)
        print(f"Notebook escrito en {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
