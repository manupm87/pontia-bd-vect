"""Shared cell factories and boilerplate for every notebook in the series."""

from __future__ import annotations

from textwrap import dedent

import nbformat
from nbformat import NotebookNode

KERNEL_NAME = "aurum-market-eval"
KERNEL_DISPLAY_NAME = "Python (Aurum Market · Actividad)"


def markdown(source: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str) -> NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n")


def setup_cells() -> list[NotebookNode]:
    """Boilerplate shared by every notebook: root, env, pandas/plotly, config."""
    return [
        code(
            r"""
            import sys
            from pathlib import Path

            project_root = Path.cwd().resolve()
            while not (project_root / "pyproject.toml").exists():
                project_root = project_root.parent
            sys.path.insert(0, str(project_root / "src"))
            print(f"Root del proyecto: {project_root}")
            """
        ),
        code(
            r"""
            import json
            import os

            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TQDM_DISABLE", "1")

            import pandas as pd
            import plotly.io as pio
            from dotenv import load_dotenv

            from aurum_discovery import load_run_config

            load_dotenv(project_root / ".env")
            pio.templates.default = "plotly_white"
            pd.set_option("display.max_colwidth", 90)
            run_config = load_run_config()
            print(f"Configuración final: {run_config.embedding_configuration}")
            """
        ),
    ]


def store_cells() -> list[NotebookNode]:
    """Connection cell shared by every notebook that talks to Qdrant."""
    return [
        code(
            r"""
            from aurum_discovery import CatalogVectorStore, load_embedding_set

            embedding_set = load_embedding_set(run_config.embedding_configuration)
            store = CatalogVectorStore(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY", ""),
                collection_name=os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo"),
                vector_size=embedding_set.configuration.dimension,
                hnsw=run_config.hnsw,
                ef_search=run_config.ef_search,
            )
            store.ping()
            print(f"Colección: {store.collection_name} · registros: {store.count()}")
            """
        ),
    ]


def build_notebook(cells: list[NotebookNode]) -> NotebookNode:
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": KERNEL_DISPLAY_NAME,
                "language": "python",
                "name": KERNEL_NAME,
            },
            "language_info": {"name": "python", "version": "3.12"},
            "case_study": "Aurum Market · Evaluación de Bases de Datos Vectoriales",
        },
    )


__all__ = [
    "KERNEL_DISPLAY_NAME",
    "KERNEL_NAME",
    "build_notebook",
    "code",
    "markdown",
    "setup_cells",
    "store_cells",
]
