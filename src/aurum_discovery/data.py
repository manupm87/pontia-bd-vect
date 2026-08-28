"""Validated loading of the Aurum Market snapshot and deterministic batching."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import DATA_DIRECTORY
from .contracts import VectorRecord

RECORD_NAMESPACE = uuid.UUID("34ef9344-7a3f-5eb2-b30b-aceff745758d")

CATALOG_COLUMNS = (
    "record_id",
    "product_id",
    "title",
    "brand",
    "color",
    "locale",
    "text",
    "catalog_version",
    "active",
)

CATALOG_PATH = DATA_DIRECTORY / "catalogo_productos.csv.gz"
CATALOG_SAMPLE_PATH = DATA_DIRECTORY / "catalogo_muestra.csv"
DEVELOPMENT_QUERIES_PATH = DATA_DIRECTORY / "consultas_desarrollo.csv"
DEVELOPMENT_JUDGMENTS_PATH = DATA_DIRECTORY / "relevancias_desarrollo.csv"
EVALUATION_QUERIES_PATH = DATA_DIRECTORY / "consultas_evaluacion.csv"
FILTERED_QUERIES_PATH = DATA_DIRECTORY / "consultas_filtradas.csv"
CATALOG_EVENTS_PATH = DATA_DIRECTORY / "eventos_catalogo.csv"
INCOMING_DEVELOPMENT_PATH = DATA_DIRECTORY / "altas_desarrollo.csv"
INCOMING_EVALUATION_PATH = DATA_DIRECTORY / "altas_evaluacion.csv"
MANIFEST_PATH = DATA_DIRECTORY / "manifest.json"


def record_id_for_product(product_id: str) -> str:
    """Return the stable UUIDv5 that identifies a product in the vector store."""
    normalized = product_id.strip()
    if not normalized:
        raise ValueError("product_id no puede estar vacío.")
    return str(uuid.uuid5(RECORD_NAMESPACE, normalized))


def _read_csv(path: Any, **kwargs: Any) -> pd.DataFrame:
    """Read a snapshot CSV keeping empty cells as empty strings, never NaN."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def _parse_bool_column(values: pd.Series, *, column: str) -> pd.Series:
    normalized = values.str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(
            f"La columna {column!r} contiene valores no booleanos: {unexpected}."
        )
    return normalized == "true"


def load_manifest() -> dict[str, Any]:
    """Return the provenance manifest shipped with the snapshot."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_catalog(*, sample: bool = False) -> pd.DataFrame:
    """Load the product catalog (full by default, 1.500-row sample on demand)."""
    path = CATALOG_SAMPLE_PATH if sample else CATALOG_PATH
    catalog = _read_csv(path)
    _validate_catalog(catalog, expected_rows=1_500 if sample else 15_000)
    catalog = catalog.copy()
    catalog["catalog_version"] = catalog["catalog_version"].astype(int)
    catalog["active"] = _parse_bool_column(catalog["active"], column="active")
    return catalog


def load_development_queries() -> pd.DataFrame:
    """Load the eight labeled development queries."""
    queries = _read_csv(DEVELOPMENT_QUERIES_PATH)
    _require_columns(
        queries,
        ("workload_id", "query_id", "query_text", "query_type"),
        name="consultas_desarrollo",
    )
    if len(queries) != 8:
        raise ValueError(
            f"consultas_desarrollo.csv debe tener 8 consultas, no {len(queries)}."
        )
    return queries


def load_development_judgments() -> pd.DataFrame:
    """Load the graded ESCI judgments for the development queries."""
    judgments = _read_csv(DEVELOPMENT_JUDGMENTS_PATH)
    _require_columns(
        judgments,
        ("query_id", "product_id", "esci_label", "relevance"),
        name="relevancias_desarrollo",
    )
    judgments = judgments.copy()
    judgments["relevance"] = judgments["relevance"].astype(int)
    unexpected = sorted(set(judgments["esci_label"]) - {"E", "S", "C", "I"})
    if unexpected:
        raise ValueError(f"Etiquetas ESCI desconocidas: {unexpected}.")
    return judgments


def load_evaluation_queries() -> pd.DataFrame:
    """Load the twelve blind evaluation queries."""
    queries = _read_csv(EVALUATION_QUERIES_PATH)
    _require_columns(
        queries,
        ("evaluation_id", "query_text", "query_type"),
        name="consultas_evaluacion",
    )
    if len(queries) != 12:
        raise ValueError(
            f"consultas_evaluacion.csv debe tener 12 consultas, no {len(queries)}."
        )
    return queries


def load_filtered_queries() -> pd.DataFrame:
    """Load the four brand-constrained queries."""
    queries = _read_csv(FILTERED_QUERIES_PATH)
    _require_columns(
        queries,
        (
            "workload_id",
            "query_text",
            "filter_field",
            "filter_operator",
            "filter_value",
        ),
        name="consultas_filtradas",
    )
    unsupported = queries[
        (queries["filter_field"] != "brand") | (queries["filter_operator"] != "equals")
    ]
    if not unsupported.empty:
        raise ValueError(
            "Solo se soporta el filtro brand/equals definido por la actividad; "
            f"filas inesperadas: {unsupported['workload_id'].tolist()}."
        )
    return queries


def load_catalog_events() -> pd.DataFrame:
    """Load the ordered catalog events (upserts and deletions)."""
    events = _read_csv(CATALOG_EVENTS_PATH)
    _require_columns(
        events,
        ("sequence", "event_id", "operation", *CATALOG_COLUMNS),
        name="eventos_catalogo",
    )
    events = events.copy()
    events["sequence"] = events["sequence"].astype(int)
    events["catalog_version"] = events["catalog_version"].astype(int)
    events["active"] = _parse_bool_column(events["active"], column="active")
    events = events.sort_values("sequence", kind="stable").reset_index(drop=True)
    unexpected = sorted(set(events["operation"]) - {"UPSERT", "DELETE"})
    if unexpected:
        raise ValueError(f"Operaciones de evento desconocidas: {unexpected}.")
    if events["sequence"].tolist() != list(range(1, len(events) + 1)):
        raise ValueError("La columna sequence debe ser 1..N sin huecos.")
    _validate_record_ids(events, name="eventos_catalogo")
    return events


def load_incoming_products(*, labeled: bool) -> pd.DataFrame:
    """Load incoming records: labeled development cases or blind evaluation ones."""
    path = INCOMING_DEVELOPMENT_PATH if labeled else INCOMING_EVALUATION_PATH
    incoming = _read_csv(path)
    expected = ("incoming_id", "title", "brand", "color", "text")
    if labeled:
        expected += ("is_duplicate", "reference_product_id")
    _require_columns(incoming, expected, name=path.name)
    if len(incoming) != 14:
        raise ValueError(f"{path.name} debe tener 14 altas, no {len(incoming)}.")
    if labeled:
        incoming = incoming.copy()
        incoming["is_duplicate"] = _parse_bool_column(
            incoming["is_duplicate"], column="is_duplicate"
        )
    return incoming


def iter_record_batches(
    catalog: pd.DataFrame,
    embeddings: NDArray[np.float32],
    *,
    batch_size: int = 256,
) -> Iterator[list[VectorRecord]]:
    """Yield deterministic batches of records aligned row-by-row with embeddings."""
    if batch_size < 1:
        raise ValueError("batch_size debe ser positivo.")
    if len(catalog) != embeddings.shape[0]:
        raise ValueError(
            "El catálogo y la matriz de embeddings no están alineados: "
            f"{len(catalog)} filas frente a {embeddings.shape[0]} vectores."
        )
    for start in range(0, len(catalog), batch_size):
        chunk = catalog.iloc[start : start + batch_size]
        yield [
            build_vector_record(row, embeddings[position])
            for position, (_, row) in zip(
                range(start, start + len(chunk)), chunk.iterrows(), strict=True
            )
        ]


def build_vector_record(row: pd.Series, embedding: NDArray[np.float32]) -> VectorRecord:
    """Build the canonical ingest unit for one catalog row."""
    return VectorRecord(
        record_id=row["record_id"],
        product_id=row["product_id"],
        title=row["title"],
        brand=row["brand"],
        color=row["color"],
        locale=row["locale"],
        catalog_version=int(row["catalog_version"]),
        active=bool(row["active"]),
        text=row["text"],
        embedding=np.asarray(embedding, dtype=np.float32).tolist(),
    )


def _require_columns(
    frame: pd.DataFrame, expected: tuple[str, ...], *, name: str
) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {name}: {missing}.")


def _validate_record_ids(frame: pd.DataFrame, *, name: str) -> None:
    """Enforce the UUIDv5 contract between product_id and record_id."""
    for _, row in frame.iterrows():
        expected = record_id_for_product(row["product_id"])
        if row["record_id"] != expected:
            raise ValueError(
                f"record_id inconsistente en {name} para {row['product_id']!r}: "
                f"esperado {expected}, encontrado {row['record_id']!r}."
            )


def _validate_catalog(catalog: pd.DataFrame, *, expected_rows: int) -> None:
    _require_columns(catalog, CATALOG_COLUMNS, name="catálogo")
    if len(catalog) != expected_rows:
        raise ValueError(
            f"El catálogo debe tener {expected_rows} filas, no {len(catalog)}."
        )
    if catalog["record_id"].duplicated().any():
        raise ValueError("El catálogo contiene record_id duplicados.")
    if catalog["product_id"].duplicated().any():
        raise ValueError("El catálogo contiene product_id duplicados.")
    _validate_record_ids(catalog.head(100), name="catálogo")


__all__ = [
    "CATALOG_COLUMNS",
    "CATALOG_EVENTS_PATH",
    "CATALOG_PATH",
    "CATALOG_SAMPLE_PATH",
    "RECORD_NAMESPACE",
    "build_vector_record",
    "iter_record_batches",
    "load_catalog",
    "load_catalog_events",
    "load_development_judgments",
    "load_development_queries",
    "load_evaluation_queries",
    "load_filtered_queries",
    "load_incoming_products",
    "load_manifest",
    "record_id_for_product",
]
