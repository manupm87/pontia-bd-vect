"""Validated loading and deterministic records for the ESCI snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import VectorRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "esci"
RECORD_NAMESPACE = uuid.UUID("3b4da450-802c-5dd6-80df-9bb588e5a4ac")


@dataclass(frozen=True, slots=True)
class SessionData:
    """Aligned catalog, workload, judgments, vectors, and provenance."""

    products: pd.DataFrame
    queries: pd.DataFrame
    judgments: pd.DataFrame
    semantic_queries: pd.DataFrame
    product_embeddings: NDArray[np.float32]
    query_embeddings: NDArray[np.float32]
    manifest: dict[str, object]
    embedding_metadata: dict[str, object]

    def query_row(self, workload_id: str) -> pd.Series:
        """Resolve one workload row and reject missing or duplicate IDs."""
        matches = self.queries.loc[self.queries["workload_id"] == workload_id]
        if len(matches) != 1:
            raise KeyError(f"Expected one query {workload_id!r}, found {len(matches)}")
        return matches.iloc[0]

    def query_vector(self, workload_id: str) -> NDArray[np.float32]:
        """Return the vector aligned with a workload ID."""
        matching_positions = np.flatnonzero(
            self.queries["workload_id"].to_numpy() == workload_id
        )
        if len(matching_positions) != 1:
            raise KeyError(f"Expected one query vector for {workload_id!r}")
        return np.asarray(
            self.query_embeddings[matching_positions[0]], dtype=np.float32
        )


def record_id_for_product(product_id: str) -> str:
    """Create the stable UUIDv5 used by every database."""
    resolved = str(product_id).strip()
    if not resolved:
        raise ValueError("product_id cannot be blank")
    return str(uuid.uuid5(RECORD_NAMESPACE, resolved))


def load_session_data(
    *, memory_map: bool = True, verify_checksums: bool = False
) -> SessionData:
    """Load the self-contained snapshot and reject silent misalignment."""
    products = pd.read_csv(DATA_DIRECTORY / "products.csv.gz")
    queries = pd.read_csv(DATA_DIRECTORY / "query_workload.csv")
    judgments = pd.read_csv(DATA_DIRECTORY / "judgments.csv")
    semantic_queries = pd.read_csv(DATA_DIRECTORY / "semantic_queries.csv")
    mapping_mode = "r" if memory_map else None
    product_embeddings = np.load(
        DATA_DIRECTORY / "product_embeddings.npy",
        mmap_mode=mapping_mode,
        allow_pickle=False,
    )
    query_embeddings = np.load(
        DATA_DIRECTORY / "query_embeddings.npy",
        mmap_mode=mapping_mode,
        allow_pickle=False,
    )
    manifest = json.loads(
        (DATA_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    )
    embedding_metadata = json.loads(
        (DATA_DIRECTORY / "embedding_metadata.json").read_text(encoding="utf-8")
    )
    if verify_checksums:
        _verify_checksums(manifest, embedding_metadata)
    _validate_alignment(products, queries, product_embeddings, query_embeddings)
    return SessionData(
        products=products,
        queries=queries,
        judgments=judgments,
        semantic_queries=semantic_queries,
        product_embeddings=product_embeddings,
        query_embeddings=query_embeddings,
        manifest=manifest,
        embedding_metadata=embedding_metadata,
    )


def iter_record_batches(
    data: SessionData, *, batch_size: int = 500
) -> Iterator[list[VectorRecord]]:
    """Yield deterministic, provider-neutral product batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(data.products), batch_size):
        stop = min(start + batch_size, len(data.products))
        records = []
        for position in range(start, stop):
            row = data.products.iloc[position]
            product_id = _clean_scalar(row["product_id"])
            records.append(
                VectorRecord(
                    record_id=record_id_for_product(product_id),
                    product_id=product_id,
                    vector_id=int(row["vector_id"]),
                    title=_clean_scalar(row["product_title"]),
                    brand=_clean_scalar(row["product_brand"]),
                    color=_clean_scalar(row["product_color"]),
                    locale="es",
                    text=_clean_scalar(row["searchable_text"]),
                    embedding=np.asarray(
                        data.product_embeddings[position], dtype=np.float32
                    ).tolist(),
                )
            )
        yield records


def _clean_scalar(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _validate_alignment(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    product_embeddings: NDArray[np.float32],
    query_embeddings: NDArray[np.float32],
) -> None:
    required_columns = {
        "vector_id",
        "product_id",
        "product_title",
        "product_brand",
        "product_color",
        "searchable_text",
    }
    missing = required_columns.difference(products.columns)
    if missing:
        raise ValueError(f"Missing product columns: {sorted(missing)}")
    if len(products) != 50_000 or len(queries) != 276:
        raise ValueError("Unexpected snapshot counts")
    expected_ids = np.arange(len(products), dtype=np.int64)
    np.testing.assert_array_equal(products["vector_id"].to_numpy(), expected_ids)
    if product_embeddings.shape != (50_000, 384):
        raise ValueError(f"Unexpected product shape: {product_embeddings.shape}")
    if query_embeddings.shape != (276, 384):
        raise ValueError(f"Unexpected query shape: {query_embeddings.shape}")
    if product_embeddings.dtype != np.float32 or query_embeddings.dtype != np.float32:
        raise ValueError("Embeddings must use float32")
    if not np.allclose(np.linalg.norm(product_embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Product embeddings are not L2-normalized")
    if not np.allclose(np.linalg.norm(query_embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Query embeddings are not L2-normalized")
    if products["product_id"].astype(str).duplicated().any():
        raise ValueError("product_id values must be unique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(
    manifest: dict[str, object], embedding_metadata: dict[str, object]
) -> None:
    declared = dict(manifest["files"]) | dict(embedding_metadata["files"])
    for filename, expected in declared.items():
        actual = _sha256(DATA_DIRECTORY / filename)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {filename}: {actual}")
