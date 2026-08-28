"""Validated loading of the self-contained marketplace benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "esci"


@dataclass(frozen=True, slots=True)
class SessionData:
    """Aligned metadata, queries, judgments, and normalized vectors."""

    products: pd.DataFrame
    queries: pd.DataFrame
    judgments: pd.DataFrame
    semantic_queries: pd.DataFrame
    product_embeddings: NDArray[np.float32]
    query_embeddings: NDArray[np.float32]
    manifest: dict[str, object]
    embedding_metadata: dict[str, object]


def load_session_data(*, memory_map: bool = True) -> SessionData:
    """Load every session artifact and validate cross-file alignment."""
    
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


def _validate_alignment(
    products: pd.DataFrame,
    queries: pd.DataFrame,
    product_embeddings: NDArray[np.float32],
    query_embeddings: NDArray[np.float32],
) -> None:
    """Reject silent ID/vector misalignment before any index is built."""
    
    expected_ids = np.arange(len(products), dtype=np.int64)
    np.testing.assert_array_equal(products["vector_id"].to_numpy(), expected_ids)
    
    if product_embeddings.shape[0] != len(products):
        raise ValueError("Product metadata and embedding rows are misaligned")
    if query_embeddings.shape[0] != len(queries):
        raise ValueError("Query metadata and embedding rows are misaligned")
    if product_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("Embedding artifacts must be two-dimensional")
    if product_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("Product and query dimensions do not match")
    
    product_norms = np.linalg.norm(product_embeddings, axis=1)
    query_norms = np.linalg.norm(query_embeddings, axis=1)
    
    if not np.allclose(product_norms, 1.0, atol=1e-4):
        raise ValueError("Product embeddings are not L2-normalized")
    if not np.allclose(query_norms, 1.0, atol=1e-4):
        raise ValueError("Query embeddings are not L2-normalized")
