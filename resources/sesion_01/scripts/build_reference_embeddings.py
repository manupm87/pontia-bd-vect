"""Generate reproducible E5 embeddings for the compact ESCI snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from vector_search_session.ecommerce import (
    JUDGMENTS_PATH,
    PRODUCTS_PATH,
    load_esci_sample,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "esci"
SEMANTIC_QUERIES_PATH = OUTPUT_DIRECTORY / "semantic_queries.csv"


def sha256_file(file_path: Path) -> str:
    """Return the SHA-256 digest of one source file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for file_chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(file_chunk)
    return digest.hexdigest()


def main() -> None:
    """Encode products and unique queries with the configured E5 model."""
    load_dotenv(PROJECT_ROOT / ".env")
    model_id = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    sample = load_esci_sample()
    unique_queries = (
        sample.judgments[["query_id", "query"]]
        .drop_duplicates()
        .sort_values("query_id", ignore_index=True)
    )
    semantic_queries = pd.read_csv(SEMANTIC_QUERIES_PATH)

    product_inputs = [
        f"passage: {product_text}"
        for product_text in sample.products["searchable_text"]
    ]
    query_inputs = [f"query: {query_text}" for query_text in unique_queries["query"]]
    semantic_query_inputs = [
        f"query: {query_text}" for query_text in semantic_queries["semantic_query"]
    ]

    model = SentenceTransformer(model_id)
    product_embeddings = model.encode(
        product_inputs,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    query_embeddings = model.encode(
        query_inputs,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    semantic_query_embeddings = model.encode(
        semantic_query_inputs,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    products_output = OUTPUT_DIRECTORY / "e5_products.npz"
    queries_output = OUTPUT_DIRECTORY / "e5_queries.npz"
    semantic_queries_output = OUTPUT_DIRECTORY / "e5_semantic_queries.npz"
    metadata_output = OUTPUT_DIRECTORY / "e5_metadata.json"

    np.savez_compressed(
        products_output,
        product_ids=sample.products["product_id"].to_numpy(dtype=str),
        embeddings=product_embeddings,
    )
    np.savez_compressed(
        semantic_queries_output,
        query_ids=semantic_queries["query_id"].to_numpy(dtype=np.int64),
        embeddings=semantic_query_embeddings,
    )
    np.savez_compressed(
        queries_output,
        query_ids=unique_queries["query_id"].to_numpy(dtype=np.int64),
        embeddings=query_embeddings,
    )

    metadata = {
        "model_id": model_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "dimension": int(product_embeddings.shape[1]),
        "dtype": str(product_embeddings.dtype),
        "normalized": True,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "products_sha256": sha256_file(PRODUCTS_PATH),
        "judgments_sha256": sha256_file(JUDGMENTS_PATH),
        "product_count": len(sample.products),
        "query_count": len(unique_queries),
        "semantic_query_count": len(semantic_queries),
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Generated {product_embeddings.shape} product embeddings and "
        f"{query_embeddings.shape} original query embeddings and "
        f"{semantic_query_embeddings.shape} semantic query embeddings"
    )


if __name__ == "__main__":
    main()
