"""Encode the session catalog and ANN query workload with multilingual E5."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "esci"


def sha256_file(file_path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Generate normalized product and query matrices plus metadata."""
    load_dotenv(PROJECT_ROOT / ".env")
    model_id = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    products_path = DATA_DIRECTORY / "products.csv.gz"
    queries_path = DATA_DIRECTORY / "query_workload.csv"
    products = pd.read_csv(products_path)
    queries = pd.read_csv(queries_path)
    model = SentenceTransformer(model_id)

    product_inputs = [
        f"passage: {text}" for text in products["searchable_text"].fillna("")
    ]
    query_inputs = [f"query: {text}" for text in queries["query_text"]]
    product_embeddings = model.encode(
        product_inputs,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    query_embeddings = model.encode(
        query_inputs,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    products_output = DATA_DIRECTORY / "product_embeddings.npy"
    queries_output = DATA_DIRECTORY / "query_embeddings.npy"
    np.save(products_output, product_embeddings, allow_pickle=False)
    np.save(queries_output, query_embeddings, allow_pickle=False)
    metadata = {
        "model_id": model_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "dimension": int(product_embeddings.shape[1]),
        "dtype": str(product_embeddings.dtype),
        "normalized": True,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "product_count": len(products),
        "query_count": len(queries),
        "source_sha256": {
            products_path.name: sha256_file(products_path),
            queries_path.name: sha256_file(queries_path),
        },
        "files": {
            products_output.name: sha256_file(products_output),
            queries_output.name: sha256_file(queries_output),
        },
    }
    (DATA_DIRECTORY / "embedding_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Generated product matrix {product_embeddings.shape} and "
        f"query matrix {query_embeddings.shape}"
    )


if __name__ == "__main__":
    main()
