"""Command-line interface: one query in, normalized results out as JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from ingest_catalog import build_store

from aurum_discovery.config import PROJECT_ROOT, load_run_config
from aurum_discovery.embeddings import get_configuration
from aurum_discovery.service import DiscoveryService
from aurum_discovery.vector_store import (
    EmptyCollectionError,
    VectorStoreUnavailableError,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Texto de la consulta.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--brand", default=None, help="Restringe los resultados a una marca exacta."
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Usa búsqueda exhaustiva (oráculo) en lugar del índice HNSW.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one semantic search and print the normalized result contract."""
    load_dotenv(PROJECT_ROOT / ".env")
    arguments = parse_arguments()
    config = load_run_config()
    service = DiscoveryService(
        store=build_store(os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo")),
        configuration=get_configuration(config.embedding_configuration),
    )
    try:
        hits = service.search_text(
            arguments.query,
            top_k=arguments.top_k,
            brand=arguments.brand,
            exact=arguments.exact,
        )
    except VectorStoreUnavailableError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except EmptyCollectionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(3) from error
    if not hits:
        print(
            json.dumps(
                {
                    "query": arguments.query,
                    "brand": arguments.brand,
                    "results": [],
                    "note": "El filtro no devolvió resultados.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            {
                "query": arguments.query,
                "brand": arguments.brand,
                "results": [hit.as_dict() for hit in hits],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
