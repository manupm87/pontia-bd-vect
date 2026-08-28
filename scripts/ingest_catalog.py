"""Idempotent batch ingestion of the full catalog into the Qdrant collection."""

from __future__ import annotations

import argparse
import os
from time import perf_counter

from dotenv import load_dotenv

from aurum_discovery.config import ARTIFACTS_DIRECTORY, PROJECT_ROOT, load_run_config
from aurum_discovery.data import iter_record_batches, load_catalog
from aurum_discovery.embeddings import SET_PRODUCTS, load_embedding_set
from aurum_discovery.operations import write_json_artifact
from aurum_discovery.vector_store import CatalogVectorStore


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=None,
        help="Nombre de la colección; por defecto QDRANT_COLLECTION del .env.",
    )
    return parser.parse_args()


def build_store(collection_name: str) -> CatalogVectorStore:
    config = load_run_config()
    configuration = load_embedding_set(config.embedding_configuration).configuration
    return CatalogVectorStore(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", ""),
        collection_name=collection_name,
        vector_size=configuration.dimension,
        hnsw=config.hnsw,
        ef_search=config.ef_search,
    )


def main() -> None:
    """Ingest the catalog and verify count and index status before accepting queries."""
    load_dotenv(PROJECT_ROOT / ".env")
    arguments = parse_arguments()
    config = load_run_config()
    collection_name = arguments.collection or os.getenv(
        "QDRANT_COLLECTION", "aurum-market-eval-catalogo"
    )
    allow_reset = os.getenv("AURUM_ALLOW_RESET", "false").lower() == "true"

    embedding_set = load_embedding_set(config.embedding_configuration)
    catalog = load_catalog()
    stored_ids = embedding_set.identifiers[SET_PRODUCTS]
    if stored_ids != catalog["record_id"].tolist():
        raise ValueError(
            "Los embeddings persistidos no están alineados con el catálogo; "
            "regenera con `make embeddings`."
        )

    store = build_store(collection_name)
    store.ensure_collection(allow_reset=allow_reset)
    count_before = store.count()

    started_at = perf_counter()
    sent = store.upsert_records(
        iter_record_batches(
            catalog,
            embedding_set.matrix(SET_PRODUCTS),
            batch_size=config.batch_size,
        )
    )
    ingestion_seconds = perf_counter() - started_at
    index_state = store.wait_until_indexed()
    count_after = store.count()

    if count_after != len(catalog):
        raise RuntimeError(
            f"Recuento inesperado tras la ingesta: {count_after} registros "
            f"frente a {len(catalog)} esperados."
        )

    report = {
        "collection": collection_name,
        "embedding_configuration": config.embedding_configuration,
        "batch_size": config.batch_size,
        "records_sent": sent,
        "count_before": count_before,
        "count_after": count_after,
        "idempotent": count_before in (0, len(catalog)) and count_after == len(catalog),
        "index_state": index_state,
        "ingestion_seconds": round(ingestion_seconds, 2),
    }
    path = write_json_artifact(
        report, ARTIFACTS_DIRECTORY / "ingesta" / "informe_ingesta.json"
    )
    print(
        f"OK ingesta: {count_after} registros, estado {index_state['status']} con "
        f"{index_state['indexed_vectors_count']} vectores indexados, "
        f"{ingestion_seconds:.1f}s"
    )
    print(f"Informe escrito en {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
