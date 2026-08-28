"""Build every embedding set required by the activity, with provenance metadata."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
from dotenv import load_dotenv

from aurum_discovery.config import PROJECT_ROOT
from aurum_discovery.data import (
    CATALOG_EVENTS_PATH,
    CATALOG_PATH,
    DEVELOPMENT_QUERIES_PATH,
    EVALUATION_QUERIES_PATH,
    FILTERED_QUERIES_PATH,
    INCOMING_DEVELOPMENT_PATH,
    INCOMING_EVALUATION_PATH,
    load_catalog,
    load_catalog_events,
    load_development_queries,
    load_evaluation_queries,
    load_filtered_queries,
    load_incoming_products,
)
from aurum_discovery.embeddings import (
    EMBEDDING_CONFIGURATIONS,
    SET_DEVELOPMENT_QUERIES,
    SET_EVALUATION_QUERIES,
    SET_EVENT_DOCUMENTS,
    SET_FILTERED_QUERIES,
    SET_INCOMING_DEVELOPMENT,
    SET_INCOMING_EVALUATION,
    SET_PRODUCTS,
    EmbeddingConfiguration,
    compose_document_text,
    encode_texts,
    load_encoder,
)
from aurum_discovery.operations import sha256_file

SOURCE_PATHS = {
    "catalogo_productos.csv.gz": CATALOG_PATH,
    "consultas_desarrollo.csv": DEVELOPMENT_QUERIES_PATH,
    "consultas_evaluacion.csv": EVALUATION_QUERIES_PATH,
    "consultas_filtradas.csv": FILTERED_QUERIES_PATH,
    "altas_desarrollo.csv": INCOMING_DEVELOPMENT_PATH,
    "altas_evaluacion.csv": INCOMING_EVALUATION_PATH,
    "eventos_catalogo.csv": CATALOG_EVENTS_PATH,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        action="append",
        choices=sorted(EMBEDDING_CONFIGURATIONS),
        help="Configuración a construir; repetible. Por defecto, todas.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera aunque ya existan embeddings persistidos.",
    )
    return parser.parse_args()


def build_configuration(
    configuration: EmbeddingConfiguration, *, batch_size: int, force: bool
) -> None:
    directory = configuration.directory
    metadata_path = directory / "embedding_metadata.json"
    if metadata_path.exists() and not force:
        print(
            f"OK {configuration.name}: ya existe, se omite (usa --force para regenerar)"
        )
        return
    catalog = load_catalog()
    events = load_catalog_events()
    upserts = events[events["operation"] == "UPSERT"]
    document_sets = {
        SET_PRODUCTS: (
            catalog["record_id"].tolist(),
            [
                compose_document_text(row, composition=configuration.composition)
                for _, row in catalog.iterrows()
            ],
        ),
        SET_INCOMING_DEVELOPMENT: _incoming_texts(configuration, labeled=True),
        SET_INCOMING_EVALUATION: _incoming_texts(configuration, labeled=False),
        SET_EVENT_DOCUMENTS: (
            upserts["event_id"].tolist(),
            [
                compose_document_text(row, composition=configuration.composition)
                for _, row in upserts.iterrows()
            ],
        ),
    }
    query_sets = {
        SET_DEVELOPMENT_QUERIES: (
            load_development_queries()["workload_id"].tolist(),
            load_development_queries()["query_text"].tolist(),
        ),
        SET_EVALUATION_QUERIES: (
            load_evaluation_queries()["evaluation_id"].tolist(),
            load_evaluation_queries()["query_text"].tolist(),
        ),
        SET_FILTERED_QUERIES: (
            load_filtered_queries()["workload_id"].tolist(),
            load_filtered_queries()["query_text"].tolist(),
        ),
    }
    encoder = load_encoder(configuration.model_id)
    directory.mkdir(parents=True, exist_ok=True)
    identifiers: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    started_at = datetime.now(UTC)
    for set_name, (set_identifiers, texts) in {**document_sets, **query_sets}.items():
        prefix = (
            configuration.document_prefix
            if set_name in document_sets
            else configuration.query_prefix
        )
        matrix = encode_texts(
            encoder,
            texts,
            prefix=prefix,
            normalize=configuration.normalize,
            batch_size=batch_size,
        )
        np.save(directory / f"{set_name}.npy", matrix, allow_pickle=False)
        identifiers[set_name] = list(set_identifiers)
        counts[set_name] = len(texts)
        print(f"  {configuration.name}/{set_name}: {matrix.shape}")
    metadata = {
        "model_id": configuration.model_id,
        "configuration": configuration.name,
        "composition": configuration.composition,
        "generated_at": started_at.isoformat(),
        "dimension": configuration.dimension,
        "dtype": "float32",
        "normalized": configuration.normalize,
        "document_prefix": configuration.document_prefix,
        "query_prefix": configuration.query_prefix,
        "counts": counts,
        "identifiers": identifiers,
        "source_sha256": {
            name: sha256_file(path) for name, path in SOURCE_PATHS.items()
        },
        "files": {
            f"{set_name}.npy": sha256_file(directory / f"{set_name}.npy")
            for set_name in identifiers
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    elapsed = (datetime.now(UTC) - started_at).total_seconds()
    print(
        f"OK {configuration.name}: {elapsed:.1f}s -> {directory.relative_to(PROJECT_ROOT)}"
    )


def _incoming_texts(
    configuration: EmbeddingConfiguration, *, labeled: bool
) -> tuple[list[str], list[str]]:
    incoming = load_incoming_products(labeled=labeled)
    return (
        incoming["incoming_id"].tolist(),
        [
            compose_document_text(row, composition=configuration.composition)
            for _, row in incoming.iterrows()
        ],
    )


def main() -> None:
    """Build the embedding sets for every requested configuration."""
    load_dotenv(PROJECT_ROOT / ".env")
    arguments = parse_arguments()
    names = arguments.configuration or sorted(EMBEDDING_CONFIGURATIONS)
    for name in names:
        build_configuration(
            EMBEDDING_CONFIGURATIONS[name],
            batch_size=arguments.batch_size,
            force=arguments.force,
        )


if __name__ == "__main__":
    main()
