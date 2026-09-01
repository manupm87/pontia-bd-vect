"""Live catalog operations: alta gated by the duplicate rule, update, delete.

This is the enunciado's "control de altas" as an interactive command: given a
new product record, retrieve the closest catalog product and decide whether a
duplicate must be reviewed before publishing. Updates and deletions verify
visibility by ID and by search; the duplicate gate applies only to new
records, because the catalog legitimately contains near-twin variants and an
edit to an existing identity should not be blocked by a pre-existing twin.

Everything runs against a sandbox collection seeded from the pristine
catalog, so the reproducible artifacts of the main and events collections
stay intact. Exit code 2 means "blocked by the duplicate rule"; use --force
to publish anyway.
"""

from __future__ import annotations

import argparse
import os
import sys
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from ingest_catalog import build_store

from aurum_discovery.config import PROJECT_ROOT, RunConfig, load_run_config
from aurum_discovery.contracts import SearchHit, VectorRecord
from aurum_discovery.data import (
    iter_record_batches,
    load_catalog,
    record_id_for_product,
)
from aurum_discovery.duplicates import DuplicateRule
from aurum_discovery.embeddings import (
    SET_PRODUCTS,
    EmbeddingConfiguration,
    compose_document_text,
    encode_texts,
    get_configuration,
    load_embedding_set,
    load_encoder,
)
from aurum_discovery.operations import wait_until
from aurum_discovery.vector_store import CatalogVectorStore

CANDIDATE_COUNT = 5
EXIT_BLOCKED_BY_DUPLICATE = 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    alta = subparsers.add_parser(
        "alta",
        help="Publica una ficha nueva, pasando antes por la regla de duplicados.",
    )
    alta.add_argument("--title", required=True, help="Título del producto.")
    alta.add_argument("--brand", default="", help="Marca (opcional).")
    alta.add_argument("--color", default="", help="Color (opcional).")
    alta.add_argument(
        "--product-id",
        default=None,
        help="ID de producto; por defecto se genera uno con prefijo CLI-.",
    )
    alta.add_argument(
        "--check-only",
        action="store_true",
        help="Solo muestra la decisión de duplicados, sin publicar.",
    )
    alta.add_argument(
        "--force",
        action="store_true",
        help="Publica aunque la regla marque duplicado.",
    )

    update = subparsers.add_parser(
        "update",
        help="Actualiza una ficha existente y verifica la visibilidad del cambio.",
    )
    update.add_argument(
        "--product-id", required=True, help="ID del producto a actualizar."
    )
    update.add_argument("--title", default=None, help="Nuevo título.")
    update.add_argument("--brand", default=None, help="Nueva marca.")
    update.add_argument("--color", default=None, help="Nuevo color.")

    baja = subparsers.add_parser(
        "baja", help="Elimina una ficha y verifica su ausencia por ID y por búsqueda."
    )
    baja.add_argument("--product-id", required=True, help="ID del producto a eliminar.")

    subparsers.add_parser(
        "estado", help="Recuento y estado de indexación de la colección sandbox."
    )
    return parser.parse_args()


def sandbox_store() -> CatalogVectorStore:
    collection = os.getenv("QDRANT_SANDBOX_COLLECTION", "aurum-market-eval-sandbox")
    return build_store(collection)


def ensure_seeded(store: CatalogVectorStore, config: RunConfig) -> None:
    """Seed the sandbox from the pristine catalog the first time it is used."""
    store.ensure_collection(
        allow_reset=os.getenv("AURUM_ALLOW_RESET", "false").lower() == "true"
    )
    if store.count() > 0:
        return
    print(f"Colección {store.collection_name!r} vacía: sembrando el catálogo…")
    embedding_set = load_embedding_set(config.embedding_configuration)
    catalog = load_catalog()
    store.upsert_records(
        iter_record_batches(
            catalog, embedding_set.matrix(SET_PRODUCTS), batch_size=config.batch_size
        )
    )
    state = store.wait_until_indexed()
    print(
        f"Sembrados {store.count()} registros "
        f"({state['indexed_vectors_count']} indexados)."
    )


def encode_document(
    configuration: EmbeddingConfiguration, title: str, brand: str, color: str
) -> tuple[str, list[float]]:
    """Compose and encode one record from its structured fields.

    A live intake only carries title/brand/color, so with a ``full_text``
    configuration this deliberately falls back to the title composition
    (``compose_document_text`` handles the empty ``text``); with the delivered
    ``title_brand_color`` configuration it matches the catalog documents
    exactly.
    """
    row = pd.Series({"title": title, "brand": brand, "color": color, "text": ""})
    composed = compose_document_text(row, composition=configuration.composition)
    encoder = load_encoder(configuration.model_id)
    matrix = encode_texts(
        encoder,
        [composed],
        prefix=configuration.document_prefix,
        normalize=configuration.normalize,
    )
    return composed, matrix[0].tolist()


def print_duplicate_verdict(
    rule: DuplicateRule, incoming_id: str, candidates: list[SearchHit]
) -> bool:
    """Show the candidates and the rule's decision; return True if duplicate."""
    decision = rule.decide(incoming_id, candidates)
    print(
        f"Regla de duplicados: score >= {rule.score_threshold}"
        + (f" y margen >= {rule.margin_threshold}" if rule.margin_threshold else "")
    )
    if not candidates:
        print("Sin candidatos: la colección no devolvió vecinos.")
        return False
    for hit in candidates[:2]:
        print(
            f"  candidato {hit.rank}: {hit.product_id} · score {hit.native_score:.4f}"
            f" · {hit.title[:70]}"
        )
    print(f"  margen sobre el segundo: {decision.margin:.4f}")
    if decision.predicted_duplicate:
        print(
            f"DECISIÓN: DUPLICADO de {decision.matched_product_id} "
            f"(score {decision.score:.4f})"
        )
    else:
        print(f"DECISIÓN: no duplicado (mejor score {decision.score:.4f})")
    return decision.predicted_duplicate


def verify_visible(
    store: CatalogVectorStore, record_id: str, vector: list[float]
) -> None:
    payload = store.retrieve(record_id)
    if payload is None:
        raise RuntimeError(f"El registro {record_id} no es legible tras escribirse.")
    _, elapsed, attempts = wait_until(
        lambda: [hit.record_id for hit in store.search(vector, top_k=3)],
        accept=lambda ids: record_id in ids,
        description=f"visibilidad en búsqueda de {record_id}",
    )
    print(
        f"Visibilidad: lectura por ID ok, en búsqueda en {elapsed:.3f}s "
        f"({attempts} intentos) · versión {payload['catalog_version']}"
    )


def run_alta(arguments: argparse.Namespace, config: RunConfig) -> int:
    store = sandbox_store()
    ensure_seeded(store, config)
    configuration = get_configuration(config.embedding_configuration)
    product_id = arguments.product_id or f"CLI-{uuid4().hex[:10].upper()}"
    composed, vector = encode_document(
        configuration, arguments.title, arguments.brand, arguments.color
    )
    print(f"Alta {product_id}: «{composed}»")
    rule = DuplicateRule(
        score_threshold=config.duplicate_score_threshold,
        margin_threshold=config.duplicate_margin_threshold,
    )
    candidates = store.search(vector, top_k=CANDIDATE_COUNT)
    is_duplicate = print_duplicate_verdict(rule, product_id, candidates)
    if arguments.check_only:
        print("(--check-only: no se publica nada)")
        return 0
    if is_duplicate and not arguments.force:
        print(
            "Alta BLOQUEADA para revisión humana. Publica de todos modos con --force."
        )
        return EXIT_BLOCKED_BY_DUPLICATE
    record = VectorRecord(
        record_id=record_id_for_product(product_id),
        product_id=product_id,
        title=arguments.title.strip(),
        brand=arguments.brand.strip(),
        color=arguments.color.strip(),
        locale="es",
        catalog_version=1,
        active=True,
        text=composed,
        embedding=vector,
    )
    count_before = store.count()
    store.upsert_records([[record]])
    print(f"Publicado: recuento {count_before} -> {store.count()}")
    verify_visible(store, record.record_id, vector)
    return 0


def run_update(arguments: argparse.Namespace, config: RunConfig) -> int:
    if all(
        value is None for value in (arguments.title, arguments.brand, arguments.color)
    ):
        print("Nada que actualizar: indica --title, --brand o --color.")
        return 1
    store = sandbox_store()
    ensure_seeded(store, config)
    configuration = get_configuration(config.embedding_configuration)
    record_id = record_id_for_product(arguments.product_id)
    payload = store.retrieve(record_id)
    if payload is None:
        print(f"El producto {arguments.product_id} no existe en la colección sandbox.")
        return 1
    title = arguments.title if arguments.title is not None else str(payload["title"])
    brand = arguments.brand if arguments.brand is not None else str(payload["brand"])
    color = arguments.color if arguments.color is not None else str(payload["color"])
    composed, vector = encode_document(configuration, title, brand, color)
    next_version = int(str(payload["catalog_version"])) + 1
    print(f"Actualización {arguments.product_id} (v{next_version}): «{composed}»")
    record = VectorRecord(
        record_id=record_id,
        product_id=arguments.product_id,
        title=title.strip(),
        brand=brand.strip(),
        color=color.strip(),
        locale=str(payload["locale"]),
        catalog_version=next_version,
        active=bool(payload["active"]),
        text=composed,
        embedding=vector,
    )
    store.upsert_records([[record]])
    print(
        f"Actualizado: misma clave, versión {payload['catalog_version']} -> {next_version}"
    )
    verify_visible(store, record_id, vector)
    return 0


def run_baja(arguments: argparse.Namespace, config: RunConfig) -> int:
    store = sandbox_store()
    ensure_seeded(store, config)
    configuration = get_configuration(config.embedding_configuration)
    record_id = record_id_for_product(arguments.product_id)
    payload = store.retrieve(record_id)
    if payload is None:
        print(f"El producto {arguments.product_id} no existe en la colección sandbox.")
        return 1
    print(f"Baja {arguments.product_id}: «{payload['title']}»")
    _, vector = encode_document(
        configuration,
        str(payload["title"]),
        str(payload["brand"]),
        str(payload["color"]),
    )
    count_before = store.count()
    store.delete_records([record_id])
    if store.retrieve(record_id) is not None:
        raise RuntimeError(
            f"El registro {record_id} sigue siendo legible tras la baja."
        )
    _, elapsed, attempts = wait_until(
        lambda: [hit.record_id for hit in store.search(vector, top_k=10)],
        accept=lambda ids: record_id not in ids,
        description=f"ausencia en búsqueda de {record_id}",
    )
    print(
        f"Eliminado: recuento {count_before} -> {store.count()} · no legible por ID "
        f"· ausente de la búsqueda en {elapsed:.3f}s ({attempts} intentos)"
    )
    return 0


def run_estado() -> int:
    store = sandbox_store()
    store.ping()
    store.ensure_collection(allow_reset=False)
    info = store.collection_info()
    count = store.count()
    print(f"Colección sandbox: {store.collection_name}")
    print(f"Registros: {count}")
    if count == 0:
        print("Vacía: el primer alta/update/baja la sembrará desde el catálogo.")
        return 0
    state = store.wait_until_indexed()
    print(
        f"Indexación: {state['indexed_vectors_count']} vectores en HNSW "
        f"(m={info.config.hnsw_config.m}, ef_construct={info.config.hnsw_config.ef_construct})"
    )
    return 0


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    arguments = parse_arguments()
    config = load_run_config()
    if arguments.command == "alta":
        code = run_alta(arguments, config)
    elif arguments.command == "update":
        code = run_update(arguments, config)
    elif arguments.command == "baja":
        code = run_baja(arguments, config)
    else:
        code = run_estado()
    sys.exit(code)


if __name__ == "__main__":
    main()
