"""Apply the 24 ordered catalog events with idempotency and visibility checks.

The events run against a dedicated collection seeded from the full catalog,
so the main collection keeps serving the reproducible rankings delivered in
resultados/. Both collections share schema, configuration, and ingest path.
"""

from __future__ import annotations

import os
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from ingest_catalog import build_store

from aurum_discovery.config import ARTIFACTS_DIRECTORY, PROJECT_ROOT, load_run_config
from aurum_discovery.data import (
    build_vector_record,
    iter_record_batches,
    load_catalog,
    load_catalog_events,
)
from aurum_discovery.embeddings import (
    SET_EVENT_DOCUMENTS,
    SET_PRODUCTS,
    EmbeddingSet,
    load_embedding_set,
)
from aurum_discovery.operations import wait_until, write_json_artifact
from aurum_discovery.vector_store import CatalogVectorStore


def seed_collection(
    store: CatalogVectorStore,
    catalog: pd.DataFrame,
    embedding_set: EmbeddingSet,
    *,
    batch_size: int,
) -> dict[str, object]:
    """Ensure the events collection starts from the pristine catalog."""
    store.ensure_collection(
        allow_reset=os.getenv("AURUM_ALLOW_RESET", "false").lower() == "true"
    )
    count_before = store.count()
    seeded = False
    if count_before != len(catalog):
        store.upsert_records(
            iter_record_batches(
                catalog, embedding_set.matrix(SET_PRODUCTS), batch_size=batch_size
            )
        )
        store.wait_until_indexed()
        seeded = True
    return {
        "count_before": count_before,
        "seeded": seeded,
        "count_after_seed": store.count(),
    }


def apply_event_pass(
    store: CatalogVectorStore,
    events: pd.DataFrame,
    embedding_set: EmbeddingSet,
) -> list[dict[str, object]]:
    """Apply every event in sequence order and record what was done."""
    event_ids = embedding_set.identifiers[SET_EVENT_DOCUMENTS]
    event_matrix = embedding_set.matrix(SET_EVENT_DOCUMENTS)
    outcomes = []
    for _, event in events.iterrows():
        started_at = perf_counter()
        if event["operation"] == "UPSERT":
            position = event_ids.index(event["event_id"])
            record = build_vector_record(event, event_matrix[position])
            store.upsert_records([[record]])
        else:
            store.delete_records([event["record_id"]])
        outcomes.append(
            {
                "sequence": int(event["sequence"]),
                "event_id": event["event_id"],
                "operation": event["operation"],
                "record_id": event["record_id"],
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        )
    return outcomes


def check_visibility(
    store: CatalogVectorStore,
    events: pd.DataFrame,
    catalog: pd.DataFrame,
    embedding_set: EmbeddingSet,
) -> list[dict[str, object]]:
    """Verify one update, one insertion, and one deletion via ID and search."""
    catalog_ids = set(catalog["record_id"])
    upserts = events[events["operation"] == "UPSERT"]
    updates = upserts[upserts["record_id"].isin(catalog_ids)]
    insertions = upserts[~upserts["record_id"].isin(catalog_ids)]
    deletions = events[events["operation"] == "DELETE"]
    if updates.empty or insertions.empty or deletions.empty:
        raise ValueError(
            "Los eventos deben incluir al menos una actualización, un alta y "
            "una eliminación para medir la visibilidad."
        )
    event_ids = embedding_set.identifiers[SET_EVENT_DOCUMENTS]
    event_matrix = embedding_set.matrix(SET_EVENT_DOCUMENTS)
    product_ids = embedding_set.identifiers[SET_PRODUCTS]
    product_matrix = embedding_set.matrix(SET_PRODUCTS)
    checks = []

    update = updates.iloc[0]
    payload = store.retrieve(update["record_id"])
    if payload is None or int(str(payload["catalog_version"])) != int(
        update["catalog_version"]
    ):
        raise RuntimeError(
            f"La actualización {update['event_id']} no es visible por ID: {payload!r}."
        )
    update_vector = event_matrix[event_ids.index(update["event_id"])]
    _, elapsed, attempts = wait_until(
        lambda: [hit.record_id for hit in store.search(update_vector, top_k=3)],
        accept=lambda ids: update["record_id"] in ids,
        description=f"visibilidad en búsqueda de {update['event_id']}",
    )
    checks.append(
        {
            "kind": "actualizacion",
            "event_id": update["event_id"],
            "read_by_id": "version nueva visible",
            "search_elapsed_s": round(elapsed, 3),
            "search_attempts": attempts,
        }
    )

    insertion = insertions.iloc[0]
    payload = store.retrieve(insertion["record_id"])
    if payload is None:
        raise RuntimeError(f"El alta {insertion['event_id']} no es legible por ID.")
    insertion_vector = event_matrix[event_ids.index(insertion["event_id"])]
    _, elapsed, attempts = wait_until(
        lambda: [hit.record_id for hit in store.search(insertion_vector, top_k=3)],
        accept=lambda ids: insertion["record_id"] in ids,
        description=f"visibilidad en búsqueda de {insertion['event_id']}",
    )
    checks.append(
        {
            "kind": "alta",
            "event_id": insertion["event_id"],
            "read_by_id": "registro visible",
            "search_elapsed_s": round(elapsed, 3),
            "search_attempts": attempts,
        }
    )

    deletion = deletions.iloc[0]
    payload = store.retrieve(deletion["record_id"])
    if payload is not None:
        raise RuntimeError(
            f"La eliminación {deletion['event_id']} sigue siendo legible por ID."
        )
    deletion_vector = product_matrix[product_ids.index(deletion["record_id"])]
    _, elapsed, attempts = wait_until(
        lambda: [hit.record_id for hit in store.search(deletion_vector, top_k=10)],
        accept=lambda ids: deletion["record_id"] not in ids,
        description=f"ausencia en búsqueda de {deletion['event_id']}",
    )
    checks.append(
        {
            "kind": "eliminacion",
            "event_id": deletion["event_id"],
            "read_by_id": "registro no recuperable",
            "search_elapsed_s": round(elapsed, 3),
            "search_attempts": attempts,
        }
    )
    return checks


def main() -> None:
    """Seed, apply the events twice, and prove the final state is stable."""
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_run_config()
    catalog = load_catalog()
    events = load_catalog_events()
    embedding_set = load_embedding_set(config.embedding_configuration)
    collection_name = os.getenv("QDRANT_EVENTS_COLLECTION", "aurum-market-eval-eventos")
    store = build_store(collection_name)

    seed_report = seed_collection(
        store, catalog, embedding_set, batch_size=config.batch_size
    )

    catalog_ids = set(catalog["record_id"])
    final_ids = set(catalog_ids)
    for _, event in events.iterrows():
        if event["operation"] == "UPSERT":
            final_ids.add(event["record_id"])
        else:
            final_ids.discard(event["record_id"])
    expected_count = len(final_ids)

    first_pass = apply_event_pass(store, events, embedding_set)
    store.wait_until_indexed()
    count_after_first = store.count()
    second_pass = apply_event_pass(store, events, embedding_set)
    store.wait_until_indexed()
    count_after_second = store.count()

    if count_after_first != expected_count or count_after_second != expected_count:
        raise RuntimeError(
            f"Recuento inesperado tras los eventos: {count_after_first} y "
            f"{count_after_second} frente a {expected_count} esperados."
        )
    visibility = check_visibility(store, events, catalog, embedding_set)

    report = {
        "collection": collection_name,
        "seed": seed_report,
        "event_count": len(events),
        "expected_final_count": expected_count,
        "count_after_first_pass": count_after_first,
        "count_after_second_pass": count_after_second,
        "idempotent": count_after_first == count_after_second == expected_count,
        "first_pass": first_pass,
        "second_pass_elapsed_ms": [outcome["elapsed_ms"] for outcome in second_pass],
        "visibility": visibility,
    }
    path = write_json_artifact(
        report, ARTIFACTS_DIRECTORY / "eventos" / "informe_eventos.json"
    )
    print(
        f"OK eventos: {len(events)} operaciones x2 pasadas, recuento estable en "
        f"{count_after_second}"
    )
    for check in visibility:
        print(
            f"  {check['kind']}: lectura por ID ok, búsqueda en "
            f"{check['search_elapsed_s']}s ({check['search_attempts']} intentos)"
        )
    print(f"Informe escrito en {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
