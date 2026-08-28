"""Integration tests against a live Qdrant: schema, ingest, filters, mutations."""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from aurum_discovery.config import HnswSettings
from aurum_discovery.contracts import VectorRecord
from aurum_discovery.vector_store import CatalogVectorStore, EmptyCollectionError

pytestmark = pytest.mark.integration

TEST_COLLECTION = "aurum-market-eval-test"
DIMENSION = 8


def build_record(product_id: str, vector: list[float], *, brand: str) -> VectorRecord:
    return VectorRecord(
        record_id=str(uuid.uuid5(uuid.NAMESPACE_URL, product_id)),
        product_id=product_id,
        title=f"Producto {product_id}",
        brand=brand,
        color="",
        locale="es",
        catalog_version=1,
        active=True,
        text=f"Producto {product_id}",
        embedding=vector,
    )


def normalized(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    return (array / np.linalg.norm(array)).tolist()


@pytest.fixture()
def store() -> CatalogVectorStore:
    instance = CatalogVectorStore(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=TEST_COLLECTION,
        vector_size=DIMENSION,
        hnsw=HnswSettings(m=8, ef_construct=32),
        ef_search=32,
    )
    instance.ensure_collection(allow_reset=True)
    yield instance
    instance.delete_collection(confirmation=f"DELETE:{TEST_COLLECTION}")


def seed(store: CatalogVectorStore) -> list[VectorRecord]:
    base = np.eye(DIMENSION, dtype=np.float32)
    records = [
        build_record(
            f"P{index}",
            normalized(base[index % DIMENSION].tolist()),
            brand="NIKE" if index % 2 == 0 else "Einhell",
        )
        for index in range(6)
    ]
    store.upsert_records([records])
    return records


def test_ingest_is_idempotent(store: CatalogVectorStore) -> None:
    records = seed(store)
    assert store.count() == len(records)
    store.upsert_records([records])
    assert store.count() == len(records)


def test_search_on_empty_collection_raises(store: CatalogVectorStore) -> None:
    with pytest.raises(EmptyCollectionError, match="vacía"):
        store.search(normalized([1.0] + [0.0] * (DIMENSION - 1)), top_k=3)


def test_brand_filter_is_applied_by_the_database(store: CatalogVectorStore) -> None:
    seed(store)
    hits = store.search(normalized([1.0] * DIMENSION), top_k=6, brand="Einhell")
    assert hits
    assert {hit.brand for hit in hits} == {"Einhell"}


def test_filter_without_matches_returns_empty_list(store: CatalogVectorStore) -> None:
    seed(store)
    assert (
        store.search(normalized([1.0] * DIMENSION), top_k=3, brand="MarcaInexistente")
        == []
    )


def test_mutations_are_visible_by_id_and_search(store: CatalogVectorStore) -> None:
    records = seed(store)
    target = records[0]
    updated = VectorRecord(
        record_id=target.record_id,
        product_id=target.product_id,
        title="Producto revisado",
        brand=target.brand,
        color="",
        locale="es",
        catalog_version=2,
        active=True,
        text="Producto revisado",
        embedding=target.embedding,
    )
    store.upsert_records([[updated]])
    payload = store.retrieve(target.record_id)
    assert payload is not None
    assert payload["catalog_version"] == 2
    store.delete_records([target.record_id])
    assert store.retrieve(target.record_id) is None
    hits = store.search(np.asarray(target.embedding, dtype=np.float32), top_k=6)
    assert target.record_id not in {hit.record_id for hit in hits}


def test_exact_search_orders_by_cosine_similarity(store: CatalogVectorStore) -> None:
    records = seed(store)
    hits = store.search(
        np.asarray(records[1].embedding, dtype=np.float32), top_k=3, exact=True
    )
    assert hits[0].record_id == records[1].record_id
    assert hits[0].native_score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score_kind == "similarity"
