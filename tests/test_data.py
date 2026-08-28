"""Contract tests for the snapshot loaders, IDs, and batching."""

from __future__ import annotations

import numpy as np
import pytest

from aurum_discovery.data import (
    iter_record_batches,
    load_catalog,
    load_catalog_events,
    load_development_judgments,
    load_development_queries,
    load_filtered_queries,
    load_incoming_products,
    load_manifest,
    record_id_for_product,
)


def test_record_id_matches_manifest_namespace() -> None:
    assert record_id_for_product("B000G3T55M") == "e1a0e559-6a49-5be5-b617-ec8a4899e975"


def test_record_id_rejects_blank_product() -> None:
    with pytest.raises(ValueError, match="vacío"):
        record_id_for_product("   ")


def test_sample_catalog_keeps_missing_values_as_empty_strings() -> None:
    catalog = load_catalog(sample=True)
    assert len(catalog) == 1_500
    assert catalog["brand"].isna().sum() == 0
    assert "nan" not in set(catalog["brand"].str.lower())
    assert catalog["active"].dtype == bool


def test_catalog_events_are_ordered_and_typed() -> None:
    events = load_catalog_events()
    assert len(events) == 24
    assert set(events["operation"]) == {"UPSERT", "DELETE"}
    assert events["sequence"].tolist() == list(range(1, 25))


def test_development_workload_is_complete() -> None:
    queries = load_development_queries()
    judgments = load_development_judgments()
    assert len(queries) == 8
    assert set(judgments["query_id"]) == set(queries["query_id"])
    assert set(judgments["esci_label"]) <= {"E", "S", "C", "I"}


def test_filtered_queries_use_supported_brand_filter() -> None:
    queries = load_filtered_queries()
    assert len(queries) == 4
    assert set(queries["filter_field"]) == {"brand"}
    assert set(queries["filter_operator"]) == {"equals"}


def test_incoming_products_expose_labels_only_in_development() -> None:
    development = load_incoming_products(labeled=True)
    evaluation = load_incoming_products(labeled=False)
    assert development["is_duplicate"].dtype == bool
    assert "is_duplicate" not in evaluation.columns


def test_iter_record_batches_preserves_order_and_alignment() -> None:
    catalog = load_catalog(sample=True).head(7)
    embeddings = np.eye(7, 4, dtype=np.float32)
    batches = list(iter_record_batches(catalog, embeddings, batch_size=3))
    assert [len(batch) for batch in batches] == [3, 3, 1]
    flattened = [record for batch in batches for record in batch]
    assert [record.record_id for record in flattened] == catalog["record_id"].tolist()
    assert flattened[2].embedding == embeddings[2].tolist()


def test_iter_record_batches_rejects_misaligned_matrices() -> None:
    catalog = load_catalog(sample=True).head(5)
    with pytest.raises(ValueError, match="alineados"):
        list(iter_record_batches(catalog, np.zeros((4, 4), dtype=np.float32)))


def test_manifest_declares_the_snapshot_contract() -> None:
    manifest = load_manifest()
    assert manifest["counts"]["catalog_records"] == 15_000
    assert manifest["selection"]["relevance_mapping"] == {
        "E": 3,
        "S": 2,
        "C": 1,
        "I": 0,
    }
