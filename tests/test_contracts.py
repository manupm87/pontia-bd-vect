"""Validation rules of the shared record and result contracts."""

from __future__ import annotations

import pytest

from aurum_discovery.contracts import DuplicateDecision, SearchHit, VectorRecord


def build_hit(rank: int = 1, product_id: str = "B01", score: float = 0.9) -> SearchHit:
    return SearchHit(
        rank=rank,
        record_id="e1a0e559-6a49-5be5-b617-ec8a4899e975",
        product_id=product_id,
        title="Taladro 24V",
        brand="Einhell",
        native_score=score,
        score_kind="similarity",
        higher_is_better=True,
    )


def test_vector_record_requires_embedding_and_ids() -> None:
    with pytest.raises(ValueError, match="embedding"):
        VectorRecord(
            record_id="abc",
            product_id="B01",
            title="x",
            brand="",
            color="",
            locale="es",
            catalog_version=1,
            active=True,
            text="x",
            embedding=[],
        )


def test_vector_record_payload_excludes_vector_and_text() -> None:
    record = VectorRecord(
        record_id="abc",
        product_id="B01",
        title="Taladro",
        brand="Einhell",
        color="",
        locale="es",
        catalog_version=1,
        active=True,
        text="Taladro. Marca: Einhell",
        embedding=[0.1, 0.2],
    )
    payload = record.payload()
    assert payload["product_id"] == "B01"
    assert payload["catalog_version"] == 1
    assert "embedding" not in payload


def test_search_hit_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        build_hit(rank=0)


def test_positive_duplicate_decision_requires_candidate() -> None:
    with pytest.raises(ValueError, match="product_id"):
        DuplicateDecision(
            incoming_id="EVAL-DUP-001",
            predicted_duplicate=True,
            matched_product_id="",
            score=0.99,
            margin=0.1,
        )


def test_negative_duplicate_decision_writes_empty_candidate() -> None:
    decision = DuplicateDecision(
        incoming_id="EVAL-DUP-002",
        predicted_duplicate=False,
        matched_product_id="B0X",
        score=0.5,
        margin=0.0,
    )
    row = decision.as_result_row()
    assert row["matched_product_id"] == ""
    assert row["predicted_duplicate"] is False
