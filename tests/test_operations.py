"""Safety guards, polling behavior, and delivery-artifact validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aurum_discovery.contracts import DuplicateDecision, SearchHit
from aurum_discovery.operations import (
    validate_resource_name,
    wait_until,
    write_duplicate_results,
    write_search_results,
)


def build_hit(rank: int, product_id: str) -> SearchHit:
    return SearchHit(
        rank=rank,
        record_id=f"id-{product_id}",
        product_id=product_id,
        title=f"Producto {product_id}",
        brand="NIKE",
        native_score=1.0 - rank * 0.01,
        score_kind="similarity",
        higher_is_better=True,
    )


def test_resource_names_outside_the_activity_prefix_are_refused() -> None:
    assert validate_resource_name("aurum-market-eval-catalogo")
    with pytest.raises(ValueError, match="prefijo"):
        validate_resource_name("produccion-catalogo")


def test_wait_until_returns_value_elapsed_and_attempts() -> None:
    values = iter([0, 0, 3])
    value, elapsed, attempts = wait_until(
        lambda: next(values), interval_seconds=0.0, timeout_seconds=5.0
    )
    assert value == 3
    assert attempts == 3
    assert elapsed >= 0.0


def test_wait_until_times_out_with_last_value() -> None:
    with pytest.raises(TimeoutError, match="last value=0"):
        wait_until(
            lambda: 0,
            interval_seconds=0.0,
            timeout_seconds=0.05,
            description="recuento estable",
        )


def test_write_search_results_enforces_the_contract(tmp_path: Path) -> None:
    rankings = {
        "EVAL-1": [build_hit(rank, f"P{rank}") for rank in range(1, 11)],
    }
    path = write_search_results(
        rankings, tmp_path / "resultados_busqueda.csv", top_k=10
    )
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert rows[0]["evaluation_id"] == "EVAL-1"
    assert [row["rank"] for row in rows] == [str(rank) for rank in range(1, 11)]


def test_write_search_results_rejects_repeated_products(tmp_path: Path) -> None:
    hits = [build_hit(rank, "P1" if rank < 3 else f"P{rank}") for rank in range(1, 11)]
    with pytest.raises(ValueError, match="repite"):
        write_search_results({"EVAL-1": hits}, tmp_path / "salida.csv", top_k=10)


def test_write_search_results_rejects_short_rankings(tmp_path: Path) -> None:
    hits = [build_hit(rank, f"P{rank}") for rank in range(1, 9)]
    with pytest.raises(ValueError, match="exactamente"):
        write_search_results({"EVAL-1": hits}, tmp_path / "salida.csv", top_k=10)


def test_write_duplicate_results_serializes_booleans(tmp_path: Path) -> None:
    decisions = [
        DuplicateDecision(
            incoming_id="EVAL-DUP-001",
            predicted_duplicate=True,
            matched_product_id="B01",
            score=0.97,
            margin=0.2,
        ),
        DuplicateDecision(
            incoming_id="EVAL-DUP-002",
            predicted_duplicate=False,
            matched_product_id="",
            score=0.55,
            margin=0.01,
        ),
    ]
    path = write_duplicate_results(decisions, tmp_path / "resultados_duplicados.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["predicted_duplicate"] == "true"
    assert rows[0]["matched_product_id"] == "B01"
    assert rows[1]["predicted_duplicate"] == "false"
    assert rows[1]["matched_product_id"] == ""
