"""Hand-checked cases for the graded ranking metrics and the ANN overlap."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from aurum_discovery.evaluation import (
    evaluate_query,
    macro_average,
    measure_latency,
    mrr_at_k,
    ndcg_at_k,
    overlap_at_k,
    recall_at_k,
)


def build_judgments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query_id": "q1", "product_id": "A", "esci_label": "E", "relevance": 3},
            {"query_id": "q1", "product_id": "B", "esci_label": "S", "relevance": 2},
            {"query_id": "q1", "product_id": "C", "esci_label": "I", "relevance": 0},
            {"query_id": "q1", "product_id": "D", "esci_label": "C", "relevance": 1},
        ]
    )


def test_ndcg_matches_hand_computation() -> None:
    gains = {"A": 3, "B": 2, "D": 1}
    ranking = ["B", "A", "X", "D", "Y"]
    observed_dcg = 2 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(5)
    ideal_dcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranking, gains, k=5) == pytest.approx(observed_dcg / ideal_dcg)


def test_ndcg_perfect_ranking_is_one() -> None:
    gains = {"A": 3, "B": 2}
    assert ndcg_at_k(["A", "B", "X"], gains, k=3) == pytest.approx(1.0)


def test_recall_uses_full_relevant_set_as_denominator() -> None:
    relevant = {"A", "B", "C", "D"}
    assert recall_at_k(["A", "X", "B"], relevant, k=3) == pytest.approx(0.5)


def test_mrr_returns_reciprocal_of_first_relevant_position() -> None:
    assert mrr_at_k(["X", "Y", "A"], {"A"}, k=3) == pytest.approx(1 / 3)
    assert mrr_at_k(["X", "Y", "Z"], {"A"}, k=3) == pytest.approx(0.0)


def test_rankings_with_duplicates_are_rejected() -> None:
    with pytest.raises(ValueError, match="repetidos"):
        ndcg_at_k(["A", "A", "B"], {"A": 3}, k=3)


def test_evaluate_query_declares_relevant_labels() -> None:
    metrics = evaluate_query(
        "q1",
        ["A", "C", "B"],
        build_judgments(),
        k=3,
        recall_relevant_labels=("E", "S"),
        mrr_relevant_labels=("E",),
    )
    assert metrics.recall_at_10 == pytest.approx(1.0)
    assert metrics.mrr_at_10 == pytest.approx(1.0)
    assert metrics.relevant_count == 2
    aggregates = macro_average([metrics])
    assert aggregates["mrr_at_10"] == pytest.approx(1.0)


def test_overlap_at_k_measures_ann_fidelity() -> None:
    assert overlap_at_k(["a", "b", "c"], ["c", "b", "x"], k=3) == pytest.approx(2 / 3)
    assert overlap_at_k(["a"], ["a"], k=3) == pytest.approx(1.0)


def test_measure_latency_reports_percentiles() -> None:
    report = measure_latency([lambda: None, lambda: None], warmup=1, repeats=3)
    assert report.query_count == 2
    assert report.repeats == 3
    assert report.p50_ms >= 0.0
    assert report.p95_ms >= report.p50_ms


def test_measure_latency_rejects_invalid_protocol() -> None:
    with pytest.raises(ValueError, match="warmup"):
        measure_latency([lambda: None], warmup=-1, repeats=1)
