"""Graded ranking metrics, ANN fidelity against an exact oracle, and latency."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

from .contracts import LatencyReport, QueryMetrics

ESCI_GAINS: dict[str, int] = {"E": 3, "S": 2, "C": 1, "I": 0}


def gains_for_query(judgments: pd.DataFrame, query_id: str) -> dict[str, int]:
    """Return product_id -> graded gain for one development query."""
    subset = judgments[judgments["query_id"] == query_id]
    if subset.empty:
        raise ValueError(f"No hay relevancias para la consulta {query_id!r}.")
    return {
        row["product_id"]: ESCI_GAINS[row["esci_label"]] for _, row in subset.iterrows()
    }


def _discounted_cumulative_gain(gains: Sequence[int]) -> float:
    return sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))


def ndcg_at_k(
    ranked_product_ids: Sequence[str],
    gains_by_product: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Graded nDCG@k; unjudged retrieved products contribute zero gain."""
    _validate_ranking(ranked_product_ids, k=k)
    observed = [
        gains_by_product.get(product_id, 0) for product_id in ranked_product_ids[:k]
    ]
    ideal = sorted(gains_by_product.values(), reverse=True)[:k]
    ideal_dcg = _discounted_cumulative_gain(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _discounted_cumulative_gain(observed) / ideal_dcg


def recall_at_k(
    ranked_product_ids: Sequence[str],
    relevant_product_ids: set[str],
    *,
    k: int,
) -> float:
    """Fraction of the relevant set retrieved in the top-k.

    The denominator is the full relevant set, so queries with more than k
    relevant products cannot reach 1.0; the ceiling is documented in the
    report instead of silently capping the denominator.
    """
    _validate_ranking(ranked_product_ids, k=k)
    if not relevant_product_ids:
        return 0.0
    retrieved = set(ranked_product_ids[:k])
    return len(retrieved & relevant_product_ids) / len(relevant_product_ids)


def mrr_at_k(
    ranked_product_ids: Sequence[str],
    relevant_product_ids: set[str],
    *,
    k: int,
) -> float:
    """Reciprocal rank of the first relevant product within the top-k."""
    _validate_ranking(ranked_product_ids, k=k)
    for position, product_id in enumerate(ranked_product_ids[:k], start=1):
        if product_id in relevant_product_ids:
            return 1.0 / position
    return 0.0


def evaluate_query(
    query_id: str,
    ranked_product_ids: Sequence[str],
    judgments: pd.DataFrame,
    *,
    k: int,
    recall_relevant_labels: Sequence[str],
    mrr_relevant_labels: Sequence[str],
) -> QueryMetrics:
    """Compute the three declared ranking metrics for one development query."""
    gains = gains_for_query(judgments, query_id)
    subset = judgments[judgments["query_id"] == query_id]
    recall_relevant = set(
        subset[subset["esci_label"].isin(tuple(recall_relevant_labels))]["product_id"]
    )
    mrr_relevant = set(
        subset[subset["esci_label"].isin(tuple(mrr_relevant_labels))]["product_id"]
    )
    return QueryMetrics(
        query_id=query_id,
        ndcg_at_10=ndcg_at_k(ranked_product_ids, gains, k=k),
        recall_at_10=recall_at_k(ranked_product_ids, recall_relevant, k=k),
        mrr_at_10=mrr_at_k(ranked_product_ids, mrr_relevant, k=k),
        judged_count=len(subset),
        relevant_count=len(recall_relevant),
    )


def macro_average(per_query: Sequence[QueryMetrics]) -> dict[str, float]:
    """Macro-average the per-query metrics into the reported aggregates."""
    if not per_query:
        raise ValueError("No hay métricas por consulta que agregar.")
    return {
        "ndcg_at_10": float(np.mean([metrics.ndcg_at_10 for metrics in per_query])),
        "recall_at_10": float(np.mean([metrics.recall_at_10 for metrics in per_query])),
        "mrr_at_10": float(np.mean([metrics.mrr_at_10 for metrics in per_query])),
    }


def overlap_at_k(
    exact_ids: Sequence[str], approximate_ids: Sequence[str], *, k: int
) -> float:
    """ANN fidelity for one query: fraction of exact top-k IDs recovered."""
    if k < 1:
        raise ValueError("k debe ser positivo.")
    expected = set(exact_ids[:k])
    if not expected:
        raise ValueError("El oráculo exacto no devolvió resultados.")
    observed = set(approximate_ids[:k])
    return len(expected & observed) / min(k, len(expected))


def measure_latency(
    operations: Sequence[Callable[[], object]],
    *,
    warmup: int,
    repeats: int,
) -> LatencyReport:
    """Time individual operations after warm-up and report p50/p95 in ms.

    Every operation runs `warmup` untimed rounds first, then `repeats` timed
    rounds; percentiles are computed over all individual samples.
    """
    if not operations:
        raise ValueError("No hay operaciones que medir.")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup debe ser >= 0 y repeats debe ser >= 1.")
    for _ in range(warmup):
        for operation in operations:
            operation()
    samples_ms: list[float] = []
    for _ in range(repeats):
        for operation in operations:
            started_at = perf_counter()
            operation()
            samples_ms.append((perf_counter() - started_at) * 1000)
    return LatencyReport(
        p50_ms=float(median(samples_ms)),
        p95_ms=float(np.percentile(samples_ms, 95)),
        warmup=warmup,
        repeats=repeats,
        query_count=len(operations),
    )


def _validate_ranking(ranked_product_ids: Sequence[str], *, k: int) -> None:
    if k < 1:
        raise ValueError("k debe ser positivo.")
    if len(ranked_product_ids) < k:
        raise ValueError(
            f"El ranking tiene {len(ranked_product_ids)} resultados; "
            f"se necesitan al menos {k}."
        )
    top = list(ranked_product_ids[:k])
    if len(set(top)) != len(top):
        raise ValueError("El ranking contiene product_id repetidos en el top-k.")


__all__ = [
    "ESCI_GAINS",
    "evaluate_query",
    "gains_for_query",
    "macro_average",
    "measure_latency",
    "mrr_at_k",
    "ndcg_at_k",
    "overlap_at_k",
    "recall_at_k",
]
