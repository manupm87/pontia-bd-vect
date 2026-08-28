"""Ranking-fidelity metrics for approximate nearest-neighbor indexes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def recall_per_query(
    exact_ids: NDArray[np.int64],
    approximate_ids: NDArray[np.int64],
    *,
    k: int,
) -> NDArray[np.float64]:
    """Return the fraction of exact top-k IDs recovered for every query."""
    _validate_rankings(exact_ids, approximate_ids, k)
    recalls = np.empty(exact_ids.shape[0], dtype=np.float64)
    for query_position in range(exact_ids.shape[0]):
        expected = set(exact_ids[query_position, :k].tolist())
        observed = set(approximate_ids[query_position, :k].tolist())
        observed.discard(-1)
        recalls[query_position] = len(expected & observed) / k
    return recalls


def recall_at_k(
    exact_ids: NDArray[np.int64],
    approximate_ids: NDArray[np.int64],
    *,
    k: int,
) -> float:
    """Return macro mean ANN recall@k against an exact top-k ranking."""
    return float(recall_per_query(exact_ids, approximate_ids, k=k).mean())


def _validate_rankings(
    exact_ids: NDArray[np.int64],
    approximate_ids: NDArray[np.int64],
    k: int,
) -> None:
    if exact_ids.ndim != 2 or approximate_ids.ndim != 2:
        raise ValueError("Rankings must be two-dimensional")
    if exact_ids.shape[0] != approximate_ids.shape[0]:
        raise ValueError("Rankings must contain the same number of queries")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if exact_ids.shape[1] < k or approximate_ids.shape[1] < k:
        raise ValueError("Both rankings must contain at least k neighbors")
