"""Controlled latency and recall measurement for FAISS searches."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import faiss
import numpy as np
from numpy.typing import NDArray

from .metrics import recall_at_k


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One reproducible index/search configuration measurement."""

    index_name: str
    search_parameter: str
    parameter_value: int
    recall_at_k: float
    median_latency_ms: float
    p95_latency_ms: float
    queries_per_second: float
    index_size_bytes: int
    distance_computations_per_query: float | None

    def as_record(self) -> dict[str, object]:
        return {
            "index": self.index_name,
            "parameter": self.search_parameter,
            "value": self.parameter_value,
            "recall_at_k": self.recall_at_k,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "queries_per_second": self.queries_per_second,
            "index_size_bytes": self.index_size_bytes,
            "distance_computations_per_query": self.distance_computations_per_query,
        }


def benchmark_search(
    index: faiss.Index,
    queries: NDArray[np.float32],
    exact_ids: NDArray[np.int64],
    *,
    k: int,
    index_name: str,
    search_parameter: str,
    parameter_value: int,
    repeats: int = 12,
    measured_index_size_bytes: int | None = None,
) -> tuple[BenchmarkResult, NDArray[np.int64]]:
    """Measure batched search after warm-up and return its ranking."""
    
    query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
    index.search(query_matrix[: min(8, len(query_matrix))], k)
    
    elapsed_samples = []
    result_ids = None
    for _ in range(repeats):
        started_at = perf_counter()
        _, result_ids = index.search(query_matrix, k)
        elapsed_samples.append(perf_counter() - started_at)
    
    if result_ids is None:
        raise RuntimeError("Benchmark did not execute")
    
    elapsed_array = np.asarray(elapsed_samples, dtype=np.float64)
    median_seconds = float(np.median(elapsed_array))
    _reset_distance_stats(index)
    _, result_ids = index.search(query_matrix, k)
    distance_computations = _read_distance_computations(index)
    computations_per_query = (
        distance_computations / len(query_matrix)
        if distance_computations is not None
        else None
    )
    result = BenchmarkResult(
        index_name=index_name,
        search_parameter=search_parameter,
        parameter_value=parameter_value,
        recall_at_k=recall_at_k(exact_ids, result_ids, k=k),
        median_latency_ms=median_seconds * 1_000,
        p95_latency_ms=float(np.percentile(elapsed_array, 95) * 1_000),
        queries_per_second=len(query_matrix) / median_seconds,
        index_size_bytes=(
            measured_index_size_bytes
            if measured_index_size_bytes is not None
            else serialized_size_bytes(index)
        ),
        distance_computations_per_query=computations_per_query,
    )
    return result, result_ids


def serialized_size_bytes(index: faiss.Index) -> int:
    """Measure the complete serialized FAISS index size."""
    return int(faiss.serialize_index(index).nbytes)


def _read_distance_computations(index: faiss.Index) -> int | None:
    if isinstance(index, faiss.IndexIVF):
        return int(faiss.cvar.indexIVF_stats.ndis)
    if isinstance(index, faiss.IndexHNSW):
        return int(faiss.cvar.hnsw_stats.ndis)
    return None


def _reset_distance_stats(index: faiss.Index) -> None:
    if isinstance(index, faiss.IndexIVF):
        faiss.cvar.indexIVF_stats.reset()
    elif isinstance(index, faiss.IndexHNSW):
        faiss.cvar.hnsw_stats.reset()
