"""Reusable components for the FAISS and ANN teaching session."""

from .benchmark import BenchmarkResult, benchmark_search, serialized_size_bytes
from .data import SessionData, load_session_data
from .indexes import (
    build_flat_index,
    build_hnsw_index,
    build_ivf_flat_index,
    build_ivf_pq_index,
    configure_faiss_threads,
)
from .metrics import recall_at_k, recall_per_query

__all__ = [
    "BenchmarkResult",
    "SessionData",
    "benchmark_search",
    "build_flat_index",
    "build_hnsw_index",
    "build_ivf_flat_index",
    "build_ivf_pq_index",
    "configure_faiss_threads",
    "load_session_data",
    "recall_at_k",
    "recall_per_query",
    "serialized_size_bytes",
]
