"""Explicit FAISS index builders with reproducible defaults."""

from __future__ import annotations

import os

import faiss
import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]


def configure_faiss_threads(thread_count: int | None = None) -> int:
    """Set and return the FAISS OpenMP thread count."""
    resolved_count = thread_count or int(os.getenv("FAISS_NUM_THREADS", "1"))
    if resolved_count <= 0:
        raise ValueError("thread_count must be positive")
    faiss.omp_set_num_threads(resolved_count)
    return resolved_count


def build_flat_index(embeddings: FloatMatrix) -> faiss.IndexFlatIP:
    """Build the exact inner-product ground-truth index."""
    vectors = _validated_vectors(embeddings)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_ivf_flat_index(
    embeddings: FloatMatrix,
    *,
    nlist: int = 256,
    training_size: int = 30_000,
    seed: int = 42,
) -> faiss.IndexIVFFlat:
    """Train and populate an IVF index that stores full vectors."""
    vectors = _validated_vectors(embeddings)
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFFlat(
        quantizer,
        vectors.shape[1],
        nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    training_vectors = _training_sample(vectors, training_size, seed)
    index.train(training_vectors)
    index.add(vectors)
    return index


def build_hnsw_index(
    embeddings: FloatMatrix,
    *,
    graph_degree: int = 24,
    ef_construction: int = 120,
) -> faiss.IndexHNSWFlat:
    """Build a graph-based HNSW index over full vectors."""
    vectors = _validated_vectors(embeddings)
    index = faiss.IndexHNSWFlat(
        vectors.shape[1], graph_degree, faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def build_ivf_pq_index(
    embeddings: FloatMatrix,
    *,
    nlist: int = 256,
    subquantizers: int = 48,
    bits_per_code: int = 8,
    training_size: int = 30_000,
    seed: int = 42,
) -> faiss.IndexIVFPQ:
    """Train IVF with Product Quantization residual codes."""
    vectors = _validated_vectors(embeddings)
    if vectors.shape[1] % subquantizers != 0:
        raise ValueError("Embedding dimension must be divisible by subquantizers")
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(
        quantizer,
        vectors.shape[1],
        nlist,
        subquantizers,
        bits_per_code,
        faiss.METRIC_INNER_PRODUCT,
    )
    training_vectors = _training_sample(vectors, training_size, seed)
    index.train(training_vectors)
    index.add(vectors)
    return index


def _validated_vectors(embeddings: FloatMatrix) -> FloatMatrix:
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("embeddings must be a non-empty two-dimensional matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain NaN or infinity")
    return vectors


def _training_sample(
    vectors: FloatMatrix,
    training_size: int,
    seed: int,
) -> FloatMatrix:
    sample_size = min(training_size, len(vectors))
    random_generator = np.random.default_rng(seed)
    positions = random_generator.choice(len(vectors), sample_size, replace=False)
    return np.ascontiguousarray(vectors[positions], dtype=np.float32)
