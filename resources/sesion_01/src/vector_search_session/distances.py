"""Numerically safe vector similarities and distances implemented with NumPy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DEFAULT_DISTANCE_BATCH_SIZE, DistanceMetric, coerce_distance_metric

FloatArray = NDArray[np.floating[Any]]


def _as_finite_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(f"{name} debe contener únicamente valores numéricos reales.")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    elif array.dtype == np.float16:
        array = array.astype(np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contiene NaN o infinito.")
    return array


def validate_vector_collection(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Validate and return a matrix of candidates and one compatible query."""

    candidates = _as_finite_float_array(
        candidate_vectors,
        name="candidate_vectors",
    )
    query = _as_finite_float_array(query_vector, name="query_vector")

    if candidates.ndim != 2:
        raise ValueError(
            "candidate_vectors debe tener forma (n_documentos, dimensiones)."
        )
    if query.ndim != 1:
        raise ValueError("query_vector debe tener forma (dimensiones,).")
    if candidates.shape[0] == 0:
        raise ValueError("candidate_vectors debe contener al menos un vector.")
    if candidates.shape[1] == 0:
        raise ValueError("Los vectores deben tener al menos una dimensión.")
    if candidates.shape[1] != query.shape[0]:
        raise ValueError(
            "Dimensiones incompatibles: "
            f"candidatos={candidates.shape[1]}, consulta={query.shape[0]}."
        )
    return candidates, query


def safe_l2_normalize(
    values: ArrayLike,
    *,
    axis: int = -1,
    epsilon: float = 1e-12,
) -> FloatArray:
    """L2-normalize an array while leaving zero vectors as zeros.

    Only one output array and the much smaller norm array are allocated.  This
    matters when the function is used with a large, precomputed corpus matrix.
    """

    if epsilon <= 0:
        raise ValueError("epsilon debe ser positivo.")
    array = _as_finite_float_array(values, name="values")
    if array.ndim == 0:
        raise ValueError("values debe ser un vector o una matriz.")

    try:
        norms = np.linalg.norm(array, axis=axis, keepdims=True)
    except np.exceptions.AxisError as error:
        raise ValueError(
            f"axis={axis} no existe para un array de {array.ndim} dimensiones."
        ) from error

    normalized = np.zeros_like(array)
    np.divide(array, norms, out=normalized, where=norms > epsilon)
    return normalized


def dot_product_scores(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
) -> FloatArray:
    """Compute one dot-product similarity per candidate vector."""

    candidates, query = validate_vector_collection(candidate_vectors, query_vector)
    return np.asarray(candidates @ query)


def cosine_similarity_scores(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> FloatArray:
    """Compute cosine similarities; any comparison with a zero vector is zero."""

    if epsilon <= 0:
        raise ValueError("epsilon debe ser positivo.")
    candidates, query = validate_vector_collection(candidate_vectors, query_vector)
    dot_products = candidates @ query
    candidate_norms = np.linalg.norm(candidates, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominators = candidate_norms * query_norm
    scores = np.zeros_like(dot_products)
    np.divide(
        dot_products,
        denominators,
        out=scores,
        where=denominators > epsilon,
    )
    return np.clip(scores, -1.0, 1.0)


def squared_l2_distance_scores(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
) -> FloatArray:
    """Compute squared Euclidean distances without an ``n x d`` difference."""

    candidates, query = validate_vector_collection(candidate_vectors, query_vector)
    accumulator_dtype = np.result_type(candidates.dtype, query.dtype, np.float64)
    candidate_squared_norms = np.einsum(
        "ij,ij->i",
        candidates,
        candidates,
        dtype=accumulator_dtype,
    )
    query_squared_norm = np.einsum(
        "i,i->",
        query,
        query,
        dtype=accumulator_dtype,
    )
    dot_products = np.einsum(
        "ij,j->i",
        candidates,
        query,
        dtype=accumulator_dtype,
    )
    squared_distances = (
        candidate_squared_norms + query_squared_norm - 2.0 * dot_products
    )
    return np.maximum(squared_distances, 0.0)


def l2_distance_scores(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
) -> FloatArray:
    """Compute Euclidean distances."""

    return np.sqrt(squared_l2_distance_scores(candidate_vectors, query_vector))


def _batched_coordinate_distance(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
    *,
    reducer: Callable[..., NDArray[Any]],
    batch_size: int,
) -> FloatArray:
    if batch_size < 1:
        raise ValueError("batch_size debe ser un entero positivo.")
    candidates, query = validate_vector_collection(candidate_vectors, query_vector)
    output_dtype = np.result_type(candidates.dtype, query.dtype, np.float32)
    scores = np.empty(candidates.shape[0], dtype=output_dtype)

    for start_index in range(0, candidates.shape[0], batch_size):
        stop_index = min(start_index + batch_size, candidates.shape[0])
        absolute_differences = np.abs(candidates[start_index:stop_index] - query)
        scores[start_index:stop_index] = reducer(absolute_differences, axis=1)
    return scores


def l1_distance_scores(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
    *,
    batch_size: int = DEFAULT_DISTANCE_BATCH_SIZE,
) -> FloatArray:
    """Compute Manhattan distances in bounded-memory batches."""

    return _batched_coordinate_distance(
        candidate_vectors,
        query_vector,
        reducer=np.sum,
        batch_size=batch_size,
    )


def chebyshev_distance_scores(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
    *,
    batch_size: int = DEFAULT_DISTANCE_BATCH_SIZE,
) -> FloatArray:
    """Compute maximum-coordinate (Chebyshev) distances in batches."""

    return _batched_coordinate_distance(
        candidate_vectors,
        query_vector,
        reducer=np.max,
        batch_size=batch_size,
    )


def compute_vector_scores(
    candidate_vectors: ArrayLike,
    query_vector: ArrayLike,
    metric: DistanceMetric | str,
    *,
    batch_size: int = DEFAULT_DISTANCE_BATCH_SIZE,
) -> FloatArray:
    """Dispatch to one of the exact metrics supported by the session."""

    canonical_metric = coerce_distance_metric(metric)
    if canonical_metric is DistanceMetric.COSINE:
        return cosine_similarity_scores(candidate_vectors, query_vector)
    if canonical_metric is DistanceMetric.DOT:
        return dot_product_scores(candidate_vectors, query_vector)
    if canonical_metric is DistanceMetric.L2:
        return l2_distance_scores(candidate_vectors, query_vector)
    if canonical_metric is DistanceMetric.L2_SQUARED:
        return squared_l2_distance_scores(candidate_vectors, query_vector)
    if canonical_metric is DistanceMetric.L1:
        return l1_distance_scores(
            candidate_vectors,
            query_vector,
            batch_size=batch_size,
        )
    return chebyshev_distance_scores(
        candidate_vectors,
        query_vector,
        batch_size=batch_size,
    )


def metric_prefers_higher_scores(metric: DistanceMetric | str) -> bool:
    """Return whether a larger score represents a better match."""

    canonical_metric = coerce_distance_metric(metric)
    return canonical_metric in {DistanceMetric.COSINE, DistanceMetric.DOT}
