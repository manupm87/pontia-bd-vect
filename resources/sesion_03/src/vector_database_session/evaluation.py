"""Exact cosine oracle and provider-ranking evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .contracts import SearchHit
from .data import SessionData, record_id_for_product


def exact_top_k(
    data: SessionData,
    query_vector: NDArray[np.float32],
    *,
    k: int = 10,
    brand: str | None = None,
) -> list[SearchHit]:
    """Return exact cosine neighbors, optionally inside one brand subset."""
    if k <= 0:
        raise ValueError("k must be positive")
    vector = np.asarray(query_vector, dtype=np.float32)
    if vector.shape != (data.product_embeddings.shape[1],):
        raise ValueError(f"Unexpected query shape: {vector.shape}")
    candidate_positions = np.arange(len(data.products), dtype=np.int64)
    if brand is not None:
        brands = data.products["product_brand"].fillna("").astype(str).to_numpy()
        candidate_positions = candidate_positions[brands == brand]
    if len(candidate_positions) == 0:
        return []
    scores = np.asarray(data.product_embeddings[candidate_positions] @ vector)
    selected_count = min(k, len(candidate_positions))
    partition = np.argpartition(-scores, selected_count - 1)[:selected_count]
    ordered_local = partition[np.argsort(-scores[partition], kind="stable")]
    hits = []
    for rank, local_position in enumerate(ordered_local, start=1):
        product_position = int(candidate_positions[local_position])
        row = data.products.iloc[product_position]
        product_id = str(row["product_id"])
        hits.append(
            SearchHit(
                record_id=record_id_for_product(product_id),
                product_id=product_id,
                vector_id=int(row["vector_id"]),
                title=str(row["product_title"]),
                brand=""
                if row["product_brand"] != row["product_brand"]
                else str(row["product_brand"]),
                native_score=float(scores[local_position]),
                score_kind="similarity",
                higher_is_better=True,
                rank=rank,
            )
        )
    return hits


def evaluate_run(
    exact_hits: Iterable[SearchHit], provider_hits: Iterable[SearchHit], *, k: int = 10
) -> dict[str, object]:
    """Compare IDs only; do not pretend provider-native scores are commensurable."""
    exact_ids = [hit.record_id for hit in exact_hits][:k]
    provider_ids = [hit.record_id for hit in provider_hits][:k]
    overlap = [record_id for record_id in provider_ids if record_id in set(exact_ids)]
    denominator = min(k, len(exact_ids))
    recall = len(overlap) / denominator if denominator else 0.0
    return {
        "k": k,
        "exact_record_ids": exact_ids,
        "provider_record_ids": provider_ids,
        "overlap_record_ids": overlap,
        "recall_at_k": recall,
    }
