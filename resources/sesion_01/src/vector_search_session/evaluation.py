"""Information-retrieval metrics for binary and graded relevance judgments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

GradedQrels = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Precision, recall, success, MRR, and nDCG for one query."""

    query_id: str
    k: int
    precision: float
    recall: float
    success: float
    mrr: float
    ndcg: float

    def as_dict(self) -> dict[str, str | int | float]:
        """Return a flat row convenient for tables and Plotly."""

        return {
            "query_id": self.query_id,
            "k": self.k,
            f"precision@{self.k}": self.precision,
            f"recall@{self.k}": self.recall,
            f"success@{self.k}": self.success,
            f"mrr@{self.k}": self.mrr,
            f"ndcg@{self.k}": self.ndcg,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Per-query and macro-averaged retrieval metrics."""

    k: int
    per_query: tuple[QueryMetrics, ...]
    mean_precision: float
    mean_recall: float
    mean_success: float
    mean_mrr: float
    mean_ndcg: float

    @property
    def summary(self) -> dict[str, float]:
        """Return macro metrics with conventional ``metric@k`` labels."""

        return {
            f"precision@{self.k}": self.mean_precision,
            f"recall@{self.k}": self.mean_recall,
            f"success@{self.k}": self.mean_success,
            f"mrr@{self.k}": self.mean_mrr,
            f"ndcg@{self.k}": self.mean_ndcg,
        }

    def per_query_rows(self) -> list[dict[str, str | int | float]]:
        """Return one serializable dictionary per evaluated query."""

        return [metrics.as_dict() for metrics in self.per_query]


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k debe ser un entero positivo.")


def _validate_relevance_threshold(relevance_threshold: float) -> None:
    if not math.isfinite(relevance_threshold) or relevance_threshold <= 0:
        raise ValueError("relevance_threshold debe ser un número positivo y finito.")


def _validated_qrels(qrels: GradedQrels) -> dict[str, float]:
    validated: dict[str, float] = {}
    for document_id, raw_relevance in qrels.items():
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Los IDs de qrels deben ser strings no vacíos.")
        if isinstance(raw_relevance, bool) or not isinstance(
            raw_relevance,
            (int, float),
        ):
            raise TypeError(f"La relevancia de {document_id!r} debe ser numérica.")
        relevance = float(raw_relevance)
        if not math.isfinite(relevance) or relevance < 0:
            raise ValueError(
                f"La relevancia de {document_id!r} debe ser finita y no negativa."
            )
        validated[document_id] = relevance
    return validated


def _document_id_from_ranked_item(item: Any) -> str:
    document_id = item if isinstance(item, str) else getattr(item, "document_id", None)
    if not isinstance(document_id, str) or not document_id:
        raise TypeError(
            "Cada resultado debe ser un document_id o tener un atributo "
            "document_id no vacío."
        )
    return document_id


def _top_k_document_ids(ranking: Sequence[Any], k: int) -> tuple[str, ...]:
    _validate_k(k)
    document_ids = tuple(_document_id_from_ranked_item(item) for item in ranking[:k])
    if len(set(document_ids)) != len(document_ids):
        raise ValueError(
            "El ranking contiene document_ids duplicados dentro del top-k."
        )
    return document_ids


def precision_at_k(
    ranking: Sequence[Any],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> float:
    """Return relevant hits divided by ``k``.

    Dividing by the requested cutoff, rather than by the number returned,
    exposes systems that silently produce fewer than ``k`` candidates.
    """

    _validate_relevance_threshold(relevance_threshold)
    validated_qrels = _validated_qrels(qrels)
    document_ids = _top_k_document_ids(ranking, k)
    relevant_hits = sum(
        validated_qrels.get(document_id, 0.0) >= relevance_threshold
        for document_id in document_ids
    )
    return relevant_hits / k


def recall_at_k(
    ranking: Sequence[Any],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> float:
    """Return the fraction of judged relevant documents retrieved by ``k``."""

    _validate_relevance_threshold(relevance_threshold)
    validated_qrels = _validated_qrels(qrels)
    relevant_ids = {
        document_id
        for document_id, relevance in validated_qrels.items()
        if relevance >= relevance_threshold
    }
    if not relevant_ids:
        return 0.0
    document_ids = _top_k_document_ids(ranking, k)
    return len(relevant_ids.intersection(document_ids)) / len(relevant_ids)


def success_at_k(
    ranking: Sequence[Any],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> float:
    """Return 1 when at least one relevant result appears by ``k``."""

    _validate_relevance_threshold(relevance_threshold)
    validated_qrels = _validated_qrels(qrels)
    document_ids = _top_k_document_ids(ranking, k)
    return float(
        any(
            validated_qrels.get(document_id, 0.0) >= relevance_threshold
            for document_id in document_ids
        )
    )


def reciprocal_rank_at_k(
    ranking: Sequence[Any],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> float:
    """Return the reciprocal rank of the first relevant result by ``k``."""

    _validate_relevance_threshold(relevance_threshold)
    validated_qrels = _validated_qrels(qrels)
    document_ids = _top_k_document_ids(ranking, k)
    for rank, document_id in enumerate(document_ids, start=1):
        if validated_qrels.get(document_id, 0.0) >= relevance_threshold:
            return 1.0 / rank
    return 0.0


def mrr_at_k(
    ranking: Sequence[Any],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> float:
    """Return one query's contribution to mean reciprocal rank at ``k``."""

    return reciprocal_rank_at_k(
        ranking,
        qrels,
        k=k,
        relevance_threshold=relevance_threshold,
    )


def _discounted_cumulative_gain(relevances: Sequence[float]) -> float:
    return sum(
        (2.0**relevance - 1.0) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(relevances, start=1)
    )


def ndcg_at_k(
    ranking: Sequence[Any],
    qrels: GradedQrels,
    *,
    k: int,
) -> float:
    """Return normalized discounted cumulative gain using graded qrels."""

    validated_qrels = _validated_qrels(qrels)
    document_ids = _top_k_document_ids(ranking, k)
    observed_relevances = [
        validated_qrels.get(document_id, 0.0) for document_id in document_ids
    ]
    ideal_relevances = sorted(validated_qrels.values(), reverse=True)[:k]
    ideal_gain = _discounted_cumulative_gain(ideal_relevances)
    if ideal_gain == 0:
        return 0.0
    return _discounted_cumulative_gain(observed_relevances) / ideal_gain


def evaluate_query(
    query_id: str,
    ranking: Sequence[Any],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> QueryMetrics:
    """Compute all supported metrics for one query."""

    if not isinstance(query_id, str) or not query_id:
        raise ValueError("query_id debe ser un string no vacío.")
    return QueryMetrics(
        query_id=query_id,
        k=k,
        precision=precision_at_k(
            ranking,
            qrels,
            k=k,
            relevance_threshold=relevance_threshold,
        ),
        recall=recall_at_k(
            ranking,
            qrels,
            k=k,
            relevance_threshold=relevance_threshold,
        ),
        success=success_at_k(
            ranking,
            qrels,
            k=k,
            relevance_threshold=relevance_threshold,
        ),
        mrr=reciprocal_rank_at_k(
            ranking,
            qrels,
            k=k,
            relevance_threshold=relevance_threshold,
        ),
        ndcg=ndcg_at_k(ranking, qrels, k=k),
    )


def evaluate_rankings(
    rankings: Mapping[str, Sequence[Any]],
    qrels_by_query: Mapping[str, GradedQrels],
    *,
    k: int,
    relevance_threshold: float = 1.0,
) -> EvaluationReport:
    """Evaluate every qrels query and macro-average each metric.

    Missing rankings are evaluated as empty rankings, preventing incomplete
    runs from looking artificially strong. Rankings for unknown query IDs are
    rejected because they usually indicate an identifier mismatch.
    """

    _validate_k(k)
    _validate_relevance_threshold(relevance_threshold)
    if not qrels_by_query:
        raise ValueError("qrels_by_query no puede estar vacío.")
    unknown_query_ids = set(rankings) - set(qrels_by_query)
    if unknown_query_ids:
        values = ", ".join(sorted(unknown_query_ids))
        raise ValueError(f"Hay rankings sin qrels para: {values}.")

    per_query = tuple(
        evaluate_query(
            query_id,
            rankings.get(query_id, ()),
            query_qrels,
            k=k,
            relevance_threshold=relevance_threshold,
        )
        for query_id, query_qrels in sorted(qrels_by_query.items())
    )
    return EvaluationReport(
        k=k,
        per_query=per_query,
        mean_precision=fmean(item.precision for item in per_query),
        mean_recall=fmean(item.recall for item in per_query),
        mean_success=fmean(item.success for item in per_query),
        mean_mrr=fmean(item.mrr for item in per_query),
        mean_ndcg=fmean(item.ndcg for item in per_query),
    )
