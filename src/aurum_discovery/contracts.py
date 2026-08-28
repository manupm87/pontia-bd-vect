"""Provider-neutral records and reports without erasing native score semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

ScoreKind = Literal["similarity", "distance", "relevance", "unknown"]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Canonical ingest unit: one catalog product with its embedding."""

    record_id: str
    product_id: str
    title: str
    brand: str
    color: str
    locale: str
    catalog_version: int
    active: bool
    text: str
    embedding: list[float] = field(repr=False)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id no puede estar vacío.")
        if not self.product_id.strip():
            raise ValueError("product_id no puede estar vacío.")
        if not self.embedding:
            raise ValueError("El registro debe incluir un embedding no vacío.")

    def payload(self) -> dict[str, Any]:
        """Return the metadata stored next to the vector in the database."""
        return {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "color": self.color,
            "locale": self.locale,
            "catalog_version": self.catalog_version,
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Normalized single search result keeping the provider-native score."""

    rank: int
    record_id: str
    product_id: str
    title: str
    brand: str
    native_score: float
    score_kind: ScoreKind
    higher_is_better: bool

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank debe ser un entero positivo empezando en 1.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Ranking quality metrics for a single development query."""

    query_id: str
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    judged_count: int
    relevant_count: int

    def as_record(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_10": self.recall_at_10,
            "mrr_at_10": self.mrr_at_10,
            "judged_count": self.judged_count,
            "relevant_count": self.relevant_count,
        }


@dataclass(frozen=True, slots=True)
class LatencyReport:
    """Latency percentiles for one reproducible measurement protocol."""

    p50_ms: float
    p95_ms: float
    warmup: int
    repeats: int
    query_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    """Decision emitted by the duplicate-control rule for one incoming record."""

    incoming_id: str
    predicted_duplicate: bool
    matched_product_id: str
    score: float
    margin: float

    def __post_init__(self) -> None:
        if self.predicted_duplicate and not self.matched_product_id.strip():
            raise ValueError(
                "Una predicción positiva debe señalar el product_id duplicado."
            )

    def as_result_row(self) -> dict[str, object]:
        """Return the row contract required by resultados_duplicados.csv."""
        return {
            "incoming_id": self.incoming_id,
            "predicted_duplicate": self.predicted_duplicate,
            "matched_product_id": self.matched_product_id
            if self.predicted_duplicate
            else "",
            "score": self.score,
        }


@dataclass(slots=True)
class EvaluationRun:
    """Auditable report persisted after evaluating the final configuration."""

    snapshot_id: str
    embedding_configuration: str
    collection: str
    record_count: int
    score_kind: ScoreKind
    higher_is_better: bool
    top_k: int
    per_query: list[QueryMetrics] = field(default_factory=list)
    ndcg_at_10: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    recall_relevant_labels: list[str] = field(default_factory=list)
    mrr_relevant_labels: list[str] = field(default_factory=list)
    ann_fidelity: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    duplicates: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = "1.0"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["per_query"] = [metrics.as_record() for metrics in self.per_query]
        return payload


__all__ = [
    "DuplicateDecision",
    "EvaluationRun",
    "LatencyReport",
    "QueryMetrics",
    "ScoreKind",
    "SearchHit",
    "VectorRecord",
]
