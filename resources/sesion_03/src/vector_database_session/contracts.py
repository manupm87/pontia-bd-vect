"""Provider-neutral records and reports without erasing provider semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

ScoreKind = Literal["similarity", "distance", "relevance", "unknown"]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One logical product record shared by every provider notebook."""

    record_id: str
    product_id: str
    vector_id: int
    title: str
    brand: str
    color: str
    locale: str
    text: str
    embedding: list[float] = field(repr=False)

    def flat_metadata(self) -> dict[str, str | int]:
        """Return the scalar metadata shape used by Pinecone and Chroma."""
        return {
            "record_id": self.record_id,
            "product_id": self.product_id,
            "vector_id": self.vector_id,
            "title": self.title,
            "brand": self.brand,
            "color": self.color,
            "locale": self.locale,
        }

    def qdrant_payload(self) -> dict[str, Any]:
        """Use LangChain-compatible content and metadata payload paths."""
        return {"text": self.text, "metadata": self.flat_metadata()}


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Normalized result while retaining the database-native score."""

    record_id: str
    product_id: str
    vector_id: int
    title: str
    brand: str
    native_score: float
    score_kind: ScoreKind
    higher_is_better: bool
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderRun:
    """Auditable outcome emitted by a provider laboratory."""

    provider: str
    provider_version: str
    target: str
    resource: str
    record_count: int
    score_kind: ScoreKind
    higher_is_better: bool
    query_id: str
    query_text: str
    top_k: int
    hits: list[SearchHit]
    exact_record_ids: list[str]
    recall_at_k: float
    filtered_query_id: str = "semantic-100455"
    filtered_brand: str = "Einhell"
    filtered_hits: list[SearchHit] = field(default_factory=list)
    filtered_recall_at_k: float | None = None
    durations_ms: dict[str, float] = field(default_factory=dict)
    mutation: dict[str, Any] = field(default_factory=dict)
    visibility: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = "1.0"

    def as_dict(self) -> dict[str, Any]:
        """Serialize nested dataclasses to a JSON-compatible mapping."""
        value = asdict(self)
        value["hits"] = [hit.as_dict() for hit in self.hits]
        value["filtered_hits"] = [hit.as_dict() for hit in self.filtered_hits]
        return value
