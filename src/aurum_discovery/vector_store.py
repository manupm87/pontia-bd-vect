"""Qdrant-backed catalog store: idempotent schema, ingest, search, mutations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from .config import HnswSettings
from .contracts import SearchHit, VectorRecord
from .operations import validate_resource_name, wait_until


class VectorStoreUnavailableError(RuntimeError):
    """Raised when the vector database cannot be reached."""


# Minimal per-segment size (KB) before Qdrant builds the HNSW graph. The
# default (10-20 MB) leaves a 15.000x384 float32 catalog split across small
# segments permanently unindexed, silently degrading every search to a full
# scan; a low threshold forces the index the activity is meant to evaluate.
INDEXING_THRESHOLD_KB = 100


class EmptyCollectionError(RuntimeError):
    """Raised when a search runs against a collection without records."""


class CatalogVectorStore:
    """Native-SDK operations over one activity-scoped Qdrant collection."""

    score_kind = "similarity"
    higher_is_better = True

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        vector_size: int,
        hnsw: HnswSettings,
        ef_search: int,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if vector_size < 1:
            raise ValueError("vector_size debe ser positivo.")
        self._collection_name = validate_resource_name(collection_name)
        self._vector_size = vector_size
        self._hnsw = hnsw
        self._ef_search = ef_search
        self._client = QdrantClient(
            url=url, api_key=api_key or None, timeout=timeout_seconds
        )
        self._url = url

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def ping(self) -> None:
        """Fail fast with a clear message when the database is unreachable."""
        try:
            self._client.get_collections()
        except Exception as error:
            raise VectorStoreUnavailableError(
                f"No se puede conectar con Qdrant en {self._url}. Arranca el "
                "servicio con `make up` y revisa QDRANT_URL en .env."
            ) from error

    def ensure_collection(self, *, allow_reset: bool = False) -> None:
        """Create the collection and its payload index only if they are absent.

        Recreating an existing collection requires the explicit
        ``allow_reset=True`` opt-in (AURUM_ALLOW_RESET en .env).
        """
        self.ping()
        exists = self._client.collection_exists(self._collection_name)
        if exists and allow_reset:
            self._client.delete_collection(self._collection_name)
            exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=self._hnsw.m, ef_construct=self._hnsw.ef_construct
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=INDEXING_THRESHOLD_KB
                ),
            )
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="brand",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self._assert_schema()

    def _assert_schema(self) -> None:
        info = self._client.get_collection(self._collection_name)
        vector_config = info.config.params.vectors
        if not isinstance(vector_config, models.VectorParams):
            raise ValueError(
                f"La colección {self._collection_name!r} usa vectores con nombre; "
                "el contrato de la actividad es un único vector sin nombre."
            )
        if (
            vector_config.size != self._vector_size
            or vector_config.distance != models.Distance.COSINE
        ):
            raise ValueError(
                f"El esquema de {self._collection_name!r} no coincide con la "
                f"configuración: {vector_config!r}. Usa AURUM_ALLOW_RESET=true "
                "para reconstruirla desde cero."
            )

    def _guarded[Value](self, operation: str, call: Callable[[], Value]) -> Value:
        """Translate transport failures into the module's clear Spanish error."""
        try:
            return call()
        except Exception as error:
            raise VectorStoreUnavailableError(
                f"La operación '{operation}' contra {self._collection_name!r} "
                f"falló. Comprueba que Qdrant sigue disponible en {self._url} "
                "(`make up`)."
            ) from error

    def upsert_records(
        self, batches: Iterable[list[VectorRecord]], *, wait: bool = True
    ) -> int:
        """Idempotently upsert record batches; returns the number sent."""
        sent = 0
        for batch in batches:
            if not batch:
                continue
            points = [
                models.PointStruct(
                    id=record.record_id,
                    vector=record.embedding,
                    payload=record.payload(),
                )
                for record in batch
            ]
            self._guarded(
                "upsert",
                lambda points=points: self._client.upsert(
                    collection_name=self._collection_name, wait=wait, points=points
                ),
            )
            sent += len(batch)
        return sent

    def count(self) -> int:
        """Exact number of records currently stored."""
        result = self._guarded(
            "count", lambda: self._client.count(self._collection_name, exact=True)
        )
        return int(result.count)

    def wait_until_indexed(
        self, *, timeout_seconds: float = 180.0, minimum_coverage: float = 0.8
    ) -> dict[str, Any]:
        """Block until the collection is green AND the HNSW index actually exists.

        Qdrant reports a green status even with ``indexed_vectors_count == 0``
        (plain full-scan segments), so waiting for the status alone would
        accept a collection whose ANN configuration is inert.
        """

        def probe() -> dict[str, Any]:
            info = self._guarded(
                "get_collection",
                lambda: self._client.get_collection(self._collection_name),
            )
            return {
                "status": str(info.status),
                "points_count": int(info.points_count or 0),
                "indexed_vectors_count": int(info.indexed_vectors_count or 0),
            }

        def accept(state: dict[str, Any]) -> bool:
            points = int(state["points_count"])
            indexed = int(state["indexed_vectors_count"])
            green = state["status"] == str(models.CollectionStatus.GREEN)
            if points == 0:
                return green
            return green and indexed >= minimum_coverage * points

        state, _, _ = wait_until(
            probe,
            accept=accept,
            timeout_seconds=timeout_seconds,
            interval_seconds=1.0,
            description=(
                f"estado green e índice HNSW construido en {self._collection_name}"
            ),
        )
        return state

    def retrieve(self, record_id: str) -> dict[str, object] | None:
        """Read one record's payload by ID, or None when it does not exist."""
        points = self._guarded(
            "retrieve",
            lambda: self._client.retrieve(
                self._collection_name,
                ids=[record_id],
                with_payload=True,
                with_vectors=False,
            ),
        )
        if not points:
            return None
        return dict(points[0].payload or {})

    def delete_records(self, record_ids: Sequence[str], *, wait: bool = True) -> None:
        """Delete records by ID; deleting an absent ID is a no-op."""
        if not record_ids:
            return
        self._guarded(
            "delete",
            lambda: self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.PointIdsList(points=list(record_ids)),
                wait=wait,
            ),
        )

    def search(
        self,
        vector: NDArray[np.float32] | Sequence[float],
        *,
        top_k: int,
        brand: str | None = None,
        exact: bool = False,
    ) -> list[SearchHit]:
        """Vector search executed by the database, with optional brand filter.

        The brand condition travels inside the query (server-side filtering),
        never as a client-side post-filter. With ``exact=True`` Qdrant scans
        the collection exhaustively, which is the oracle used to measure ANN
        fidelity.
        """
        if top_k < 1:
            raise ValueError("top_k debe ser positivo.")
        query_filter = None
        if brand is not None:
            if not brand.strip():
                raise ValueError("El filtro de marca no puede estar vacío.")
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="brand", match=models.MatchValue(value=brand)
                    )
                ]
            )
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=list(np.asarray(vector, dtype=np.float32)),
                limit=top_k,
                query_filter=query_filter,
                search_params=models.SearchParams(
                    hnsw_ef=None if exact else self._ef_search, exact=exact
                ),
                with_payload=True,
            )
        except Exception as error:
            raise VectorStoreUnavailableError(
                f"La búsqueda contra {self._collection_name!r} falló. Comprueba "
                "que Qdrant sigue disponible (`make up`)."
            ) from error
        if not response.points:
            if self.count() == 0:
                raise EmptyCollectionError(
                    f"La colección {self._collection_name!r} está vacía. Ingiere "
                    "el catálogo con `make ingest` antes de buscar."
                )
            return []
        hits = []
        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})
            hits.append(
                SearchHit(
                    rank=rank,
                    record_id=str(point.id),
                    product_id=str(payload.get("product_id", "")),
                    title=str(payload.get("title", "")),
                    brand=str(payload.get("brand", "")),
                    native_score=float(point.score),
                    score_kind=self.score_kind,
                    higher_is_better=self.higher_is_better,
                )
            )
        return hits

    def delete_collection(self, *, confirmation: str) -> None:
        """Destructive cleanup gated behind the exact confirmation token."""
        expected = f"DELETE:{self._collection_name}"
        if confirmation != expected:
            raise ValueError(
                "Limpieza no confirmada. Define AURUM_CONFIRM_CLEANUP con el "
                f"valor exacto {expected!r} para habilitarla."
            )
        self._client.delete_collection(self._collection_name)


__all__ = [
    "CatalogVectorStore",
    "EmptyCollectionError",
    "VectorStoreUnavailableError",
]
