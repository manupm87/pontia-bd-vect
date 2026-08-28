"""Common retrieval interface: free text in, normalized results out."""

from __future__ import annotations

from typing import Any

from .contracts import SearchHit
from .embeddings import EmbeddingConfiguration, encode_texts, load_encoder
from .vector_store import CatalogVectorStore


class DiscoveryService:
    """Text-to-results facade combining the encoder and the vector store.

    This is the single entry point the rest of the system (CLI, scripts,
    tests) uses to run a query: it applies the configured query prefix,
    encodes the text, and delegates the search — including the brand filter —
    to the database.
    """

    def __init__(
        self,
        *,
        store: CatalogVectorStore,
        configuration: EmbeddingConfiguration,
    ) -> None:
        self._store = store
        self._configuration = configuration
        self._encoder: Any | None = None

    @property
    def store(self) -> CatalogVectorStore:
        return self._store

    @property
    def configuration(self) -> EmbeddingConfiguration:
        return self._configuration

    def _resolve_encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = load_encoder(self._configuration.model_id)
        return self._encoder

    def search_text(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        brand: str | None = None,
        exact: bool = False,
    ) -> list[SearchHit]:
        """Run a semantic query, optionally constrained to one brand."""
        if not query_text.strip():
            raise ValueError("La consulta no puede estar vacía.")
        matrix = encode_texts(
            self._resolve_encoder(),
            [query_text.strip()],
            prefix=self._configuration.query_prefix,
            normalize=self._configuration.normalize,
        )
        return self._store.search(matrix[0], top_k=top_k, brand=brand, exact=exact)


__all__ = ["DiscoveryService"]
