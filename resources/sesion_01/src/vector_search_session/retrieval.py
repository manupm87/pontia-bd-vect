"""Exact dense, TF-IDF, and LSA retrieval with deterministic rankings."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    DEFAULT_DISTANCE_BATCH_SIZE,
    DEFAULT_TOP_K,
    DistanceMetric,
    LatencySettings,
    LsaSettings,
    TextRole,
    TfidfSettings,
    coerce_distance_metric,
)
from .distances import (
    compute_vector_scores,
    metric_prefers_higher_scores,
    safe_l2_normalize,
)

InputValue = TypeVar("InputValue")
OutputValue = TypeVar("OutputValue")
E5_MODEL_PATTERN = re.compile(r"(?:^|[/_-])e5(?:[-_.]|$)", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked document returned by a retriever."""

    rank: int
    document_id: str
    document_index: int
    score: float


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Distribution of repeated, single-query latency measurements."""

    samples_ms: tuple[float, ...]
    mean_ms: float
    p50_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float
    operations_per_second: float

    @property
    def sample_count(self) -> int:
        """Number of timed operations."""

        return len(self.samples_ms)


def _validate_document_ids(
    document_ids: Sequence[str] | None,
    document_count: int,
) -> tuple[str, ...]:
    if document_ids is None:
        normalized_ids = tuple(str(index) for index in range(document_count))
    else:
        if isinstance(document_ids, str):
            raise TypeError("document_ids debe ser una secuencia, no un único string.")
        normalized_ids = tuple(document_ids)
    if len(normalized_ids) != document_count:
        raise ValueError(
            "document_ids y la colección deben tener la misma longitud: "
            f"{len(normalized_ids)} != {document_count}."
        )
    if any(not isinstance(item, str) or not item.strip() for item in normalized_ids):
        raise ValueError("Cada document_id debe ser un string no vacío.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("document_ids contiene identificadores duplicados.")
    return normalized_ids


def _validate_text_collection(document_texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(document_texts, str):
        raise TypeError("La colección debe ser una secuencia de textos.")
    normalized_texts = tuple(document_texts)
    if not normalized_texts:
        raise ValueError("La colección debe contener al menos un documento.")
    if any(not isinstance(text, str) or not text.strip() for text in normalized_texts):
        raise ValueError("Cada documento debe ser un string no vacío.")
    return normalized_texts


def _validate_embedding_matrix(embeddings: ArrayLike) -> NDArray[np.floating[Any]]:
    matrix = np.asarray(embeddings)
    if np.issubdtype(matrix.dtype, np.complexfloating) or not np.issubdtype(
        matrix.dtype,
        np.number,
    ):
        raise TypeError("embeddings debe contener valores numéricos reales.")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "embeddings debe tener forma no vacía (n_documentos, dimensiones)."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contiene NaN o infinito.")
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(np.float64)
    elif matrix.dtype == np.float16:
        matrix = matrix.astype(np.float32)
    return matrix


def stable_top_k_indices(
    scores: ArrayLike,
    *,
    k: int,
    higher_is_better: bool,
) -> NDArray[np.intp]:
    """Return exact top-k indices with original order as the tie breaker.

    NumPy partitioning finds the cutoff in linear average time. We then keep
    every score strictly better than the cutoff, fill the remaining positions
    with the earliest ties, and sort only the selected candidates. This keeps
    deterministic ties without paying for a full ``n log n`` sort when
    ``k`` is small.
    """

    score_array = np.asarray(scores)
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError("scores debe ser un vector unidimensional no vacío.")
    if not np.issubdtype(score_array.dtype, np.number):
        raise TypeError("scores debe contener valores numéricos.")
    numeric_scores = score_array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric_scores)):
        raise ValueError("scores contiene NaN o infinito.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k debe ser un entero positivo.")

    effective_k = min(k, score_array.size)
    if effective_k == score_array.size:
        sortable_scores = -numeric_scores if higher_is_better else numeric_scores
        return np.argsort(sortable_scores, kind="stable")

    if higher_is_better:
        cutoff_position = score_array.size - effective_k
        cutoff_score = np.partition(numeric_scores, cutoff_position)[cutoff_position]
        strictly_better = np.flatnonzero(numeric_scores > cutoff_score)
    else:
        cutoff_position = effective_k - 1
        cutoff_score = np.partition(numeric_scores, cutoff_position)[cutoff_position]
        strictly_better = np.flatnonzero(numeric_scores < cutoff_score)

    remaining_count = effective_k - strictly_better.size
    cutoff_ties = np.flatnonzero(numeric_scores == cutoff_score)[:remaining_count]
    selected_indices = np.concatenate((strictly_better, cutoff_ties))
    selected_scores = numeric_scores[selected_indices]
    primary_sort_key = -selected_scores if higher_is_better else selected_scores
    local_order = np.lexsort((selected_indices, primary_sort_key))
    return selected_indices[local_order]


def _build_results(
    indices: NDArray[np.intp],
    scores: ArrayLike,
    document_ids: Sequence[str],
) -> tuple[SearchResult, ...]:
    score_array = np.asarray(scores)
    return tuple(
        SearchResult(
            rank=rank,
            document_id=document_ids[int(document_index)],
            document_index=int(document_index),
            score=float(score_array[document_index]),
        )
        for rank, document_index in enumerate(indices, start=1)
    )


def is_e5_model(model_name: str) -> bool:
    """Return whether ``model_name`` follows the E5 input-prefix contract."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name debe ser un string no vacío.")
    return E5_MODEL_PATTERN.search(model_name.strip()) is not None


def prepare_texts_for_embedding(
    texts: Sequence[str],
    *,
    model_name: str,
    role: TextRole | str,
) -> tuple[str, ...]:
    """Add ``query:``/``passage:`` only for E5-family embedding models.

    TF-IDF and LSA retrievers deliberately use the original text. Prefixes are
    part of an embedding model's contract, not linguistic content.
    """

    if isinstance(texts, str):
        raise TypeError("texts debe ser una secuencia, no un único string.")
    normalized_texts = tuple(texts)
    if not normalized_texts:
        raise ValueError("texts debe contener al menos un texto.")
    if any(not isinstance(text, str) or not text.strip() for text in normalized_texts):
        raise ValueError("Cada texto debe ser un string no vacío.")
    try:
        text_role = TextRole(role)
    except ValueError as error:
        raise ValueError("role debe ser 'query' o 'document'.") from error
    if not is_e5_model(model_name):
        return normalized_texts

    prefix = "query: " if text_role is TextRole.QUERY else "passage: "
    prefixed_texts: list[str] = []
    for text in normalized_texts:
        stripped_text = text.strip()
        if stripped_text.casefold().startswith(prefix.casefold()):
            prefixed_texts.append(stripped_text)
        else:
            prefixed_texts.append(f"{prefix}{stripped_text}")
    return tuple(prefixed_texts)


class DenseRetriever:
    """Exact retriever over a matrix of precomputed dense embeddings."""

    def __init__(
        self,
        embeddings: ArrayLike,
        document_ids: Sequence[str] | None = None,
        *,
        metric: DistanceMetric | str = DistanceMetric.COSINE,
        normalize_embeddings: bool = False,
        distance_batch_size: int = DEFAULT_DISTANCE_BATCH_SIZE,
    ) -> None:
        embedding_matrix = _validate_embedding_matrix(embeddings)
        if distance_batch_size < 1:
            raise ValueError("distance_batch_size debe ser positivo.")
        self._embeddings = (
            safe_l2_normalize(embedding_matrix, axis=1)
            if normalize_embeddings
            else embedding_matrix
        )
        self._document_ids = _validate_document_ids(
            document_ids,
            self._embeddings.shape[0],
        )
        self._metric = coerce_distance_metric(metric)
        self._distance_batch_size = distance_batch_size

    @property
    def metric(self) -> DistanceMetric:
        """Metric used to rank candidates."""

        return self._metric

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Document identifiers in the same order as the embedding matrix."""

        return self._document_ids

    @property
    def embedding_dimension(self) -> int:
        """Number of coordinates in each embedding."""

        return int(self._embeddings.shape[1])

    @property
    def document_count(self) -> int:
        """Number of indexed vectors."""

        return int(self._embeddings.shape[0])

    def search(
        self,
        query_embedding: ArrayLike,
        *,
        k: int = DEFAULT_TOP_K,
    ) -> tuple[SearchResult, ...]:
        """Run one exact vector search."""

        scores = compute_vector_scores(
            self._embeddings,
            query_embedding,
            self._metric,
            batch_size=self._distance_batch_size,
        )
        indices = stable_top_k_indices(
            scores,
            k=k,
            higher_is_better=metric_prefers_higher_scores(self._metric),
        )
        return _build_results(indices, scores, self._document_ids)

    def search_many(
        self,
        query_embeddings: ArrayLike,
        *,
        k: int = DEFAULT_TOP_K,
    ) -> tuple[tuple[SearchResult, ...], ...]:
        """Search a query matrix without duplicating the corpus matrix."""

        query_matrix = _validate_embedding_matrix(query_embeddings)
        if query_matrix.shape[1] != self.embedding_dimension:
            raise ValueError(
                "Dimensiones incompatibles: "
                f"índice={self.embedding_dimension}, consultas={query_matrix.shape[1]}."
            )
        return tuple(self.search(query, k=k) for query in query_matrix)


class TfidfRetriever:
    """Sparse lexical baseline using L2-normalized TF-IDF vectors."""

    def __init__(
        self,
        document_texts: Sequence[str],
        document_ids: Sequence[str] | None = None,
        *,
        settings: TfidfSettings | None = None,
    ) -> None:
        texts = _validate_text_collection(document_texts)
        self._document_ids = _validate_document_ids(document_ids, len(texts))
        self._settings = settings or TfidfSettings()
        self._vectorizer = _make_tfidf_vectorizer(self._settings)
        try:
            self._document_matrix = self._vectorizer.fit_transform(texts)
        except ValueError as error:
            raise ValueError(
                "No se pudo construir el vocabulario TF-IDF. Revisa el corpus "
                "y los umbrales de frecuencia."
            ) from error

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Indexed document IDs."""

        return self._document_ids

    @property
    def vocabulary_size(self) -> int:
        """Number of lexical dimensions learned from the corpus."""

        return len(self._vectorizer.vocabulary_)

    def search(
        self,
        query_text: str,
        *,
        k: int = DEFAULT_TOP_K,
    ) -> tuple[SearchResult, ...]:
        """Rank documents by sparse cosine similarity."""

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text debe ser un string no vacío.")
        query_vector = self._vectorizer.transform([query_text])
        scores = (query_vector @ self._document_matrix.T).toarray().ravel()
        indices = stable_top_k_indices(scores, k=k, higher_is_better=True)
        return _build_results(indices, scores, self._document_ids)


def _make_tfidf_vectorizer(settings: TfidfSettings) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=settings.ngram_range,
        min_df=settings.min_document_frequency,
        max_df=settings.max_document_frequency,
        max_features=settings.max_features,
        lowercase=settings.lowercase,
        strip_accents=settings.strip_accents,
        sublinear_tf=settings.sublinear_term_frequency,
        norm="l2",
        dtype=np.float32,
    )


class LsaRetriever:
    """Dense semantic baseline obtained from TF-IDF with truncated SVD."""

    def __init__(
        self,
        document_texts: Sequence[str],
        document_ids: Sequence[str] | None = None,
        *,
        tfidf_settings: TfidfSettings | None = None,
        lsa_settings: LsaSettings | None = None,
        metric: DistanceMetric | str = DistanceMetric.COSINE,
    ) -> None:
        texts = _validate_text_collection(document_texts)
        normalized_document_ids = _validate_document_ids(document_ids, len(texts))
        self._tfidf_settings = tfidf_settings or TfidfSettings()
        self._lsa_settings = lsa_settings or LsaSettings()
        self._vectorizer = _make_tfidf_vectorizer(self._tfidf_settings)
        try:
            document_tfidf = self._vectorizer.fit_transform(texts)
        except ValueError as error:
            raise ValueError(
                "No se pudo construir el vocabulario necesario para LSA."
            ) from error

        maximum_rank = min(document_tfidf.shape)
        if maximum_rank < 2:
            raise ValueError(
                "LSA necesita al menos dos documentos y dos términos utilizables."
            )
        self._effective_dimensions = min(
            self._lsa_settings.dimensions,
            maximum_rank - 1,
        )
        self._svd = TruncatedSVD(
            n_components=self._effective_dimensions,
            random_state=self._lsa_settings.random_seed,
        )
        document_embeddings = self._svd.fit_transform(document_tfidf)
        if self._lsa_settings.normalize_embeddings:
            document_embeddings = safe_l2_normalize(document_embeddings, axis=1)
        self._dense_retriever = DenseRetriever(
            document_embeddings,
            normalized_document_ids,
            metric=metric,
        )

    @property
    def effective_dimensions(self) -> int:
        """Actual LSA dimensions after respecting the corpus rank."""

        return self._effective_dimensions

    @property
    def explained_variance_ratio(self) -> float:
        """Total variance captured by the retained latent dimensions."""

        return float(self._svd.explained_variance_ratio_.sum())

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Indexed document IDs."""

        return self._dense_retriever.document_ids

    def transform_queries(
        self, query_texts: Sequence[str]
    ) -> NDArray[np.floating[Any]]:
        """Project query texts into the fitted latent space."""

        queries = _validate_text_collection(query_texts)
        query_tfidf = self._vectorizer.transform(queries)
        query_embeddings = self._svd.transform(query_tfidf)
        if self._lsa_settings.normalize_embeddings:
            query_embeddings = safe_l2_normalize(query_embeddings, axis=1)
        return query_embeddings

    def search(
        self,
        query_text: str,
        *,
        k: int = DEFAULT_TOP_K,
    ) -> tuple[SearchResult, ...]:
        """Project and retrieve one query."""

        query_embedding = self.transform_queries([query_text])[0]
        return self._dense_retriever.search(query_embedding, k=k)


def measure_repeated_latency(
    operation: Callable[[InputValue], OutputValue],
    inputs: Sequence[InputValue],
    *,
    settings: LatencySettings | None = None,
) -> LatencySummary:
    """Measure one operation at a time without copying its corpus or outputs.

    Warm-up calls are excluded. The function stores only one scalar duration
    per call, so increasing repetitions does not retain result objects or large
    intermediate matrices.
    """

    latency_settings = settings or LatencySettings()
    if not inputs:
        raise ValueError("inputs debe contener al menos una entrada.")

    for _ in range(latency_settings.warmup_repetitions):
        for input_value in inputs:
            operation(input_value)

    samples_ms: list[float] = []
    for _ in range(latency_settings.repetitions):
        for input_value in inputs:
            start_time = time.perf_counter_ns()
            operation(input_value)
            elapsed_ns = time.perf_counter_ns() - start_time
            samples_ms.append(elapsed_ns / 1_000_000.0)

    sample_array = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(sample_array.mean())
    operations_per_second = float("inf") if mean_ms == 0 else 1_000.0 / mean_ms
    return LatencySummary(
        samples_ms=tuple(samples_ms),
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(sample_array, 50)),
        p95_ms=float(np.percentile(sample_array, 95)),
        minimum_ms=float(sample_array.min()),
        maximum_ms=float(sample_array.max()),
        operations_per_second=operations_per_second,
    )
