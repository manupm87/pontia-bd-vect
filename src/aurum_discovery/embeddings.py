"""Embedding configurations, text composition, and the persistence contract."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import EMBEDDINGS_DIRECTORY

VALID_COMPOSITIONS = ("full_text", "title_brand_color")

SET_PRODUCTS = "products"
SET_DEVELOPMENT_QUERIES = "consultas_desarrollo"
SET_EVALUATION_QUERIES = "consultas_evaluacion"
SET_FILTERED_QUERIES = "consultas_filtradas"
SET_INCOMING_DEVELOPMENT = "altas_desarrollo"
SET_INCOMING_EVALUATION = "altas_evaluacion"
SET_EVENT_DOCUMENTS = "eventos_upsert"

QUERY_SETS = (
    SET_DEVELOPMENT_QUERIES,
    SET_EVALUATION_QUERIES,
    SET_FILTERED_QUERIES,
)
DOCUMENT_SETS = (
    SET_PRODUCTS,
    SET_INCOMING_DEVELOPMENT,
    SET_INCOMING_EVALUATION,
    SET_EVENT_DOCUMENTS,
)
ALL_SETS = DOCUMENT_SETS + QUERY_SETS


@dataclass(frozen=True, slots=True)
class EmbeddingConfiguration:
    """One reproducible combination of model, text composition, and prefixes."""

    name: str
    model_id: str
    composition: str
    document_prefix: str
    query_prefix: str
    dimension: int
    normalize: bool = True
    provider: str = "local"

    def __post_init__(self) -> None:
        if self.composition not in VALID_COMPOSITIONS:
            choices = ", ".join(VALID_COMPOSITIONS)
            raise ValueError(
                f"Composición desconocida {self.composition!r}. "
                f"Valores válidos: {choices}."
            )
        if self.dimension < 1:
            raise ValueError("dimension debe ser positiva.")
        if self.provider not in ("local", "gemini"):
            raise ValueError(
                f"Proveedor desconocido {self.provider!r}. "
                "Valores válidos: local, gemini."
            )

    @property
    def directory(self) -> Path:
        return EMBEDDINGS_DIRECTORY / self.name


EMBEDDING_CONFIGURATIONS: dict[str, EmbeddingConfiguration] = {
    configuration.name: configuration
    for configuration in (
        EmbeddingConfiguration(
            name="e5_small_full",
            model_id="intfloat/multilingual-e5-small",
            composition="full_text",
            document_prefix="passage: ",
            query_prefix="query: ",
            dimension=384,
        ),
        EmbeddingConfiguration(
            name="e5_small_title",
            model_id="intfloat/multilingual-e5-small",
            composition="title_brand_color",
            document_prefix="passage: ",
            query_prefix="query: ",
            dimension=384,
        ),
        EmbeddingConfiguration(
            name="e5_base_title",
            model_id="intfloat/multilingual-e5-base",
            composition="title_brand_color",
            document_prefix="passage: ",
            query_prefix="query: ",
            dimension=768,
        ),
        EmbeddingConfiguration(
            name="minilm_full",
            model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            composition="full_text",
            document_prefix="",
            query_prefix="",
            dimension=384,
        ),
        # Proveedor API visto en la sesión 01. Solo se construye si existe
        # GEMINI_API_KEY en el entorno; el recorrido evaluado no depende de él.
        EmbeddingConfiguration(
            name="gemini_v2_title",
            model_id="gemini-embedding-2",
            composition="title_brand_color",
            document_prefix="",
            query_prefix="",
            dimension=768,
            provider="gemini",
        ),
    )
}


def get_configuration(name: str) -> EmbeddingConfiguration:
    """Return a registered configuration or raise listing the valid names."""
    if name not in EMBEDDING_CONFIGURATIONS:
        choices = ", ".join(sorted(EMBEDDING_CONFIGURATIONS))
        raise ValueError(
            f"Configuración de embeddings desconocida {name!r}. "
            f"Valores válidos: {choices}."
        )
    return EMBEDDING_CONFIGURATIONS[name]


def compose_document_text(row: pd.Series, *, composition: str) -> str:
    """Return the text encoded for a catalog row, an event, or an incoming record.

    Empty metadata means missing information: the field is omitted instead of
    encoding literal placeholders such as "nan".
    """
    if composition == "full_text":
        text = str(row["text"]).strip()
        if text:
            return text
        composition = "title_brand_color"
    parts = [str(row["title"]).strip()]
    brand = str(row["brand"]).strip()
    color = str(row["color"]).strip()
    if brand:
        parts.append(f"Marca: {brand}")
    if color:
        parts.append(f"Color: {color}")
    composed = ". ".join(part for part in parts if part)
    if not composed:
        raise ValueError("No hay texto que codificar: título y metadatos vacíos.")
    return composed


def load_encoder(model_id: str) -> Any:
    """Load a sentence-transformers model, imported lazily to stay offline-safe."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers no está instalado. Ejecuta "
            "`uv sync --all-extras` o instala el extra 'embeddings'."
        ) from error
    return SentenceTransformer(model_id)


def encode_texts(
    encoder: Any,
    texts: list[str],
    *,
    prefix: str,
    normalize: bool,
    batch_size: int = 128,
) -> NDArray[np.float32]:
    """Encode texts with the given prefix and return a float32 matrix."""
    if not texts:
        raise ValueError("No hay textos que codificar.")
    prefixed = [f"{prefix}{text}" for text in texts]
    matrix = encoder.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(matrix, dtype=np.float32)


def load_gemini_client() -> Any:
    """Create a google-genai client; never called without an explicit key.

    Mirrors the session's offline-by-default philosophy: importing this module
    or running the evaluated pipeline never touches the network. The Gemini
    configuration only builds when GEMINI_API_KEY is present in the
    environment (.env), and skipping it is not an error.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY en .env. La configuración Gemini es opcional: "
            "sin clave se omite y el recorrido evaluado sigue siendo local."
        )
    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError(
            "google-genai no está instalado. Ejecuta `uv sync --all-extras` "
            "(extra 'api')."
        ) from error
    return genai.Client(api_key=api_key)


def encode_texts_gemini(
    client: Any,
    texts: list[str],
    *,
    role: str,
    dimension: int,
    model_id: str = "gemini-embedding-2",
    batch_size: int = 100,
) -> NDArray[np.float32]:
    """Encode texts with Gemini Embedding 2 using the course's I/O contract.

    Queries and documents use the official textual roles (the model no longer
    accepts task_type), and the reduced-dimension output is L2-normalized
    afterwards, as in the session adapter.
    """
    if role not in ("query", "document"):
        raise ValueError(f"Rol desconocido {role!r}. Valores válidos: query, document.")
    if not texts:
        raise ValueError("No hay textos que codificar.")
    prepared = [
        f"task: search result | query: {text}"
        if role == "query"
        else f"title: none | text: {text}"
        for text in texts
    ]
    rows: list[list[float]] = []
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        response = client.models.embed_content(
            model=model_id,
            contents=[{"parts": [{"text": text}]} for text in chunk],
            config={"output_dimensionality": dimension},
        )
        rows.extend(list(embedding.values) for embedding in response.embeddings)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (len(texts), dimension) or not np.isfinite(matrix).all():
        raise ValueError(
            f"Gemini devolvió una matriz inesperada: {matrix.shape}, se esperaba "
            f"({len(texts)}, {dimension})."
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).tiny):
        raise ValueError("Gemini devolvió al menos un vector nulo.")
    return matrix / norms


@dataclass(frozen=True, slots=True)
class EmbeddingSet:
    """Validated bundle of every embedding matrix built for one configuration."""

    configuration: EmbeddingConfiguration
    metadata: dict[str, Any]
    matrices: dict[str, NDArray[np.float32]]
    identifiers: dict[str, list[str]]

    def matrix(self, set_name: str) -> NDArray[np.float32]:
        if set_name not in self.matrices:
            choices = ", ".join(sorted(self.matrices))
            raise ValueError(
                f"Conjunto de embeddings desconocido {set_name!r}. "
                f"Valores válidos: {choices}."
            )
        return self.matrices[set_name]

    def vector(self, set_name: str, identifier: str) -> NDArray[np.float32]:
        """Return the embedding of one identifier within a named set."""
        identifiers = self.identifiers[set_name]
        try:
            position = identifiers.index(identifier)
        except ValueError as error:
            raise KeyError(
                f"El identificador {identifier!r} no existe en {set_name!r}."
            ) from error
        return self.matrix(set_name)[position]


def load_embedding_set(configuration_name: str) -> EmbeddingSet:
    """Load and validate the persisted embeddings of one configuration."""
    configuration = get_configuration(configuration_name)
    directory = configuration.directory
    metadata_path = directory / "embedding_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existen embeddings para {configuration_name!r} en {directory}. "
            "Genera los ficheros con `make embeddings`."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matrices: dict[str, NDArray[np.float32]] = {}
    identifiers: dict[str, list[str]] = {}
    for set_name in ALL_SETS:
        matrix = np.load(directory / f"{set_name}.npy", allow_pickle=False)
        matrices[set_name] = np.asarray(matrix, dtype=np.float32)
        identifiers[set_name] = list(metadata["identifiers"][set_name])
    embedding_set = EmbeddingSet(
        configuration=configuration,
        metadata=metadata,
        matrices=matrices,
        identifiers=identifiers,
    )
    _validate_embedding_set(embedding_set)
    return embedding_set


def _validate_embedding_set(embedding_set: EmbeddingSet) -> None:
    configuration = embedding_set.configuration
    if embedding_set.metadata.get("model_id") != configuration.model_id:
        raise ValueError(
            "Los embeddings persistidos no corresponden al modelo configurado: "
            f"{embedding_set.metadata.get('model_id')!r} frente a "
            f"{configuration.model_id!r}."
        )
    for set_name, matrix in embedding_set.matrices.items():
        expected_rows = len(embedding_set.identifiers[set_name])
        if matrix.ndim != 2 or matrix.shape != (expected_rows, configuration.dimension):
            raise ValueError(
                f"Forma inesperada en {set_name!r}: {matrix.shape}, "
                f"se esperaba ({expected_rows}, {configuration.dimension})."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"El conjunto {set_name!r} contiene NaN o infinito.")
        if configuration.normalize:
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError(f"El conjunto {set_name!r} no está L2-normalizado.")


__all__ = [
    "ALL_SETS",
    "DOCUMENT_SETS",
    "EMBEDDING_CONFIGURATIONS",
    "QUERY_SETS",
    "SET_DEVELOPMENT_QUERIES",
    "SET_EVALUATION_QUERIES",
    "SET_EVENT_DOCUMENTS",
    "SET_FILTERED_QUERIES",
    "SET_INCOMING_DEVELOPMENT",
    "SET_INCOMING_EVALUATION",
    "SET_PRODUCTS",
    "EmbeddingConfiguration",
    "EmbeddingSet",
    "compose_document_text",
    "encode_texts",
    "encode_texts_gemini",
    "get_configuration",
    "load_embedding_set",
    "load_encoder",
    "load_gemini_client",
]
