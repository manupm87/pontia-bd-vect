"""Adaptadores seguros y opcionales para APIs de embeddings.

Este módulo no importa SDK de proveedores al cargarse y nunca inicia una
petición remota por defecto. Cada función exige dos condiciones:

1. consentimiento explícito mediante allow_remote=True;
2. una clave disponible como argumento o variable de entorno.

Los adaptadores devuelven siempre una matriz NumPy float32 bidimensional,
finita y normalizada con L2. Así, el resto del proyecto puede comparar los
resultados con similitud coseno sin depender del formato particular de cada
SDK.

Google fijó la retirada de gemini-embedding-001 para el 14 de julio de 2026.
Se conserva una constante únicamente para producir un error pedagógico claro;
las peticiones nuevas usan exclusivamente gemini-embedding-2.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

OPENAI_MODELS: Mapping[str, int] = {
    "text-embedding-3-small": 1_536,
    "text-embedding-3-large": 3_072,
}
COHERE_MODEL = "embed-v4.0"
COHERE_DIMENSIONS = frozenset({256, 512, 1_024, 1_536})
GOOGLE_MODEL = "gemini-embedding-2"
GOOGLE_RETIRED_MODEL = "gemini-embedding-001"
GOOGLE_RETIRED_DATE = "2026-07-14"
GOOGLE_MIN_DIMENSION = 128
GOOGLE_MAX_DIMENSION = 3_072

SearchInputType = Literal["search_query", "search_document"]
FloatMatrix = NDArray[np.float32]


class EmbeddingProviderError(RuntimeError):
    """Error base de los adaptadores de proveedores."""


class RemoteCallsDisabledError(EmbeddingProviderError):
    """La ejecución no ha autorizado llamadas remotas."""


class MissingApiKeyError(EmbeddingProviderError):
    """No se encontró la credencial que requiere el proveedor."""


class MissingProviderDependencyError(EmbeddingProviderError):
    """El extra opcional del proveedor no está instalado."""


class ProviderRequestError(EmbeddingProviderError):
    """El SDK no pudo completar una petición autorizada."""


class InvalidEmbeddingResponseError(EmbeddingProviderError):
    """La respuesta no contiene una matriz de embeddings utilizable."""


def embed_openai(
    texts: str | Sequence[str],
    *,
    model: str = "text-embedding-3-small",
    dimensions: int | None = None,
    allow_remote: bool = False,
    api_key: str | None = None,
    client: Any | None = None,
) -> FloatMatrix:
    """Genera embeddings de texto mediante la API de OpenAI.

    Args:
        texts: Texto o lote no vacío.
        model: text-embedding-3-small o text-embedding-3-large.
        dimensions: Dimensión solicitada. None conserva la dimensión nativa.
        allow_remote: Opt-in explícito para efectuar la petición.
        api_key: Credencial opcional; en su defecto usa OPENAI_API_KEY.
        client: Cliente compatible inyectado, útil para pruebas sin red.

    Returns:
        Matriz (n_textos, dimensiones) normalizada con L2.
    """

    batch = _validate_texts(texts)
    native_dimension = _validate_openai_model(model)
    requested_dimension = _validate_reduced_dimension(
        dimensions,
        maximum=native_dimension,
        provider_name="OpenAI",
    )
    resolved_key = _authorize_remote_call(
        provider_name="OpenAI",
        environment_variable="OPENAI_API_KEY",
        allow_remote=allow_remote,
        api_key=api_key,
    )
    active_client = (
        client if client is not None else _create_openai_client(resolved_key)
    )

    request_arguments: dict[str, Any] = {
        "input": batch,
        "model": model,
        "encoding_format": "float",
    }
    if requested_dimension is not None:
        request_arguments["dimensions"] = requested_dimension

    response = _perform_request(
        provider_name="OpenAI",
        request=lambda: active_client.embeddings.create(**request_arguments),
    )
    vectors = _extract_openai_vectors(response)
    expected_dimension = requested_dimension or native_dimension
    return normalize_embedding_matrix(
        vectors,
        expected_rows=len(batch),
        expected_dimensions=expected_dimension,
        provider_name="OpenAI",
    )


def embed_cohere(
    texts: str | Sequence[str],
    *,
    input_type: SearchInputType,
    output_dimension: int = 1_536,
    model: str = COHERE_MODEL,
    allow_remote: bool = False,
    api_key: str | None = None,
    client: Any | None = None,
) -> FloatMatrix:
    """Genera embeddings de consulta o documento con Cohere Embed 4.

    search_query debe emplearse para consultas y search_document para
    elementos que se indexarán. La función solicita explícitamente embeddings
    de tipo float a ClientV2.
    """

    batch = _validate_texts(texts, maximum_items=96)
    _validate_search_input_type(input_type)
    if model != COHERE_MODEL:
        raise ValueError(f"Modelo Cohere no admitido: {model!r}. Usa {COHERE_MODEL!r}.")
    _validate_dimension_type(output_dimension, parameter_name="output_dimension")
    if output_dimension not in COHERE_DIMENSIONS:
        allowed = ", ".join(str(value) for value in sorted(COHERE_DIMENSIONS))
        raise ValueError(
            "Cohere Embed 4 solo admite output_dimension en "
            f"{{{allowed}}}; recibido {output_dimension}."
        )

    resolved_key = _authorize_remote_call(
        provider_name="Cohere",
        environment_variable="COHERE_API_KEY",
        allow_remote=allow_remote,
        api_key=api_key,
    )
    active_client = (
        client if client is not None else _create_cohere_client(resolved_key)
    )
    response = _perform_request(
        provider_name="Cohere",
        request=lambda: active_client.embed(
            texts=batch,
            model=model,
            input_type=input_type,
            embedding_types=["float"],
            output_dimension=output_dimension,
        ),
    )
    vectors = _extract_cohere_vectors(response)
    return normalize_embedding_matrix(
        vectors,
        expected_rows=len(batch),
        expected_dimensions=output_dimension,
        provider_name="Cohere",
    )


def embed_google(
    texts: str | Sequence[str],
    *,
    input_type: SearchInputType,
    titles: Sequence[str | None] | None = None,
    output_dimension: int = 3_072,
    model: str = GOOGLE_MODEL,
    allow_remote: bool = False,
    api_key: str | None = None,
    client: Any | None = None,
) -> FloatMatrix:
    """Genera embeddings con el modelo estable gemini-embedding-2.

    Google Embedding 2 ya no acepta task_type. Esta función aplica los
    prefijos oficiales para consulta y documento. Cada texto se envía como un
    objeto Content independiente para obtener un vector por entrada y no un
    único vector agregado.
    """

    batch = _validate_texts(texts)
    _validate_search_input_type(input_type)
    _validate_google_model(model)
    _validate_dimension_type(output_dimension, parameter_name="output_dimension")
    if not GOOGLE_MIN_DIMENSION <= output_dimension <= GOOGLE_MAX_DIMENSION:
        raise ValueError(
            "Google Embedding 2 admite output_dimension entre "
            f"{GOOGLE_MIN_DIMENSION} y {GOOGLE_MAX_DIMENSION}; "
            f"recibido {output_dimension}."
        )

    prepared_texts = prepare_google_inputs(
        batch,
        input_type=input_type,
        titles=titles,
    )
    resolved_key = _authorize_remote_call(
        provider_name="Google",
        environment_variable="GEMINI_API_KEY",
        allow_remote=allow_remote,
        api_key=api_key,
    )
    active_client = (
        client if client is not None else _create_google_client(resolved_key)
    )

    # El SDK google-genai acepta diccionarios para sus tipos Pydantic. Un
    # Content por texto evita la agregación multimodal accidental.
    contents = [
        {"parts": [{"text": prepared_text}]} for prepared_text in prepared_texts
    ]
    response = _perform_request(
        provider_name="Google",
        request=lambda: active_client.models.embed_content(
            model=model,
            contents=contents,
            config={"output_dimensionality": output_dimension},
        ),
    )
    vectors = _extract_google_vectors(response)
    return normalize_embedding_matrix(
        vectors,
        expected_rows=len(batch),
        expected_dimensions=output_dimension,
        provider_name="Google",
    )


def prepare_google_inputs(
    texts: Sequence[str],
    *,
    input_type: SearchInputType,
    titles: Sequence[str | None] | None = None,
) -> list[str]:
    """Aplica el contrato de consulta/documento de Google Embedding 2."""

    _validate_search_input_type(input_type)
    batch = _validate_texts(texts)

    if input_type == "search_query":
        if titles is not None:
            raise ValueError("Las consultas de Google no admiten títulos.")
        return [f"task: search result | query: {text}" for text in batch]

    normalized_titles = _normalize_titles(titles, expected_count=len(batch))
    return [
        f"title: {title} | text: {text}"
        for title, text in zip(normalized_titles, batch, strict=True)
    ]


def normalize_embedding_matrix(
    values: Any,
    *,
    expected_rows: int,
    expected_dimensions: int,
    provider_name: str,
) -> FloatMatrix:
    """Valida y normaliza una respuesta heterogénea de embeddings."""

    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingResponseError(
            f"{provider_name} devolvió valores que no forman una matriz numérica."
        ) from exc

    if matrix.ndim != 2:
        raise InvalidEmbeddingResponseError(
            f"{provider_name} devolvió {matrix.ndim} dimensiones; se esperaba "
            "una matriz bidimensional."
        )
    if matrix.shape != (expected_rows, expected_dimensions):
        raise InvalidEmbeddingResponseError(
            f"{provider_name} devolvió shape={matrix.shape}; se esperaba "
            f"({expected_rows}, {expected_dimensions})."
        )
    if not np.isfinite(matrix).all():
        raise InvalidEmbeddingResponseError(f"{provider_name} devolvió NaN o infinito.")

    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms <= np.finfo(np.float32).tiny):
        raise InvalidEmbeddingResponseError(
            f"{provider_name} devolvió al menos un vector nulo."
        )

    normalized = matrix / norms[:, np.newaxis]
    if not np.isfinite(normalized).all():
        raise InvalidEmbeddingResponseError(
            f"No fue posible normalizar la respuesta de {provider_name}."
        )
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _validate_texts(
    texts: str | Sequence[str],
    *,
    maximum_items: int | None = None,
) -> list[str]:
    if isinstance(texts, str):
        candidates = [texts]
    else:
        if not isinstance(texts, Sequence):
            raise TypeError("texts debe ser un string o una secuencia de strings.")
        candidates = list(texts)

    if not candidates:
        raise ValueError("Debe proporcionarse al menos un texto.")
    if maximum_items is not None and len(candidates) > maximum_items:
        raise ValueError(
            f"Este proveedor admite como máximo {maximum_items} entradas por llamada."
        )

    cleaned: list[str] = []
    for position, value in enumerate(candidates):
        if not isinstance(value, str):
            raise TypeError(f"El texto en la posición {position} no es un string.")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"El texto en la posición {position} está vacío.")
        cleaned.append(stripped)
    return cleaned


def _normalize_titles(
    titles: Sequence[str | None] | None,
    *,
    expected_count: int,
) -> list[str]:
    if titles is None:
        return ["none"] * expected_count
    if isinstance(titles, str):
        raise TypeError("titles debe ser una secuencia, no un único string.")
    if not isinstance(titles, Sequence):
        raise TypeError("titles debe ser una secuencia de strings o None.")

    normalized = list(titles)
    if len(normalized) != expected_count:
        raise ValueError(
            f"Se esperaban {expected_count} títulos y se recibieron {len(normalized)}."
        )

    result: list[str] = []
    for position, title in enumerate(normalized):
        if title is None:
            result.append("none")
            continue
        if not isinstance(title, str):
            raise TypeError(
                f"El título en la posición {position} no es un string ni None."
            )
        result.append(title.strip() or "none")
    return result


def _validate_openai_model(model: str) -> int:
    if model not in OPENAI_MODELS:
        allowed = ", ".join(repr(name) for name in OPENAI_MODELS)
        raise ValueError(f"Modelo OpenAI no admitido: {model!r}. Usa {allowed}.")
    return OPENAI_MODELS[model]


def _validate_reduced_dimension(
    dimensions: int | None,
    *,
    maximum: int,
    provider_name: str,
) -> int | None:
    if dimensions is None:
        return None
    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise TypeError("dimensions debe ser un entero positivo o None.")
    if not 1 <= dimensions <= maximum:
        raise ValueError(
            f"{provider_name} requiere dimensions entre 1 y {maximum}; "
            f"recibido {dimensions}."
        )
    return dimensions


def _validate_search_input_type(input_type: str) -> None:
    if input_type not in {"search_query", "search_document"}:
        raise ValueError("input_type debe ser 'search_query' o 'search_document'.")


def _validate_dimension_type(value: Any, *, parameter_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{parameter_name} debe ser un entero positivo.")


def _validate_google_model(model: str) -> None:
    if model == GOOGLE_RETIRED_MODEL:
        raise ValueError(
            f"Google fijó la retirada de {GOOGLE_RETIRED_MODEL!r} para el "
            f"{GOOGLE_RETIRED_DATE}. Usa el modelo estable {GOOGLE_MODEL!r} "
            "y vuelve a generar el índice."
        )
    if model != GOOGLE_MODEL:
        raise ValueError(f"Modelo Google no admitido: {model!r}. Usa {GOOGLE_MODEL!r}.")


def _authorize_remote_call(
    *,
    provider_name: str,
    environment_variable: str,
    allow_remote: bool,
    api_key: str | None,
) -> str:
    if allow_remote is not True:
        raise RemoteCallsDisabledError(
            f"Las llamadas a {provider_name} están desactivadas. "
            "Pasa allow_remote=True de forma explícita para habilitarlas."
        )

    if api_key is not None and not isinstance(api_key, str):
        raise TypeError("api_key debe ser un string o None.")

    resolved_key = api_key or os.getenv(environment_variable)
    if not resolved_key or not resolved_key.strip():
        raise MissingApiKeyError(
            f"Falta {environment_variable}. Añádela a tu entorno antes de "
            f"habilitar {provider_name}; no incluyas la clave en el código."
        )
    return resolved_key


def _create_openai_client(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MissingProviderDependencyError(
            "Falta el SDK de OpenAI. Instala el extra del proyecto con "
            "'pip install -e .[api]'."
        ) from exc
    return OpenAI(api_key=api_key)


def _create_cohere_client(api_key: str) -> Any:
    try:
        import cohere
    except ImportError as exc:
        raise MissingProviderDependencyError(
            "Falta el SDK de Cohere. Instala el extra del proyecto con "
            "'pip install -e .[api]'."
        ) from exc
    return cohere.ClientV2(api_key=api_key)


def _create_google_client(api_key: str) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise MissingProviderDependencyError(
            "Falta google-genai. Instala el extra del proyecto con "
            "'pip install -e .[api]'."
        ) from exc
    return genai.Client(api_key=api_key)


def _perform_request(
    *,
    provider_name: str,
    request: Callable[[], Any],
) -> Any:
    try:
        return request()
    except EmbeddingProviderError:
        raise
    except Exception as exc:
        raise ProviderRequestError(
            f"La petición autorizada a {provider_name} falló "
            f"({type(exc).__name__}). Revisa conectividad, cuota y permisos."
        ) from exc


def _extract_openai_vectors(response: Any) -> list[Any]:
    data = _read_list_field(response, "data", provider_name="OpenAI")
    raw_indices = [_optional_field(item, "index") for item in data]
    if any(index is not None for index in raw_indices):
        if not all(index is not None for index in raw_indices):
            raise InvalidEmbeddingResponseError(
                "La respuesta de OpenAI contiene índices parciales."
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in raw_indices
        ):
            raise InvalidEmbeddingResponseError(
                "La respuesta de OpenAI contiene índices no enteros."
            )
        if sorted(raw_indices) != list(range(len(data))):
            raise InvalidEmbeddingResponseError(
                "La respuesta de OpenAI contiene índices duplicados o fuera de rango."
            )
        data.sort(key=lambda item: _read_field(item, "index", "OpenAI"))
    return [_read_field(item, "embedding", provider_name="OpenAI") for item in data]


def _extract_cohere_vectors(response: Any) -> Any:
    embeddings = _read_field(response, "embeddings", provider_name="Cohere")
    # Cohere serializa el campo como "float", pero su modelo Pydantic de
    # Python lo expone como float_ para evitar colisionar con el tipo built-in.
    vectors = _optional_field(embeddings, "float")
    if vectors is None:
        vectors = _optional_field(embeddings, "float_")
    if vectors is None:
        raise InvalidEmbeddingResponseError(
            "La respuesta de Cohere no contiene embeddings float."
        )
    return vectors


def _extract_google_vectors(response: Any) -> list[Any]:
    embeddings = _read_list_field(
        response,
        "embeddings",
        provider_name="Google",
    )
    return [
        _read_field(embedding, "values", provider_name="Google")
        for embedding in embeddings
    ]


def _read_field(value: Any, name: str, provider_name: str) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    raise InvalidEmbeddingResponseError(
        f"La respuesta de {provider_name} no contiene el campo {name!r}."
    )


def _optional_field(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _read_list_field(
    value: Any,
    name: str,
    provider_name: str,
) -> list[Any]:
    raw_field = _read_field(value, name, provider_name)
    if isinstance(raw_field, (str, bytes, Mapping)):
        raise InvalidEmbeddingResponseError(
            f"El campo {name!r} de {provider_name} no es una lista."
        )
    try:
        return list(raw_field)
    except TypeError as exc:
        raise InvalidEmbeddingResponseError(
            f"El campo {name!r} de {provider_name} no es iterable."
        ) from exc


__all__ = [
    "COHERE_DIMENSIONS",
    "COHERE_MODEL",
    "GOOGLE_MODEL",
    "GOOGLE_RETIRED_DATE",
    "GOOGLE_RETIRED_MODEL",
    "OPENAI_MODELS",
    "EmbeddingProviderError",
    "InvalidEmbeddingResponseError",
    "MissingApiKeyError",
    "MissingProviderDependencyError",
    "ProviderRequestError",
    "RemoteCallsDisabledError",
    "embed_cohere",
    "embed_google",
    "embed_openai",
    "normalize_embedding_matrix",
    "prepare_google_inputs",
]
