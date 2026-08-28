"""Small, explicit configuration objects for retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

DEFAULT_RANDOM_SEED = 42
DEFAULT_TOP_K = 5
DEFAULT_DISTANCE_BATCH_SIZE = 8_192


class DistanceMetric(StrEnum):
    """Metrics supported by exact dense retrieval."""

    COSINE = "cosine"
    DOT = "dot"
    L2 = "l2"
    L2_SQUARED = "l2_squared"
    L1 = "l1"
    CHEBYSHEV = "chebyshev"


class TextRole(StrEnum):
    """Role of a text passed to an asymmetric embedding model."""

    QUERY = "query"
    DOCUMENT = "document"


_METRIC_ALIASES = {
    "cos": DistanceMetric.COSINE,
    "cosine_similarity": DistanceMetric.COSINE,
    "inner_product": DistanceMetric.DOT,
    "ip": DistanceMetric.DOT,
    "euclidean": DistanceMetric.L2,
    "l2_sq": DistanceMetric.L2_SQUARED,
    "squared_l2": DistanceMetric.L2_SQUARED,
    "manhattan": DistanceMetric.L1,
    "linf": DistanceMetric.CHEBYSHEV,
}


def coerce_distance_metric(metric: DistanceMetric | str) -> DistanceMetric:
    """Return a canonical metric or raise an error listing valid choices."""

    if isinstance(metric, DistanceMetric):
        return metric
    normalized_metric = str(metric).strip().lower().replace("-", "_")
    try:
        return DistanceMetric(normalized_metric)
    except ValueError:
        try:
            return _METRIC_ALIASES[normalized_metric]
        except KeyError as error:
            choices = ", ".join(item.value for item in DistanceMetric)
            raise ValueError(
                f"Métrica desconocida {metric!r}. Valores válidos: {choices}."
            ) from error


@dataclass(frozen=True, slots=True)
class TfidfSettings:
    """Settings for the lexical TF-IDF baseline."""

    ngram_range: tuple[int, int] = (1, 2)
    min_document_frequency: int | float = 1
    max_document_frequency: int | float = 1.0
    max_features: int | None = None
    lowercase: bool = True
    strip_accents: str | None = "unicode"
    sublinear_term_frequency: bool = True

    def __post_init__(self) -> None:
        minimum_ngram, maximum_ngram = self.ngram_range
        if (
            isinstance(minimum_ngram, bool)
            or isinstance(maximum_ngram, bool)
            or not isinstance(minimum_ngram, int)
            or not isinstance(maximum_ngram, int)
            or minimum_ngram < 1
            or maximum_ngram < minimum_ngram
        ):
            raise ValueError(
                "ngram_range debe contener enteros positivos en orden creciente."
            )
        _validate_document_frequency(
            self.min_document_frequency,
            name="min_document_frequency",
        )
        _validate_document_frequency(
            self.max_document_frequency,
            name="max_document_frequency",
        )
        if type(self.min_document_frequency) is type(self.max_document_frequency) and (
            self.min_document_frequency > self.max_document_frequency
        ):
            raise ValueError(
                "min_document_frequency no puede superar max_document_frequency."
            )
        if self.max_features is not None and (
            isinstance(self.max_features, bool)
            or not isinstance(self.max_features, int)
            or self.max_features < 1
        ):
            raise ValueError("max_features debe ser positivo o None.")
        if self.strip_accents not in {None, "ascii", "unicode"}:
            raise ValueError("strip_accents debe ser None, 'ascii' o 'unicode'.")


def _validate_document_frequency(value: int | float, *, name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} no puede ser booleano.")
    if isinstance(value, int) and value >= 1:
        return
    if isinstance(value, float) and 0.0 < value <= 1.0:
        return
    raise ValueError(
        f"{name} debe ser un entero >= 1 o una proporción en el intervalo (0, 1]."
    )


@dataclass(frozen=True, slots=True)
class LsaSettings:
    """Settings for TF-IDF followed by truncated singular-value decomposition."""

    dimensions: int = 64
    random_seed: int = DEFAULT_RANDOM_SEED
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions < 1
        ):
            raise ValueError("dimensions debe ser un entero positivo.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed debe ser un entero.")
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError("normalize_embeddings debe ser booleano.")


@dataclass(frozen=True, slots=True)
class LatencySettings:
    """Settings for repeatable, low-overhead latency measurements."""

    repetitions: int = 7
    warmup_repetitions: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions < 1
        ):
            raise ValueError("repetitions debe ser al menos 1.")
        if (
            isinstance(self.warmup_repetitions, bool)
            or not isinstance(self.warmup_repetitions, int)
            or self.warmup_repetitions < 0
        ):
            raise ValueError("warmup_repetitions no puede ser negativo.")
