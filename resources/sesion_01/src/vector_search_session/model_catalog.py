"""Carga tipada del catálogo docente de modelos y técnicas.

El JSON asociado es una fotografía fechada, no un leaderboard. Este módulo
mantiene la información consultable desde notebooks sin añadir dependencias
como pandas o Pydantic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

CATALOG_SCHEMA_VERSION = 1
CATALOG_SNAPSHOT_DATE = date(2026, 7, 11)
DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "model_catalog.json"
)

VALID_ACCESS_TYPES = frozenset({"api", "open_weight"})
VALID_VECTOR_OUTPUTS = frozenset(
    {
        "single-vector",
        "multi-vector",
        "sparse",
        "hybrid-multi-output",
    }
)


class CatalogValidationError(ValueError):
    """El catálogo no cumple su contrato mínimo."""


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Fuente primaria que respalda una ficha."""

    title: str
    url: str
    kind: str


@dataclass(frozen=True, slots=True)
class DimensionSpec:
    """Dimensionalidad y política de reducción de un modelo."""

    default: int | None
    minimum: int | None
    maximum: int | None
    options: tuple[int, ...]
    matryoshka: bool
    reduction: str
    normalization: str

    @property
    def label(self) -> str:
        """Representación corta, apropiada para una tabla."""

        if self.options:
            return ", ".join(str(value) for value in self.options)
        if self.minimum is not None and self.maximum is not None:
            return f"{self.minimum}-{self.maximum}"
        if self.default is not None:
            return str(self.default)
        return "variable"


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """Ficha normalizada de un modelo API u open-weight."""

    identifier: str
    display_name: str
    provider: str
    access: str
    status: str
    recommended_for_new_projects: bool
    modalities: tuple[str, ...]
    vector_output: str
    context_tokens: int | None
    dimensions: DimensionSpec
    languages: str
    license: str
    summary: str
    caveats: tuple[str, ...]
    sources: tuple[SourceReference, ...]

    def as_table_record(self) -> dict[str, Any]:
        """Convierte la ficha en un registro plano para pandas o Plotly."""

        return {
            "id": self.identifier,
            "modelo": self.display_name,
            "proveedor": self.provider,
            "acceso": self.access,
            "estado": self.status,
            "recomendado": self.recommended_for_new_projects,
            "modalidades": ", ".join(self.modalities),
            "salida": self.vector_output,
            "contexto_tokens": self.context_tokens,
            "dimensiones": self.dimensions.label,
            "matryoshka": self.dimensions.matryoshka,
            "idiomas": self.languages,
            "licencia": self.license,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    """Método o concepto necesario para interpretar el catálogo."""

    identifier: str
    display_name: str
    category: str
    summary: str
    cautions: tuple[str, ...]
    sources: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Benchmark con su alcance y advertencias de comparación."""

    identifier: str
    display_name: str
    scope: str
    primary_metric: str
    volatile_leaderboard: bool
    cautions: tuple[str, ...]
    sources: tuple[SourceReference, ...]


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Catálogo completo y sus metadatos de procedencia."""

    schema_version: int
    snapshot_date: date
    scope: str
    methodology_notes: tuple[str, ...]
    models: tuple[ModelRecord, ...]
    methods: tuple[KnowledgeEntry, ...]
    benchmarks: tuple[BenchmarkRecord, ...]

    def get_model(self, identifier: str) -> ModelRecord:
        """Devuelve un modelo por ID o produce un error con contexto."""

        for model in self.models:
            if model.identifier == identifier:
                return model
        raise KeyError(f"No existe el modelo {identifier!r} en el catálogo.")

    def select_models(
        self,
        *,
        access: str | None = None,
        modality: str | None = None,
        vector_output: str | None = None,
        recommended_only: bool = False,
    ) -> tuple[ModelRecord, ...]:
        """Filtra modelos conservando el orden editorial del snapshot."""

        if access is not None and access not in VALID_ACCESS_TYPES:
            raise ValueError(f"access debe pertenecer a {sorted(VALID_ACCESS_TYPES)}.")
        if vector_output is not None and vector_output not in VALID_VECTOR_OUTPUTS:
            raise ValueError(
                f"vector_output debe pertenecer a {sorted(VALID_VECTOR_OUTPUTS)}."
            )

        selected = self.models
        if access is not None:
            selected = tuple(model for model in selected if model.access == access)
        if modality is not None:
            selected = tuple(
                model for model in selected if modality in model.modalities
            )
        if vector_output is not None:
            selected = tuple(
                model for model in selected if model.vector_output == vector_output
            )
        if recommended_only:
            selected = tuple(
                model for model in selected if model.recommended_for_new_projects
            )
        return selected

    def model_table_records(
        self,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Registros planos de modelos, compatibles con DataFrame."""

        return [model.as_table_record() for model in self.select_models(**filters)]


def load_model_catalog(path: str | Path | None = None) -> ModelCatalog:
    """Carga y valida el snapshot JSON.

    Args:
        path: Ruta alternativa. Si se omite, usa data/model_catalog.json.
    """

    catalog_path = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogValidationError(
            f"No se encontró el catálogo en {catalog_path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise CatalogValidationError(
            f"El catálogo {catalog_path} no contiene JSON válido."
        ) from exc

    root = _require_mapping(raw, "raíz")
    schema_version = _require_integer(root, "schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogValidationError(
            f"schema_version={schema_version}; se esperaba {CATALOG_SCHEMA_VERSION}."
        )

    snapshot_text = _require_string(root, "snapshot_date")
    try:
        snapshot = date.fromisoformat(snapshot_text)
    except ValueError as exc:
        raise CatalogValidationError(
            "snapshot_date debe usar el formato ISO YYYY-MM-DD."
        ) from exc

    models = tuple(
        _parse_model(item, position)
        for position, item in enumerate(
            _require_sequence(root, "models"),
            start=1,
        )
    )
    methods = tuple(
        _parse_method(item, position)
        for position, item in enumerate(
            _require_sequence(root, "methods"),
            start=1,
        )
    )
    benchmarks = tuple(
        _parse_benchmark(item, position)
        for position, item in enumerate(
            _require_sequence(root, "benchmarks"),
            start=1,
        )
    )
    _validate_unique_identifiers(models, methods, benchmarks)

    return ModelCatalog(
        schema_version=schema_version,
        snapshot_date=snapshot,
        scope=_require_string(root, "scope"),
        methodology_notes=_string_tuple(root, "methodology_notes"),
        models=models,
        methods=methods,
        benchmarks=benchmarks,
    )


def _parse_model(raw: Any, position: int) -> ModelRecord:
    value = _require_mapping(raw, f"models[{position}]")
    identifier = _require_string(value, "id")
    access = _require_string(value, "access")
    vector_output = _require_string(value, "vector_output")
    if access not in VALID_ACCESS_TYPES:
        raise CatalogValidationError(f"{identifier}: access={access!r} no es válido.")
    if vector_output not in VALID_VECTOR_OUTPUTS:
        raise CatalogValidationError(
            f"{identifier}: vector_output={vector_output!r} no es válido."
        )

    context_tokens = value.get("context_tokens")
    if context_tokens is not None:
        context_tokens = _positive_integer(
            context_tokens,
            f"{identifier}.context_tokens",
        )

    recommended = value.get("recommended_for_new_projects")
    if not isinstance(recommended, bool):
        raise CatalogValidationError(
            f"{identifier}.recommended_for_new_projects debe ser booleano."
        )

    return ModelRecord(
        identifier=identifier,
        display_name=_require_string(value, "display_name"),
        provider=_require_string(value, "provider"),
        access=access,
        status=_require_string(value, "status"),
        recommended_for_new_projects=recommended,
        modalities=_non_empty_string_tuple(value, "modalities"),
        vector_output=vector_output,
        context_tokens=context_tokens,
        dimensions=_parse_dimensions(
            value.get("dimensions"),
            context=f"{identifier}.dimensions",
        ),
        languages=_require_string(value, "languages"),
        license=_require_string(value, "license"),
        summary=_require_string(value, "summary"),
        caveats=_string_tuple(value, "caveats"),
        sources=_parse_sources(value.get("sources"), identifier),
    )


def _parse_dimensions(raw: Any, *, context: str) -> DimensionSpec:
    value = _require_mapping(raw, context)
    default = _optional_positive_integer(value.get("default"), f"{context}.default")
    minimum = _optional_positive_integer(value.get("minimum"), f"{context}.minimum")
    maximum = _optional_positive_integer(value.get("maximum"), f"{context}.maximum")

    raw_options = value.get("options", [])
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, str):
        raise CatalogValidationError(f"{context}.options debe ser una lista.")
    options = tuple(
        _positive_integer(option, f"{context}.options") for option in raw_options
    )
    if len(set(options)) != len(options):
        raise CatalogValidationError(
            f"{context}.options contiene dimensiones duplicadas."
        )
    if minimum is not None and maximum is not None and minimum > maximum:
        raise CatalogValidationError(f"{context}: minimum no puede superar maximum.")
    if default is not None and minimum is not None and default < minimum:
        raise CatalogValidationError(
            f"{context}: default no puede ser menor que minimum."
        )
    if default is not None and maximum is not None and default > maximum:
        raise CatalogValidationError(f"{context}: default no puede superar maximum.")
    if minimum is not None and any(option < minimum for option in options):
        raise CatalogValidationError(
            f"{context}.options contiene valores menores que minimum."
        )
    if maximum is not None and any(option > maximum for option in options):
        raise CatalogValidationError(
            f"{context}.options contiene valores mayores que maximum."
        )
    if options and default is not None and default not in options:
        raise CatalogValidationError(
            f"{context}.default debe aparecer en options cuando la lista no está vacía."
        )

    matryoshka = value.get("matryoshka")
    if not isinstance(matryoshka, bool):
        raise CatalogValidationError(f"{context}.matryoshka debe ser booleano.")

    return DimensionSpec(
        default=default,
        minimum=minimum,
        maximum=maximum,
        options=options,
        matryoshka=matryoshka,
        reduction=_require_string(value, "reduction"),
        normalization=_require_string(value, "normalization"),
    )


def _parse_method(raw: Any, position: int) -> KnowledgeEntry:
    value = _require_mapping(raw, f"methods[{position}]")
    identifier = _require_string(value, "id")
    return KnowledgeEntry(
        identifier=identifier,
        display_name=_require_string(value, "display_name"),
        category=_require_string(value, "category"),
        summary=_require_string(value, "summary"),
        cautions=_string_tuple(value, "cautions"),
        sources=_parse_sources(value.get("sources"), identifier),
    )


def _parse_benchmark(raw: Any, position: int) -> BenchmarkRecord:
    value = _require_mapping(raw, f"benchmarks[{position}]")
    identifier = _require_string(value, "id")
    volatile = value.get("volatile_leaderboard")
    if not isinstance(volatile, bool):
        raise CatalogValidationError(
            f"{identifier}.volatile_leaderboard debe ser booleano."
        )
    return BenchmarkRecord(
        identifier=identifier,
        display_name=_require_string(value, "display_name"),
        scope=_require_string(value, "scope"),
        primary_metric=_require_string(value, "primary_metric"),
        volatile_leaderboard=volatile,
        cautions=_string_tuple(value, "cautions"),
        sources=_parse_sources(value.get("sources"), identifier),
    )


def _parse_sources(raw: Any, context: str) -> tuple[SourceReference, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        raise CatalogValidationError(f"{context}.sources debe ser una lista no vacía.")

    sources: list[SourceReference] = []
    for position, item in enumerate(raw, start=1):
        value = _require_mapping(item, f"{context}.sources[{position}]")
        url = _require_string(value, "url")
        if not url.startswith("https://"):
            raise CatalogValidationError(
                f"{context}.sources[{position}].url debe usar HTTPS."
            )
        sources.append(
            SourceReference(
                title=_require_string(value, "title"),
                url=url,
                kind=_require_string(value, "kind"),
            )
        )
    return tuple(sources)


def _validate_unique_identifiers(
    models: Sequence[ModelRecord],
    methods: Sequence[KnowledgeEntry],
    benchmarks: Sequence[BenchmarkRecord],
) -> None:
    identifiers = [
        *(item.identifier for item in models),
        *(item.identifier for item in methods),
        *(item.identifier for item in benchmarks),
    ]
    duplicates = sorted(
        identifier
        for identifier in set(identifiers)
        if identifiers.count(identifier) > 1
    )
    if duplicates:
        raise CatalogValidationError(f"Hay identificadores duplicados: {duplicates}.")


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(f"{context} debe ser un objeto JSON.")
    return value


def _require_sequence(
    value: Mapping[str, Any],
    key: str,
) -> Sequence[Any]:
    result = value.get(key)
    if not isinstance(result, Sequence) or isinstance(result, str):
        raise CatalogValidationError(f"{key} debe ser una lista.")
    return result


def _require_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise CatalogValidationError(f"{key} debe ser un string no vacío.")
    return result.strip()


def _string_tuple(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = _require_sequence(value, key)
    result: list[str] = []
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, str) or not item.strip():
            raise CatalogValidationError(
                f"{key}[{position}] debe ser un string no vacío."
            )
        result.append(item.strip())
    return tuple(result)


def _non_empty_string_tuple(
    value: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    result = _string_tuple(value, key)
    if not result:
        raise CatalogValidationError(f"{key} debe ser una lista no vacía.")
    return result


def _require_integer(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise CatalogValidationError(f"{key} debe ser un entero.")
    return result


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatalogValidationError(f"{context} debe ser un entero positivo.")
    return value


def _optional_positive_integer(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, context)


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CATALOG_SNAPSHOT_DATE",
    "DEFAULT_CATALOG_PATH",
    "BenchmarkRecord",
    "CatalogValidationError",
    "DimensionSpec",
    "KnowledgeEntry",
    "ModelCatalog",
    "ModelRecord",
    "SourceReference",
    "load_model_catalog",
]
