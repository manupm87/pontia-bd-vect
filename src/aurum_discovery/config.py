"""Project paths, canonical constants, and the final run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "resources" / "actividad_evaluable" / "datos"
EMBEDDINGS_DIRECTORY = PROJECT_ROOT / "data" / "embeddings"
ARTIFACTS_DIRECTORY = PROJECT_ROOT / ".artifacts"
RESULTS_DIRECTORY = PROJECT_ROOT / "resultados"
RUN_CONFIG_PATH = PROJECT_ROOT / "config" / "run_config.yaml"

DEFAULT_RANDOM_SEED = 42
DEFAULT_TOP_K = 10

VALID_DISTANCES = ("cosine",)


@dataclass(frozen=True, slots=True)
class HnswSettings:
    """Explicit HNSW build parameters shared by every collection."""

    m: int
    ef_construct: int

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ValueError("hnsw.m debe ser al menos 2.")
        if self.ef_construct < self.m:
            raise ValueError("hnsw.ef_construct no puede ser menor que hnsw.m.")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Validated snapshot of the configuration used in the final run."""

    snapshot_id: str
    random_seed: int
    embedding_configuration: str
    distance: str
    hnsw: HnswSettings
    ef_search: int
    batch_size: int
    top_k: int
    recall_relevant_labels: tuple[str, ...]
    mrr_relevant_labels: tuple[str, ...]
    fidelity_k: int
    latency_warmup: int
    latency_repeats: int
    duplicate_score_threshold: float
    duplicate_margin_threshold: float

    def __post_init__(self) -> None:
        if self.distance not in VALID_DISTANCES:
            choices = ", ".join(VALID_DISTANCES)
            raise ValueError(
                f"Distancia desconocida {self.distance!r}. Valores válidos: {choices}."
            )
        if self.top_k < 1 or self.fidelity_k < 1:
            raise ValueError("top_k y fidelity.k deben ser positivos.")
        if self.batch_size < 1:
            raise ValueError("collection.batch_size debe ser positivo.")
        if self.ef_search < self.top_k:
            raise ValueError("collection.ef_search no puede ser menor que top_k.")
        if self.latency_warmup < 0 or self.latency_repeats < 1:
            raise ValueError(
                "latency.warmup debe ser >= 0 y latency.repeats debe ser >= 1."
            )
        if not self.recall_relevant_labels or not self.mrr_relevant_labels:
            raise ValueError(
                "Las etiquetas relevantes de recall y MRR deben declararse."
            )
        if not 0.0 <= self.duplicate_score_threshold <= 1.0:
            raise ValueError("duplicates.score_threshold debe estar en [0, 1].")
        if self.duplicate_margin_threshold < 0.0:
            raise ValueError("duplicates.margin_threshold no puede ser negativo.")


def _require_mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"La sección {name!r} de run_config.yaml debe ser un mapa.")
    return payload


def load_run_config(path: Path | None = None) -> RunConfig:
    """Load and validate the run configuration used by every pipeline script."""
    config_path = RUN_CONFIG_PATH if path is None else Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No existe el fichero de configuración {config_path}. "
            "La ejecución final requiere config/run_config.yaml."
        )
    raw = _require_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "raíz"
    )
    embedding = _require_mapping(raw.get("embedding"), "embedding")
    collection = _require_mapping(raw.get("collection"), "collection")
    hnsw = _require_mapping(collection.get("hnsw"), "collection.hnsw")
    retrieval = _require_mapping(raw.get("retrieval"), "retrieval")
    metrics = _require_mapping(raw.get("metrics"), "metrics")
    fidelity = _require_mapping(raw.get("fidelity"), "fidelity")
    latency = _require_mapping(raw.get("latency"), "latency")
    duplicates = _require_mapping(raw.get("duplicates"), "duplicates")
    return RunConfig(
        snapshot_id=str(raw.get("snapshot_id", "")),
        random_seed=int(raw.get("random_seed", DEFAULT_RANDOM_SEED)),
        embedding_configuration=str(embedding.get("configuration", "")),
        distance=str(collection.get("distance", "cosine")),
        hnsw=HnswSettings(
            m=int(hnsw.get("m", 0)), ef_construct=int(hnsw.get("ef_construct", 0))
        ),
        ef_search=int(collection.get("ef_search", 0)),
        batch_size=int(collection.get("batch_size", 0)),
        top_k=int(retrieval.get("top_k", DEFAULT_TOP_K)),
        recall_relevant_labels=tuple(metrics.get("recall_relevant_labels", ())),
        mrr_relevant_labels=tuple(metrics.get("mrr_relevant_labels", ())),
        fidelity_k=int(fidelity.get("k", DEFAULT_TOP_K)),
        latency_warmup=int(latency.get("warmup", 0)),
        latency_repeats=int(latency.get("repeats", 0)),
        duplicate_score_threshold=float(duplicates.get("score_threshold", -1.0)),
        duplicate_margin_threshold=float(duplicates.get("margin_threshold", -1.0)),
    )


__all__ = [
    "ARTIFACTS_DIRECTORY",
    "DATA_DIRECTORY",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TOP_K",
    "EMBEDDINGS_DIRECTORY",
    "PROJECT_ROOT",
    "RESULTS_DIRECTORY",
    "RUN_CONFIG_PATH",
    "HnswSettings",
    "RunConfig",
    "load_run_config",
]
