"""Resource safety, visibility polling, and validated artifact persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from .contracts import DuplicateDecision, SearchHit

RESOURCE_PREFIX = "aurum-market-eval"

_PROVIDER_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def validate_resource_name(resource_name: str) -> str:
    """Refuse to operate on any resource outside the activity's namespace."""
    normalized = re.sub(r"[^a-z0-9]", "", resource_name.lower())
    expected = re.sub(r"[^a-z0-9]", "", RESOURCE_PREFIX)
    if not normalized.startswith(expected):
        raise ValueError(
            f"El recurso {resource_name!r} no pertenece a la actividad: debe "
            f"empezar por el prefijo {RESOURCE_PREFIX!r}."
        )
    return resource_name


def wait_until[Value](
    probe: Callable[[], Value],
    *,
    accept: Callable[[Value], bool] = bool,
    timeout_seconds: float = 60.0,
    interval_seconds: float = 0.5,
    description: str = "condition",
) -> tuple[Value, float, int]:
    """Poll an eventually visible state with a bounded deadline."""
    if timeout_seconds <= 0 or interval_seconds < 0:
        raise ValueError(
            "El timeout debe ser positivo y el intervalo no puede ser negativo."
        )
    started_at = monotonic()
    attempts = 0
    while True:
        attempts += 1
        last_value = probe()
        elapsed = monotonic() - started_at
        if accept(last_value):
            return last_value, elapsed, attempts
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for {description} after {elapsed:.2f}s; "
                f"last value={last_value!r}"
            )
        sleep(interval_seconds)


def sha256_file(path: Path) -> str:
    """Return the streamed SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_artifact(payload: dict[str, Any], path: Path) -> Path:
    """Persist a JSON artifact with UTF-8 text and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def write_search_results(
    rankings: dict[str, Sequence[SearchHit]], path: Path, *, top_k: int
) -> Path:
    """Write resultados_busqueda.csv enforcing the delivery contract.

    Every blind query must contribute exactly `top_k` unique product_ids,
    ranked 1..top_k.
    """
    if not rankings:
        raise ValueError("No hay rankings que escribir.")
    rows: list[dict[str, object]] = []
    for evaluation_id, hits in rankings.items():
        if len(hits) != top_k:
            raise ValueError(
                f"La consulta {evaluation_id!r} tiene {len(hits)} resultados; "
                f"el contrato exige exactamente {top_k}."
            )
        product_ids = [hit.product_id for hit in hits]
        if len(set(product_ids)) != top_k:
            raise ValueError(
                f"La consulta {evaluation_id!r} repite product_id en su top-{top_k}."
            )
        for expected_rank, hit in enumerate(hits, start=1):
            if hit.rank != expected_rank:
                raise ValueError(
                    f"Ranking no consecutivo en {evaluation_id!r}: posición "
                    f"{hit.rank} donde se esperaba {expected_rank}."
                )
            rows.append(
                {
                    "evaluation_id": evaluation_id,
                    "rank": hit.rank,
                    "product_id": hit.product_id,
                    "score": f"{hit.native_score:.6f}",
                }
            )
    return _write_csv(rows, path, ("evaluation_id", "rank", "product_id", "score"))


def write_duplicate_results(decisions: Sequence[DuplicateDecision], path: Path) -> Path:
    """Write resultados_duplicados.csv enforcing the delivery contract."""
    if not decisions:
        raise ValueError("No hay decisiones de duplicados que escribir.")
    incoming_ids = [decision.incoming_id for decision in decisions]
    if len(set(incoming_ids)) != len(incoming_ids):
        raise ValueError("Hay incoming_id repetidos en las decisiones.")
    rows = []
    for decision in decisions:
        row = decision.as_result_row()
        row["predicted_duplicate"] = "true" if decision.predicted_duplicate else "false"
        row["score"] = f"{decision.score:.6f}"
        rows.append(row)
    return _write_csv(
        rows,
        path,
        ("incoming_id", "predicted_duplicate", "matched_product_id", "score"),
    )


def _write_csv(
    rows: Sequence[dict[str, object]], path: Path, fieldnames: tuple[str, ...]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def validate_provider_slug(slug: str) -> str:
    """Validate a short lowercase identifier used in artifact file names."""
    if not _PROVIDER_SLUG_PATTERN.match(slug):
        raise ValueError(
            f"Identificador inválido {slug!r}: use minúsculas, dígitos y guiones."
        )
    return slug


__all__ = [
    "RESOURCE_PREFIX",
    "sha256_file",
    "validate_provider_slug",
    "validate_resource_name",
    "wait_until",
    "write_duplicate_results",
    "write_json_artifact",
    "write_search_results",
]
