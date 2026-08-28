"""Contratos comunes para los cinco motores de la sesión 3."""

from .contracts import ProviderRun, SearchHit, VectorRecord
from .data import (
    RECORD_NAMESPACE,
    SessionData,
    iter_record_batches,
    load_session_data,
    record_id_for_product,
)
from .evaluation import evaluate_run, exact_top_k
from .operations import (
    RESOURCE_PREFIX,
    provider_report_path,
    validate_resource_name,
    wait_until,
    write_provider_run,
)

__all__ = [
    "RECORD_NAMESPACE",
    "RESOURCE_PREFIX",
    "ProviderRun",
    "SearchHit",
    "SessionData",
    "VectorRecord",
    "evaluate_run",
    "exact_top_k",
    "iter_record_batches",
    "load_session_data",
    "provider_report_path",
    "record_id_for_product",
    "validate_resource_name",
    "wait_until",
    "write_provider_run",
]
