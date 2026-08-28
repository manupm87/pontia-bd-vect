"""Polling, resource safety, and JSON report persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from time import monotonic, sleep

from .contracts import ProviderRun

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_PREFIX = "bbdd-vectoriales-s03"


def validate_resource_name(resource_name: str) -> str:
    """Reject administrative operations outside this teaching session."""
    normalized = re.sub(r"[^a-z0-9]", "", resource_name.lower())
    expected = re.sub(r"[^a-z0-9]", "", RESOURCE_PREFIX)
    if not normalized.startswith(expected):
        raise ValueError(
            f"Resource {resource_name!r} is outside protected prefix {RESOURCE_PREFIX!r}"
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
        raise ValueError("Polling durations must be non-negative and timeout positive")
    started_at = monotonic()
    attempts = 0
    last_value: Value
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


def provider_report_path(provider: str) -> Path:
    """Return the canonical artifact path after validating the provider slug."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", provider):
        raise ValueError(f"Invalid provider slug: {provider!r}")
    return PROJECT_ROOT / ".artifacts" / "provider_runs" / f"{provider}.json"


def write_provider_run(run: ProviderRun) -> Path:
    """Atomically enough for notebooks: write one complete, indented JSON report."""
    validate_resource_name(run.resource)
    if run.record_count < 0:
        raise ValueError("record_count cannot be negative")
    path = provider_report_path(run.provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
