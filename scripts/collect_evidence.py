"""Copy the final-run evidence artifacts into resultados/evidencia/ (versioned)."""

from __future__ import annotations

import shutil

from aurum_discovery.config import ARTIFACTS_DIRECTORY, PROJECT_ROOT, RESULTS_DIRECTORY

EVIDENCE_FILES = (
    "experimentos/registro_experimentos.json",
    "experimentos/tabla_comparativa.csv",
    "evaluacion/evaluation_run.json",
    "evaluacion/barrido_ef_search.json",
    "duplicados/calibracion.json",
    "duplicados/decisiones_evaluacion.json",
    "filtros/informe_filtros.json",
    "eventos/informe_eventos.json",
    "ingesta/informe_ingesta.json",
)


def main() -> None:
    """Snapshot the regenerable evidence so the delivery is self-contained."""
    destination_root = RESULTS_DIRECTORY / "evidencia"
    missing = []
    for relative in EVIDENCE_FILES:
        source = ARTIFACTS_DIRECTORY / relative
        if not source.exists():
            missing.append(relative)
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos de evidencia; ejecuta `make pipeline` completo "
            f"antes de recolectar: {missing}."
        )
    print(
        f"{len(EVIDENCE_FILES)} artefactos copiados a "
        f"{destination_root.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
