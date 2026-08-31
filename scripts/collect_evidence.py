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

# Regenerables solo con la descarga única del parquet público de ESCI
# (`make validate-challengers`); su ausencia no invalida `make pipeline`.
OPTIONAL_EVIDENCE_FILES = (
    "experimentos/validacion_ampliada.json",
    "experimentos/tabla_validacion_ampliada.csv",
)


def main() -> None:
    """Snapshot the regenerable evidence so the delivery is self-contained."""
    destination_root = RESULTS_DIRECTORY / "evidencia"
    missing = []
    copied = 0
    for relative in EVIDENCE_FILES + OPTIONAL_EVIDENCE_FILES:
        source = ARTIFACTS_DIRECTORY / relative
        if not source.exists():
            if relative in OPTIONAL_EVIDENCE_FILES:
                print(
                    f"Aviso: falta {relative} (opcional; se regenera con "
                    "`make validate-challengers`). Se conserva el entregado."
                )
                continue
            missing.append(relative)
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos de evidencia; ejecuta `make pipeline` completo "
            f"antes de recolectar: {missing}."
        )
    print(
        f"{copied} artefactos copiados a {destination_root.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
