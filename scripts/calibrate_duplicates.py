"""Calibrate the duplicate-decision rule on the labeled development intakes."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from ingest_catalog import build_store

from aurum_discovery.config import ARTIFACTS_DIRECTORY, PROJECT_ROOT, load_run_config
from aurum_discovery.data import load_incoming_products
from aurum_discovery.duplicates import (
    CalibrationCase,
    DuplicateRule,
    calibrate_rule,
    confusion_metrics,
)
from aurum_discovery.embeddings import SET_INCOMING_DEVELOPMENT, load_embedding_set
from aurum_discovery.operations import write_json_artifact

CANDIDATE_COUNT = 5


def main() -> None:
    """Grid-search thresholds on development and audit the configured rule."""
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_run_config()
    embedding_set = load_embedding_set(config.embedding_configuration)
    incoming = load_incoming_products(labeled=True)
    collection_name = os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo")
    store = build_store(collection_name)
    store.ping()

    matrix = embedding_set.matrix(SET_INCOMING_DEVELOPMENT)
    identifiers = embedding_set.identifiers[SET_INCOMING_DEVELOPMENT]
    cases = []
    for _, row in incoming.iterrows():
        position = identifiers.index(row["incoming_id"])
        candidates = store.search(matrix[position], top_k=CANDIDATE_COUNT)
        cases.append(
            CalibrationCase(
                incoming_id=row["incoming_id"],
                is_duplicate=bool(row["is_duplicate"]),
                reference_product_id=row["reference_product_id"],
                candidates=tuple(candidates),
            )
        )

    calibration = calibrate_rule(cases)
    configured_rule = DuplicateRule(
        score_threshold=config.duplicate_score_threshold,
        margin_threshold=config.duplicate_margin_threshold,
    )
    configured_report = confusion_metrics(cases, configured_rule)

    report = {
        "collection": collection_name,
        "embedding_configuration": config.embedding_configuration,
        "candidate_count": CANDIDATE_COUNT,
        "cases": [
            {
                "incoming_id": case.incoming_id,
                "is_duplicate": case.is_duplicate,
                "reference_product_id": case.reference_product_id,
                "top_candidates": [hit.as_dict() for hit in case.candidates[:2]],
            }
            for case in cases
        ],
        "calibration": calibration,
        "configured_rule": configured_report,
    }
    path = write_json_artifact(
        report, ARTIFACTS_DIRECTORY / "duplicados" / "calibracion.json"
    )

    best_rule = calibration["best"]["rule"]
    print(
        "Mejor regla en desarrollo: "
        f"score >= {best_rule['score_threshold']}, margen >= {best_rule['margin_threshold']} "
        f"(F1 {calibration['best']['f1']:.3f})"
    )
    print(
        "Regla configurada: "
        f"score >= {configured_rule.score_threshold}, margen >= {configured_rule.margin_threshold} "
        f"(F1 {configured_report['f1']:.3f}, precision {configured_report['precision']:.3f}, "
        f"recall {configured_report['recall']:.3f})"
    )
    if configured_report["f1"] + 1e-9 < calibration["best"]["f1"]:
        print(
            "AVISO: la regla configurada en config/run_config.yaml rinde por "
            "debajo de la óptima calibrada; revisa el fichero antes de la "
            "ejecución final."
        )
    print(f"Informe escrito en {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
