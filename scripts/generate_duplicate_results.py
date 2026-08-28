"""Generate resultados_duplicados.csv for the blind incoming records."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from ingest_catalog import build_store

from aurum_discovery.config import (
    ARTIFACTS_DIRECTORY,
    PROJECT_ROOT,
    RESULTS_DIRECTORY,
    load_run_config,
)
from aurum_discovery.data import load_incoming_products
from aurum_discovery.duplicates import DuplicateRule
from aurum_discovery.embeddings import SET_INCOMING_EVALUATION, load_embedding_set
from aurum_discovery.operations import write_duplicate_results, write_json_artifact

CANDIDATE_COUNT = 5


def main() -> None:
    """Apply the calibrated rule to altas_evaluacion.csv and write the artifact."""
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_run_config()
    rule = DuplicateRule(
        score_threshold=config.duplicate_score_threshold,
        margin_threshold=config.duplicate_margin_threshold,
    )
    embedding_set = load_embedding_set(config.embedding_configuration)
    incoming = load_incoming_products(labeled=False)
    store = build_store(os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo"))
    store.ping()

    matrix = embedding_set.matrix(SET_INCOMING_EVALUATION)
    identifiers = embedding_set.identifiers[SET_INCOMING_EVALUATION]
    decisions = []
    audit_trail = []
    for _, row in incoming.iterrows():
        position = identifiers.index(row["incoming_id"])
        candidates = store.search(matrix[position], top_k=CANDIDATE_COUNT)
        decision = rule.decide(row["incoming_id"], candidates)
        decisions.append(decision)
        audit_trail.append(
            {
                "incoming_id": row["incoming_id"],
                "decision": decision.as_result_row(),
                "margin": decision.margin,
                "candidates": [hit.as_dict() for hit in candidates[:2]],
            }
        )

    results_path = write_duplicate_results(
        decisions, RESULTS_DIRECTORY / "resultados_duplicados.csv"
    )
    audit_path = write_json_artifact(
        {
            "rule": {
                "score_threshold": rule.score_threshold,
                "margin_threshold": rule.margin_threshold,
            },
            "embedding_configuration": config.embedding_configuration,
            "decisions": audit_trail,
        },
        ARTIFACTS_DIRECTORY / "duplicados" / "decisiones_evaluacion.json",
    )
    positives = sum(decision.predicted_duplicate for decision in decisions)
    print(f"OK duplicados: {positives} positivos de {len(decisions)} altas")
    print(f"Resultados escritos en {results_path.relative_to(PROJECT_ROOT)}")
    print(f"Auditoría escrita en {audit_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
