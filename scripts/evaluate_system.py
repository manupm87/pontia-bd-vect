"""Evaluate the final configuration end to end and write metricas_desarrollo.json."""

from __future__ import annotations

import json
import os
import platform
from importlib.metadata import version as package_version

import numpy as np
from dotenv import load_dotenv
from ingest_catalog import build_store

from aurum_discovery.config import (
    ARTIFACTS_DIRECTORY,
    PROJECT_ROOT,
    RESULTS_DIRECTORY,
    load_run_config,
)
from aurum_discovery.contracts import EvaluationRun
from aurum_discovery.data import load_development_judgments, load_development_queries
from aurum_discovery.embeddings import (
    SET_DEVELOPMENT_QUERIES,
    SET_EVALUATION_QUERIES,
    load_embedding_set,
)
from aurum_discovery.evaluation import (
    evaluate_query,
    macro_average,
    measure_latency,
    overlap_at_k,
)
from aurum_discovery.operations import write_json_artifact


def main() -> None:
    """Measure ranking quality, ANN fidelity, and latency on one command."""
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_run_config()
    embedding_set = load_embedding_set(config.embedding_configuration)
    store = build_store(os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo"))
    store.ping()
    record_count = store.count()

    queries = load_development_queries()
    judgments = load_development_judgments()
    development_matrix = embedding_set.matrix(SET_DEVELOPMENT_QUERIES)
    development_ids = embedding_set.identifiers[SET_DEVELOPMENT_QUERIES]

    per_query = []
    development_rankings: dict[str, list[str]] = {}
    for _, query in queries.iterrows():
        position = development_ids.index(query["workload_id"])
        hits = store.search(development_matrix[position], top_k=config.top_k)
        ranked_ids = [hit.product_id for hit in hits]
        development_rankings[query["workload_id"]] = ranked_ids
        per_query.append(
            evaluate_query(
                query["query_id"],
                ranked_ids,
                judgments,
                k=config.top_k,
                recall_relevant_labels=config.recall_relevant_labels,
                mrr_relevant_labels=config.mrr_relevant_labels,
            )
        )
    aggregates = macro_average(per_query)

    fidelity_vectors = np.concatenate(
        [development_matrix, embedding_set.matrix(SET_EVALUATION_QUERIES)]
    )
    fidelity_ids = development_ids + embedding_set.identifiers[SET_EVALUATION_QUERIES]
    overlaps = {}
    for row_position, query_id in enumerate(fidelity_ids):
        exact_hits = store.search(
            fidelity_vectors[row_position], top_k=config.fidelity_k, exact=True
        )
        approximate_hits = store.search(
            fidelity_vectors[row_position], top_k=config.fidelity_k
        )
        overlaps[query_id] = overlap_at_k(
            [hit.record_id for hit in exact_hits],
            [hit.record_id for hit in approximate_hits],
            k=config.fidelity_k,
        )
    fidelity = {
        "k": config.fidelity_k,
        "query_count": len(overlaps),
        "mean_overlap": float(np.mean(list(overlaps.values()))),
        "min_overlap": float(np.min(list(overlaps.values()))),
        "per_query": overlaps,
        "oracle": "misma colección con búsqueda exhaustiva (SearchParams.exact)",
    }

    latency_matrix = embedding_set.matrix(SET_EVALUATION_QUERIES)
    operations = [
        (lambda vector=latency_matrix[row]: store.search(vector, top_k=config.top_k))
        for row in range(latency_matrix.shape[0])
    ]
    latency = measure_latency(
        operations, warmup=config.latency_warmup, repeats=config.latency_repeats
    )

    duplicates_block: dict[str, object] = {}
    calibration_path = ARTIFACTS_DIRECTORY / "duplicados" / "calibracion.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        configured = calibration["configured_rule"]
        duplicates_block = {
            "precision": configured["precision"],
            "recall": configured["recall"],
            "f1": configured["f1"],
            "score_threshold": config.duplicate_score_threshold,
            "margin_threshold": config.duplicate_margin_threshold,
        }

    run = EvaluationRun(
        snapshot_id=config.snapshot_id,
        embedding_configuration=config.embedding_configuration,
        collection=store.collection_name,
        record_count=record_count,
        score_kind=store.score_kind,
        higher_is_better=store.higher_is_better,
        top_k=config.top_k,
        per_query=per_query,
        ndcg_at_10=aggregates["ndcg_at_10"],
        recall_at_10=aggregates["recall_at_10"],
        mrr_at_10=aggregates["mrr_at_10"],
        recall_relevant_labels=list(config.recall_relevant_labels),
        mrr_relevant_labels=list(config.mrr_relevant_labels),
        ann_fidelity=fidelity,
        latency=latency.as_dict(),
        duplicates=duplicates_block,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "qdrant_client": package_version("qdrant-client"),
            "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "hnsw": {"m": config.hnsw.m, "ef_construct": config.hnsw.ef_construct},
            "ef_search": config.ef_search,
        },
        notes=[
            "Las latencias describen esta ejecución local y no comparan proveedores.",
            "development_rankings conserva los IDs recuperados por consulta.",
        ],
    )
    run_payload = run.as_dict()
    run_payload["development_rankings"] = development_rankings
    run_path = write_json_artifact(
        run_payload, ARTIFACTS_DIRECTORY / "evaluacion" / "evaluation_run.json"
    )

    metrics = {
        "ndcg_at_10": round(aggregates["ndcg_at_10"], 4),
        "recall_at_10": round(aggregates["recall_at_10"], 4),
        "mrr_at_10": round(aggregates["mrr_at_10"], 4),
        "latency_p50_ms": round(latency.p50_ms, 2),
        "latency_p95_ms": round(latency.p95_ms, 2),
        "recall_relevant_labels": list(config.recall_relevant_labels),
        "mrr_relevant_labels": list(config.mrr_relevant_labels),
        "ann_fidelity_mean_overlap_at_10": round(fidelity["mean_overlap"], 4),
        "duplicados_desarrollo": duplicates_block,
        "embedding_configuration": config.embedding_configuration,
        "collection": store.collection_name,
        "record_count": record_count,
        "latency_protocol": {
            "warmup": config.latency_warmup,
            "repeats": config.latency_repeats,
            "query_count": latency.query_count,
        },
    }
    metrics_path = write_json_artifact(
        metrics, RESULTS_DIRECTORY / "metricas_desarrollo.json"
    )
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "ndcg_at_10",
                    "recall_at_10",
                    "mrr_at_10",
                    "latency_p50_ms",
                    "latency_p95_ms",
                )
            },
            indent=2,
        )
    )
    print(f"Fidelidad ANN media@{config.fidelity_k}: {fidelity['mean_overlap']:.4f}")
    print(f"Métricas escritas en {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Ejecución completa registrada en {run_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
