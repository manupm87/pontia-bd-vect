"""Compare representations on development queries against an exact oracle.

The experiments isolate representation quality from index behavior: every
dense configuration is ranked with exhaustive cosine search over its own
embedding matrix, and the lexical baseline ranks with BM25. The winning
configuration is then fixed in config/run_config.yaml.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from aurum_discovery.config import (
    ARTIFACTS_DIRECTORY,
    DEFAULT_TOP_K,
    PROJECT_ROOT,
    load_run_config,
)
from aurum_discovery.data import (
    load_catalog,
    load_development_judgments,
    load_development_queries,
)
from aurum_discovery.embeddings import (
    EMBEDDING_CONFIGURATIONS,
    SET_DEVELOPMENT_QUERIES,
    SET_PRODUCTS,
    compose_document_text,
    load_embedding_set,
)
from aurum_discovery.evaluation import evaluate_query, macro_average
from aurum_discovery.lexical import Bm25Index
from aurum_discovery.operations import write_json_artifact


def exact_top_k_positions(
    product_matrix: np.ndarray, query_vector: np.ndarray, *, k: int
) -> list[int]:
    """Exhaustive cosine top-k over L2-normalized embeddings."""
    scores = product_matrix @ query_vector
    partition = np.argpartition(-scores, k - 1)[:k]
    ordered = partition[np.argsort(-scores[partition], kind="stable")]
    return [int(position) for position in ordered]


def evaluate_ranking_set(
    experiment_name: str,
    rankings: dict[str, list[str]],
    judgments: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    config_recall_labels: tuple[str, ...],
    config_mrr_labels: tuple[str, ...],
    extra: dict[str, object],
) -> dict[str, object]:
    """Evaluate one experiment keeping configuration, metrics, and IDs."""
    per_query = []
    for _, query in queries.iterrows():
        per_query.append(
            evaluate_query(
                query["query_id"],
                rankings[query["workload_id"]],
                judgments,
                k=DEFAULT_TOP_K,
                recall_relevant_labels=config_recall_labels,
                mrr_relevant_labels=config_mrr_labels,
            )
        )
    return {
        "experiment": experiment_name,
        **extra,
        "metrics": macro_average(per_query),
        "per_query": [metrics.as_record() for metrics in per_query],
        "retrieved_ids": rankings,
    }


def main() -> None:
    """Run the lexical baseline and every dense configuration on development."""
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_run_config()
    catalog = load_catalog()
    queries = load_development_queries()
    judgments = load_development_judgments()
    product_ids = catalog["product_id"].tolist()
    experiments: list[dict[str, object]] = []

    baseline_started = perf_counter()
    corpus = [
        compose_document_text(row, composition="full_text")
        for _, row in catalog.iterrows()
    ]
    bm25 = Bm25Index(corpus)
    bm25_rankings = {
        query["workload_id"]: [
            product_ids[position]
            for position, _ in bm25.search(query["query_text"], top_k=DEFAULT_TOP_K)
        ]
        for _, query in queries.iterrows()
    }
    experiments.append(
        evaluate_ranking_set(
            "bm25_full_text",
            bm25_rankings,
            judgments,
            queries,
            config_recall_labels=config.recall_relevant_labels,
            config_mrr_labels=config.mrr_relevant_labels,
            extra={
                "kind": "lexical_baseline",
                "parameters": {"k1": 1.5, "b": 0.75},
                "seconds": round(perf_counter() - baseline_started, 2),
            },
        )
    )

    for name in sorted(EMBEDDING_CONFIGURATIONS):
        dense_started = perf_counter()
        embedding_set = load_embedding_set(name)
        product_matrix = embedding_set.matrix(SET_PRODUCTS)
        query_matrix = embedding_set.matrix(SET_DEVELOPMENT_QUERIES)
        query_ids = embedding_set.identifiers[SET_DEVELOPMENT_QUERIES]
        rankings = {}
        for row_position, workload_id in enumerate(query_ids):
            positions = exact_top_k_positions(
                product_matrix, query_matrix[row_position], k=DEFAULT_TOP_K
            )
            rankings[workload_id] = [product_ids[position] for position in positions]
        configuration = embedding_set.configuration
        experiments.append(
            evaluate_ranking_set(
                name,
                rankings,
                judgments,
                queries,
                config_recall_labels=config.recall_relevant_labels,
                config_mrr_labels=config.mrr_relevant_labels,
                extra={
                    "kind": "dense_exact",
                    "parameters": {
                        "model_id": configuration.model_id,
                        "composition": configuration.composition,
                        "document_prefix": configuration.document_prefix,
                        "query_prefix": configuration.query_prefix,
                        "normalized": configuration.normalize,
                        "dimension": configuration.dimension,
                    },
                    "seconds": round(perf_counter() - dense_started, 2),
                },
            )
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "k": DEFAULT_TOP_K,
        "recall_relevant_labels": list(config.recall_relevant_labels),
        "mrr_relevant_labels": list(config.mrr_relevant_labels),
        "experiments": experiments,
    }
    path = write_json_artifact(
        report, ARTIFACTS_DIRECTORY / "experimentos" / "registro_experimentos.json"
    )

    table = pd.DataFrame(
        [
            {"experiment": experiment["experiment"], **experiment["metrics"]}
            for experiment in experiments
        ]
    )
    table_path = ARTIFACTS_DIRECTORY / "experimentos" / "tabla_comparativa.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Registro escrito en {path.relative_to(PROJECT_ROOT)}")
    print(f"Tabla escrita en {table_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
