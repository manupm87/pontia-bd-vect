"""Generate resultados_busqueda.csv and verify the brand-filtered queries."""

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
from aurum_discovery.data import load_evaluation_queries, load_filtered_queries
from aurum_discovery.embeddings import (
    SET_EVALUATION_QUERIES,
    SET_FILTERED_QUERIES,
    load_embedding_set,
)
from aurum_discovery.operations import write_json_artifact, write_search_results

MISSING_BRAND_PROBE = "MarcaInexistenteAurum"


def main() -> None:
    """Run the blind top-10 and the four server-side filtered searches."""
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_run_config()
    embedding_set = load_embedding_set(config.embedding_configuration)
    store = build_store(os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo"))
    store.ping()

    evaluation_queries = load_evaluation_queries()
    matrix = embedding_set.matrix(SET_EVALUATION_QUERIES)
    identifiers = embedding_set.identifiers[SET_EVALUATION_QUERIES]
    rankings = {}
    for _, query in evaluation_queries.iterrows():
        position = identifiers.index(query["evaluation_id"])
        rankings[query["evaluation_id"]] = store.search(
            matrix[position], top_k=config.top_k
        )
    results_path = write_search_results(
        rankings, RESULTS_DIRECTORY / "resultados_busqueda.csv", top_k=config.top_k
    )

    filtered_queries = load_filtered_queries()
    filtered_matrix = embedding_set.matrix(SET_FILTERED_QUERIES)
    filtered_identifiers = embedding_set.identifiers[SET_FILTERED_QUERIES]
    filtered_report = []
    for _, query in filtered_queries.iterrows():
        position = filtered_identifiers.index(query["workload_id"])
        hits = store.search(
            filtered_matrix[position],
            top_k=config.top_k,
            brand=query["filter_value"],
        )
        brands = sorted({hit.brand for hit in hits})
        all_match = brands == [query["filter_value"]] if hits else False
        if not all_match:
            raise RuntimeError(
                f"El filtro de {query['workload_id']} devolvió marcas inesperadas: "
                f"{brands}."
            )
        filtered_report.append(
            {
                "workload_id": query["workload_id"],
                "query_text": query["query_text"],
                "filter": {
                    "field": "brand",
                    "operator": "equals",
                    "value": query["filter_value"],
                },
                "result_count": len(hits),
                "all_results_match_brand": all_match,
                "hits": [hit.as_dict() for hit in hits],
            }
        )

    empty_filter_hits = store.search(
        filtered_matrix[0], top_k=config.top_k, brand=MISSING_BRAND_PROBE
    )
    filtered_path = write_json_artifact(
        {
            "collection": store.collection_name,
            "embedding_configuration": config.embedding_configuration,
            "queries": filtered_report,
            "empty_filter_probe": {
                "brand": MISSING_BRAND_PROBE,
                "result_count": len(empty_filter_hits),
                "behavior": "lista vacía sin error",
            },
        },
        ARTIFACTS_DIRECTORY / "filtros" / "informe_filtros.json",
    )
    print(f"OK búsqueda ciega: {len(rankings)} consultas x {config.top_k} resultados")
    print(f"OK filtros: {len(filtered_report)} consultas verificadas por marca")
    print(f"Resultados escritos en {results_path.relative_to(PROJECT_ROOT)}")
    print(f"Informe de filtros escrito en {filtered_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
