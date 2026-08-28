"""Sweep ef_search measuring ANN fidelity against the exact oracle.

Two graphs are measured: the delivered configuration (m=24, fully optimized,
where fidelity stays at 1.0 across the whole sweep at this scale) and a
deliberately weak graph (m=4) on a dedicated collection, where the
fidelity/ef_search trade-off becomes visible and index losses can be
demonstrated reproducibly.
"""

from __future__ import annotations

import argparse
import os
from statistics import median
from time import perf_counter

import numpy as np
from dotenv import load_dotenv

from aurum_discovery.config import (
    ARTIFACTS_DIRECTORY,
    PROJECT_ROOT,
    HnswSettings,
    load_run_config,
)
from aurum_discovery.data import iter_record_batches, load_catalog
from aurum_discovery.embeddings import (
    SET_DEVELOPMENT_QUERIES,
    SET_EVALUATION_QUERIES,
    SET_PRODUCTS,
    load_embedding_set,
)
from aurum_discovery.operations import write_json_artifact
from aurum_discovery.vector_store import CatalogVectorStore

EF_VALUES = (10, 16, 24, 32, 64, 128)
WEAK_COLLECTION = "aurum-market-eval-hnsw-debil"
WEAK_HNSW = HnswSettings(m=4, ef_construct=16)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def build_store(
    collection_name: str, ef_search: int, *, hnsw: HnswSettings, vector_size: int
) -> CatalogVectorStore:
    return CatalogVectorStore(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", ""),
        collection_name=collection_name,
        vector_size=vector_size,
        hnsw=hnsw,
        ef_search=ef_search,
    )


def sweep_collection(
    collection_name: str,
    *,
    hnsw: HnswSettings,
    vectors: np.ndarray,
    query_ids: list[str],
    exact_ids: list[set[str]],
    fidelity_k: int,
    vector_size: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    """Measure overlap@k and latency for every ef_search value."""
    rows = []
    for ef_search in EF_VALUES:
        store = build_store(
            collection_name, ef_search, hnsw=hnsw, vector_size=vector_size
        )
        for vector in vectors[:warmup]:
            store.search(vector, top_k=fidelity_k)
        overlaps: list[float] = []
        samples_ms: list[float] = []
        for repeat in range(repeats):
            for position, vector in enumerate(vectors):
                started_at = perf_counter()
                hits = store.search(vector, top_k=fidelity_k)
                samples_ms.append((perf_counter() - started_at) * 1000)
                if repeat == 0:
                    observed = {hit.record_id for hit in hits}
                    overlaps.append(len(observed & exact_ids[position]) / fidelity_k)
        worst_position = int(np.argmin(overlaps))
        rows.append(
            {
                "ef_search": ef_search,
                "mean_overlap_at_10": float(np.mean(overlaps)),
                "min_overlap_at_10": float(np.min(overlaps)),
                "worst_query_id": query_ids[worst_position],
                "p50_ms": round(float(median(samples_ms)), 3),
                "p95_ms": round(float(np.percentile(samples_ms, 95)), 3),
            }
        )
        print(f"  {collection_name}: {rows[-1]}")
    return rows


def ensure_weak_collection(
    catalog_size_store: CatalogVectorStore, embedding_set, batch_size: int
) -> None:
    """Create and seed the weak-graph collection when absent or incomplete."""
    catalog_size_store.ensure_collection(
        allow_reset=os.getenv("AURUM_ALLOW_RESET", "false").lower() == "true"
    )
    catalog = load_catalog()
    if catalog_size_store.count() != len(catalog):
        catalog_size_store.upsert_records(
            iter_record_batches(
                catalog, embedding_set.matrix(SET_PRODUCTS), batch_size=batch_size
            )
        )
    catalog_size_store.wait_until_indexed()


def main() -> None:
    """Run both sweeps and persist the fidelity/latency evidence."""
    load_dotenv(PROJECT_ROOT / ".env")
    arguments = parse_arguments()
    config = load_run_config()
    embedding_set = load_embedding_set(config.embedding_configuration)
    dimension = embedding_set.configuration.dimension
    vectors = np.concatenate(
        [
            embedding_set.matrix(SET_DEVELOPMENT_QUERIES),
            embedding_set.matrix(SET_EVALUATION_QUERIES),
        ]
    )
    query_ids = (
        embedding_set.identifiers[SET_DEVELOPMENT_QUERIES]
        + embedding_set.identifiers[SET_EVALUATION_QUERIES]
    )

    final_collection = os.getenv("QDRANT_COLLECTION", "aurum-market-eval-catalogo")
    oracle = build_store(
        final_collection, config.ef_search, hnsw=config.hnsw, vector_size=dimension
    )
    oracle.ping()
    exact_ids = [
        {
            hit.record_id
            for hit in oracle.search(vector, top_k=config.fidelity_k, exact=True)
        }
        for vector in vectors
    ]

    print("Barrido sobre la configuración final:")
    final_sweep = sweep_collection(
        final_collection,
        hnsw=config.hnsw,
        vectors=vectors,
        query_ids=query_ids,
        exact_ids=exact_ids,
        fidelity_k=config.fidelity_k,
        vector_size=dimension,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )

    print("Barrido sobre el grafo débil (m=4):")
    weak_store = build_store(
        WEAK_COLLECTION, EF_VALUES[0], hnsw=WEAK_HNSW, vector_size=dimension
    )
    ensure_weak_collection(weak_store, embedding_set, config.batch_size)
    weak_sweep = sweep_collection(
        WEAK_COLLECTION,
        hnsw=WEAK_HNSW,
        vectors=vectors,
        query_ids=query_ids,
        exact_ids=exact_ids,
        fidelity_k=config.fidelity_k,
        vector_size=dimension,
        warmup=arguments.warmup,
        repeats=max(3, arguments.repeats // 3),
    )

    path = write_json_artifact(
        {
            "queries": len(query_ids),
            "k": config.fidelity_k,
            "warmup": arguments.warmup,
            "repeats_latency": arguments.repeats,
            "final_config": {
                "collection": final_collection,
                "hnsw": {"m": config.hnsw.m, "ef_construct": config.hnsw.ef_construct},
                "sweep": final_sweep,
            },
            "weak_graph": {
                "collection": WEAK_COLLECTION,
                "hnsw": {"m": WEAK_HNSW.m, "ef_construct": WEAK_HNSW.ef_construct},
                "sweep": weak_sweep,
            },
        },
        ARTIFACTS_DIRECTORY / "evaluacion" / "barrido_ef_search.json",
    )
    print(f"Barrido escrito en {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
