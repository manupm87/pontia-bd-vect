"""Create a deterministic 50k-product Spanish ESCI catalog and query workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "esci"


def parse_arguments() -> argparse.Namespace:
    """Read source paths and target size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--semantic-queries", type=Path, required=True)
    parser.add_argument("--catalog-size", type=int, default=50_000)
    parser.add_argument("--probe-count", type=int, default=256)
    return parser.parse_args()


def sha256_file(file_path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    """Normalize nullable product text without inventing fields."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def compose_searchable_text(product: pd.Series) -> str:
    """Compose the exact product representation used by the encoder."""
    sections = [clean_text(product["product_title"])]
    optional_fields = [
        ("Marca", product["product_brand"]),
        ("Color", product["product_color"]),
        ("Características", product["product_bullet_point"]),
        ("Descripción", product["product_description"]),
    ]
    for label, value in optional_fields:
        cleaned_value = clean_text(value)
        if cleaned_value:
            sections.append(f"{label}: {cleaned_value}")
    return ". ".join(sections)[:3_000]


def load_catalog(
    products_path: Path,
    judged_ids: pd.Series,
    catalog_size: int,
) -> pd.DataFrame:
    """Select every judged product plus a stable hashed background sample."""
    connection = duckdb.connect()
    judged_frame = pd.DataFrame({"product_id": judged_ids.astype(str).unique()})
    connection.register("judged_ids", judged_frame)
    judged_count = len(judged_frame)
    background_count = catalog_size - judged_count
    if background_count <= 0:
        raise ValueError("catalog_size must exceed the number of judged products")

    query = """
        WITH spanish AS (
            SELECT * FROM read_parquet(?) WHERE product_locale = 'es'
        ), judged AS (
            SELECT spanish.* FROM spanish INNER JOIN judged_ids USING (product_id)
        ), background AS (
            SELECT spanish.* FROM spanish
            ANTI JOIN judged_ids USING (product_id)
            ORDER BY hash(product_id)
            LIMIT ?
        )
        SELECT * FROM judged UNION ALL SELECT * FROM background
    """
    catalog = connection.execute(
        query, [str(products_path), background_count]
    ).fetch_df()
    connection.close()
    catalog = catalog.drop_duplicates("product_id").sort_values(
        "product_id", ignore_index=True
    )
    if len(catalog) != catalog_size:
        raise ValueError(f"Expected {catalog_size} products, found {len(catalog)}")
    return catalog


def build_query_workload(
    catalog: pd.DataFrame,
    judgments: pd.DataFrame,
    semantic_queries: pd.DataFrame,
    probe_count: int,
) -> pd.DataFrame:
    """Combine business queries with deterministic title probes."""
    original_queries = (
        judgments[["query_id", "query"]]
        .drop_duplicates()
        .sort_values("query_id", ignore_index=True)
    )
    original_rows = pd.DataFrame(
        {
            "workload_id": "original-" + original_queries["query_id"].astype(str),
            "query_id": original_queries["query_id"],
            "query_type": "esci_original",
            "query_text": original_queries["query"],
            "source_product_id": pd.NA,
        }
    )
    semantic_rows = pd.DataFrame(
        {
            "workload_id": "semantic-" + semantic_queries["query_id"].astype(str),
            "query_id": semantic_queries["query_id"],
            "query_type": "semantic_paraphrase",
            "query_text": semantic_queries["semantic_query"],
            "source_product_id": pd.NA,
        }
    )
    eligible = catalog.loc[
        catalog["product_title"].fillna("").str.len().between(25, 140)
    ].copy()
    probe_sample = eligible.sample(probe_count, random_state=42).sort_values(
        "product_id", ignore_index=True
    )
    probe_rows = pd.DataFrame(
        {
            "workload_id": "probe-" + np.arange(probe_count).astype(str),
            "query_id": pd.array([pd.NA] * probe_count, dtype="Int64"),
            "query_type": "product_title_probe",
            "query_text": probe_sample["product_title"],
            "source_product_id": probe_sample["product_id"],
        }
    )
    return pd.concat([original_rows, semantic_rows, probe_rows], ignore_index=True)


def main() -> None:
    """Generate compressed data files and a provenance manifest."""
    arguments = parse_arguments()
    judgments = pd.read_csv(arguments.judgments)
    semantic_queries = pd.read_csv(arguments.semantic_queries)
    catalog = load_catalog(
        arguments.products,
        judgments["product_id"],
        arguments.catalog_size,
    )
    catalog["searchable_text"] = catalog.apply(compose_searchable_text, axis=1)
    catalog.insert(0, "vector_id", np.arange(len(catalog), dtype=np.int64))
    workload = build_query_workload(
        catalog, judgments, semantic_queries, arguments.probe_count
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    products_output = OUTPUT_DIRECTORY / "products.csv.gz"
    judgments_output = OUTPUT_DIRECTORY / "judgments.csv"
    semantic_output = OUTPUT_DIRECTORY / "semantic_queries.csv"
    workload_output = OUTPUT_DIRECTORY / "query_workload.csv"
    catalog[
        [
            "vector_id",
            "product_id",
            "product_title",
            "product_brand",
            "product_color",
            "searchable_text",
        ]
    ].to_csv(products_output, index=False, compression="gzip")
    judgments.to_csv(judgments_output, index=False)
    semantic_queries.to_csv(semantic_output, index=False)
    workload.to_csv(workload_output, index=False)

    generated_files = [
        products_output,
        judgments_output,
        semantic_output,
        workload_output,
    ]
    manifest = {
        "snapshot_id": "amazon-esci-es-ann-session-02",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_repository": "https://github.com/amazon-science/esci-data",
        "source_license": "Apache-2.0",
        "selection": {
            "locale": "es",
            "catalog_size": len(catalog),
            "background_order": "DuckDB hash(product_id)",
            "probe_random_state": 42,
        },
        "counts": {
            "products": len(catalog),
            "judgments": len(judgments),
            "business_queries": int(
                workload["query_type"].ne("product_title_probe").sum()
            ),
            "probe_queries": int(
                workload["query_type"].eq("product_title_probe").sum()
            ),
        },
        "files": {path.name: sha256_file(path) for path in generated_files},
        "notes": [
            "All product metadata comes from the published ESCI dataset.",
            "Every judged product is retained before adding the background sample.",
            "Title probes are real product titles used only for ANN fidelity tests.",
        ],
    }
    (OUTPUT_DIRECTORY / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(catalog):,} products and {len(workload):,} queries")


if __name__ == "__main__":
    main()
