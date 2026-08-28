"""Create the small Spanish ESCI snapshot used by the teaching notebook.

This maintenance script reads the two original Parquet files released by
Amazon Science. The resulting CSV files are committed with the session, so
students never need to download the multi-gigabyte source dataset.

Run with:

    uv run --with duckdb python scripts/prepare_esci_sample.py \
        --examples /path/to/shopping_queries_dataset_examples.parquet \
        --products /path/to/shopping_queries_dataset_products.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "esci"

SELECTED_QUERY_IDS = (
    13_357,  # base tapizada 160x200 sin patas
    18_868,  # botines marrones mujer tacón medio
    28_703,  # portátil convertible 2 en 1
    31_224,  # cámaras bridge baratas
    33_633,  # disfraz de Halloween talla grande
    38_249,  # estantes sin taladro
    43_240,  # funda iPad Air 4 sin tapa
    61_533,  # lentejas sin gluten
    93_437,  # sillas de oficina ergonómicas
    96_202,  # soporte de aire acondicionado de ventana
    100_455,  # taladro 24 V con batería
    101_352,  # televisión de 28 pulgadas
)

SOURCE_REPOSITORY = "https://github.com/amazon-science/esci-data"
SOURCE_PAPER = "https://arxiv.org/abs/2206.06588"


def parse_arguments() -> argparse.Namespace:
    """Parse source Parquet locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    return parser.parse_args()


def clean_text(value: object) -> str:
    """Collapse control characters and repeated whitespace."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_file(file_path: Path) -> str:
    """Return a stable digest for a generated artifact."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for file_chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(file_chunk)
    return digest.hexdigest()


def main() -> None:
    """Extract, clean and verify the selected Spanish query groups."""
    arguments = parse_arguments()
    for source_path in (arguments.examples, arguments.products):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    selected_ids_sql = ", ".join(str(query_id) for query_id in SELECTED_QUERY_IDS)
    connection = duckdb.connect()
    examples_frame = connection.execute(
        f"""
        SELECT example_id, query, query_id, product_id, product_locale,
               esci_label, split
        FROM read_parquet(?)
        WHERE product_locale = 'es'
          AND split = 'test'
          AND small_version = 1
          AND query_id IN ({selected_ids_sql})
        ORDER BY query_id, example_id
        """,
        [str(arguments.examples)],
    ).fetchdf()

    if set(examples_frame["query_id"].unique()) != set(SELECTED_QUERY_IDS):
        missing_ids = set(SELECTED_QUERY_IDS) - set(examples_frame["query_id"].unique())
        raise RuntimeError(f"Missing selected query IDs: {sorted(missing_ids)}")

    selected_product_ids = sorted(examples_frame["product_id"].unique())
    connection.register(
        "selected_products",
        pd.DataFrame({"product_id": selected_product_ids}),
    )
    products_frame = connection.execute(
        """
        SELECT products.product_id, products.product_locale,
               products.product_title, products.product_description,
               products.product_bullet_point, products.product_brand,
               products.product_color
        FROM read_parquet(?) AS products
        INNER JOIN selected_products USING (product_id)
        WHERE products.product_locale = 'es'
        ORDER BY products.product_id
        """,
        [str(arguments.products)],
    ).fetchdf()

    text_columns = [
        "query",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]
    examples_frame["query"] = examples_frame["query"].map(clean_text)
    for column_name in text_columns[1:]:
        products_frame[column_name] = products_frame[column_name].map(clean_text)

    products_frame = products_frame.drop_duplicates(
        subset=["product_id", "product_locale"], keep="first"
    )
    missing_products = set(selected_product_ids) - set(products_frame["product_id"])
    if missing_products:
        raise RuntimeError(f"Missing product metadata for {len(missing_products)} IDs")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    products_path = OUTPUT_DIRECTORY / "products.csv"
    judgments_path = OUTPUT_DIRECTORY / "judgments.csv"
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"

    products_frame.to_csv(products_path, index=False)
    examples_frame.to_csv(judgments_path, index=False)

    label_counts = {
        label: int(count)
        for label, count in examples_frame["esci_label"].value_counts().items()
    }
    manifest = {
        "snapshot_id": "amazon-esci-es-session-01",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_repository": SOURCE_REPOSITORY,
        "source_paper": SOURCE_PAPER,
        "source_license": "Apache-2.0",
        "selection": {
            "locale": "es",
            "split": "test",
            "small_version": 1,
            "query_ids": list(SELECTED_QUERY_IDS),
        },
        "counts": {
            "queries": int(examples_frame["query_id"].nunique()),
            "products": int(products_frame["product_id"].nunique()),
            "judgments": len(examples_frame),
            "labels": label_counts,
        },
        "files": {
            products_path.name: sha256_file(products_path),
            judgments_path.name: sha256_file(judgments_path),
        },
        "notes": [
            "Product titles and metadata are preserved from the published dataset.",
            "ESCI labels are real query-product relevance judgments.",
            "Prices, stock and reviews are not present and are never fabricated.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
