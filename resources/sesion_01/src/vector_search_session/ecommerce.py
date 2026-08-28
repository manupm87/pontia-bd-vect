"""Small, explicit helpers for the frozen Spanish Amazon ESCI sample."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SESSION_ROOT = Path(__file__).resolve().parents[2]
ESCI_DATA_DIRECTORY = SESSION_ROOT / "data" / "esci"
PRODUCTS_PATH = ESCI_DATA_DIRECTORY / "products.csv"
JUDGMENTS_PATH = ESCI_DATA_DIRECTORY / "judgments.csv"
ESCI_MANIFEST_PATH = ESCI_DATA_DIRECTORY / "manifest.json"

ESCI_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
ESCI_LABEL_NAMES = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant",
}

PRODUCT_COLUMNS = {
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
}
JUDGMENT_COLUMNS = {
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "split",
}


@dataclass(frozen=True, slots=True)
class EsciSample:
    """Validated product catalog and real query-product judgments."""

    products: pd.DataFrame
    judgments: pd.DataFrame


def clean_optional_text(value: object) -> str:
    """Return a compact string for optional catalog metadata."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compose_product_text(product: pd.Series) -> str:
    """Create the exact text representation used by local retrievers."""
    title = clean_optional_text(product["product_title"])
    brand = clean_optional_text(product["product_brand"])
    color = clean_optional_text(product["product_color"])
    bullet_points = clean_optional_text(product["product_bullet_point"])
    description = clean_optional_text(product["product_description"])

    sections = [title]
    if brand:
        sections.append(f"Marca: {brand}")
    if color:
        sections.append(f"Color: {color}")
    if bullet_points:
        sections.append(bullet_points)
    if description:
        sections.append(description)
    return ". ".join(section for section in sections if section)


def add_searchable_text(products: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with one reproducible `searchable_text` column."""
    enriched_products = products.copy()
    enriched_products["searchable_text"] = enriched_products.apply(
        compose_product_text,
        axis="columns",
    )
    return enriched_products


def load_esci_sample(
    products_path: str | Path = PRODUCTS_PATH,
    judgments_path: str | Path = JUDGMENTS_PATH,
) -> EsciSample:
    """Load and validate the committed Spanish e-commerce snapshot."""
    products = pd.read_csv(products_path, keep_default_na=True)
    judgments = pd.read_csv(judgments_path, keep_default_na=True)

    missing_product_columns = PRODUCT_COLUMNS - set(products.columns)
    missing_judgment_columns = JUDGMENT_COLUMNS - set(judgments.columns)
    if missing_product_columns:
        raise ValueError(f"Missing product columns: {sorted(missing_product_columns)}")
    if missing_judgment_columns:
        raise ValueError(
            f"Missing judgment columns: {sorted(missing_judgment_columns)}"
        )
    if products["product_id"].duplicated().any():
        raise ValueError("products.csv contains duplicate product IDs")
    if not set(judgments["esci_label"]).issubset(ESCI_GAINS):
        raise ValueError("judgments.csv contains unknown ESCI labels")

    product_ids = set(products["product_id"])
    missing_products = set(judgments["product_id"]) - product_ids
    if missing_products:
        raise ValueError(
            f"Judgments reference {len(missing_products)} missing products"
        )
    if set(judgments["product_locale"]) != {"es"}:
        raise ValueError("The teaching snapshot must contain only locale 'es'")

    normalized_products = add_searchable_text(products).sort_values(
        "product_id", ignore_index=True
    )
    normalized_judgments = judgments.sort_values(
        ["query_id", "example_id"], ignore_index=True
    )
    return EsciSample(normalized_products, normalized_judgments)


__all__ = [
    "ESCI_DATA_DIRECTORY",
    "ESCI_GAINS",
    "ESCI_LABEL_NAMES",
    "ESCI_MANIFEST_PATH",
    "JUDGMENTS_PATH",
    "PRODUCTS_PATH",
    "EsciSample",
    "add_searchable_text",
    "clean_optional_text",
    "compose_product_text",
    "load_esci_sample",
]
