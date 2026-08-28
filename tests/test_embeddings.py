"""Text composition rules and the offline embedding-set contract."""

from __future__ import annotations

import pandas as pd
import pytest

from aurum_discovery.config import load_run_config
from aurum_discovery.embeddings import (
    EMBEDDING_CONFIGURATIONS,
    compose_document_text,
    get_configuration,
)


def build_row(**overrides: str) -> pd.Series:
    row = {
        "title": "Taladro inalámbrico 24V",
        "brand": "Einhell",
        "color": "Rojo",
        "text": "Taladro inalámbrico 24V. Marca: Einhell. Color: Rojo.",
    }
    row.update(overrides)
    return pd.Series(row)


def test_full_text_composition_uses_the_text_field() -> None:
    row = build_row()
    assert compose_document_text(row, composition="full_text") == row["text"]


def test_title_composition_omits_missing_metadata() -> None:
    row = build_row(brand="", color="")
    composed = compose_document_text(row, composition="title_brand_color")
    assert composed == "Taladro inalámbrico 24V"
    assert "nan" not in composed.lower()


def test_empty_text_falls_back_to_title_composition() -> None:
    row = build_row(text="   ")
    composed = compose_document_text(row, composition="full_text")
    assert composed.startswith("Taladro inalámbrico 24V")
    assert "Marca: Einhell" in composed


def test_fully_empty_rows_are_rejected() -> None:
    row = build_row(title="", brand="", color="", text="")
    with pytest.raises(ValueError, match="texto"):
        compose_document_text(row, composition="full_text")


def test_unknown_configuration_lists_the_valid_names() -> None:
    with pytest.raises(ValueError, match="e5_small_full"):
        get_configuration("desconocida")


def test_registered_configurations_cover_the_required_experiments() -> None:
    assert len(EMBEDDING_CONFIGURATIONS) >= 2
    models = {config.model_id for config in EMBEDDING_CONFIGURATIONS.values()}
    compositions = {config.composition for config in EMBEDDING_CONFIGURATIONS.values()}
    assert len(models) >= 2
    assert len(compositions) >= 2


def test_run_config_points_to_a_registered_configuration() -> None:
    config = load_run_config()
    assert config.embedding_configuration in EMBEDDING_CONFIGURATIONS
    assert config.recall_relevant_labels == ("E", "S")
    assert config.mrr_relevant_labels == ("E",)
