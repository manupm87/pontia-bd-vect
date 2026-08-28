"""Sanity checks for the BM25 baseline and the Spanish text normalization."""

from __future__ import annotations

import pytest

from aurum_discovery.lexical import Bm25Index, normalize_text, tokenize


def test_normalize_strips_accents_but_keeps_enie() -> None:
    assert normalize_text("Tacón Marrón") == "tacon marron"
    assert normalize_text("Añejo") == "añejo"


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Botines, MARRONES 42") == ["botines", "marrones", "42"]


def test_bm25_ranks_the_matching_document_first() -> None:
    corpus = [
        "taladro inalámbrico 24v con batería",
        "vestido largo de fiesta para mujer",
        "funda de silicona para ipad air",
    ]
    index = Bm25Index(corpus)
    results = index.search("taladro con bateria", top_k=3)
    assert results[0][0] == 0
    assert results[0][1] > results[1][1]


def test_bm25_breaks_score_ties_by_position() -> None:
    index = Bm25Index(["mesa camilla", "mesa camilla", "silla"])
    results = index.search("mesa", top_k=2)
    assert [position for position, _ in results] == [0, 1]


def test_bm25_rejects_empty_corpus_and_bad_parameters() -> None:
    with pytest.raises(ValueError, match="documento"):
        Bm25Index([])
    with pytest.raises(ValueError, match="BM25"):
        Bm25Index(["doc"], k1=0)
