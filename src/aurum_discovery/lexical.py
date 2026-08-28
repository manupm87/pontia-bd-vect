"""Interpretable lexical baseline: BM25 over normalized Spanish tokens."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_TOKEN_PATTERN = re.compile(r"[a-z0-9ñ]+")


def normalize_text(text: str) -> str:
    """Lowercase and strip accents so 'Tacón' and 'tacon' match."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn" or character == "̃"
    )
    return unicodedata.normalize("NFC", stripped)


def tokenize(text: str) -> list[str]:
    """Split normalized text into alphanumeric tokens."""
    return _TOKEN_PATTERN.findall(normalize_text(text))


class Bm25Index:
    """Okapi BM25 index with deterministic, position-stable tie-breaking."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("El índice BM25 necesita al menos un documento.")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Parámetros BM25 inválidos: k1 > 0 y b en [0, 1].")
        self._k1 = k1
        self._b = b
        self._document_frequencies: Counter[str] = Counter()
        self._term_counts: list[Counter[str]] = []
        self._lengths: list[int] = []
        for document in documents:
            counts = Counter(tokenize(document))
            self._term_counts.append(counts)
            self._lengths.append(sum(counts.values()))
            self._document_frequencies.update(counts.keys())
        self._document_count = len(documents)
        self._average_length = sum(self._lengths) / self._document_count

    @property
    def document_count(self) -> int:
        return self._document_count

    def score(self, query: str, document_position: int) -> float:
        """Return the BM25 score of one document for the query."""
        counts = self._term_counts[document_position]
        length = self._lengths[document_position]
        score = 0.0
        for term in tokenize(query):
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (self._document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalized_length = 1 - self._b + self._b * length / self._average_length
            score += (
                inverse_document_frequency
                * frequency
                * (self._k1 + 1)
                / (frequency + self._k1 * normalized_length)
            )
        return score

    def search(self, query: str, *, top_k: int) -> list[tuple[int, float]]:
        """Return (document_position, score) pairs sorted by descending score."""
        if top_k < 1:
            raise ValueError("top_k debe ser positivo.")
        scored = [
            (position, self.score(query, position))
            for position in range(self._document_count)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


__all__ = ["Bm25Index", "normalize_text", "tokenize"]
