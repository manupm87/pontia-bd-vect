"""Duplicate-control rule: vector candidates, thresholds, and calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import DuplicateDecision, SearchHit


@dataclass(frozen=True, slots=True)
class DuplicateRule:
    """Reproducible decision rule over the top vector candidates.

    An incoming record is flagged as duplicate when the best candidate's
    cosine similarity reaches ``score_threshold`` and its advantage over the
    second candidate reaches ``margin_threshold``. The vector database remains
    the only candidate generator.
    """

    score_threshold: float
    margin_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold debe estar en [0, 1].")
        if self.margin_threshold < 0.0:
            raise ValueError("margin_threshold no puede ser negativo.")

    def decide(
        self, incoming_id: str, candidates: Sequence[SearchHit]
    ) -> DuplicateDecision:
        """Decide over the (score-ordered) candidates returned by the store."""
        if not candidates:
            return DuplicateDecision(
                incoming_id=incoming_id,
                predicted_duplicate=False,
                matched_product_id="",
                score=0.0,
                margin=0.0,
            )
        best = candidates[0]
        runner_up_score = candidates[1].native_score if len(candidates) > 1 else 0.0
        margin = best.native_score - runner_up_score
        predicted = (
            best.native_score >= self.score_threshold
            and margin >= self.margin_threshold
        )
        return DuplicateDecision(
            incoming_id=incoming_id,
            predicted_duplicate=predicted,
            matched_product_id=best.product_id if predicted else "",
            score=best.native_score,
            margin=margin,
        )


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    """One labeled development case with its retrieved candidates."""

    incoming_id: str
    is_duplicate: bool
    reference_product_id: str
    candidates: tuple[SearchHit, ...]

    def __post_init__(self) -> None:
        if self.is_duplicate and not self.reference_product_id.strip():
            raise ValueError(
                f"El caso {self.incoming_id!r} es duplicado pero no declara "
                "reference_product_id."
            )


def confusion_metrics(
    cases: Sequence[CalibrationCase],
    rule: DuplicateRule,
    *,
    strict_match: bool = True,
) -> dict[str, Any]:
    """Precision, recall, and F1 of the rule over labeled cases.

    With ``strict_match`` a positive prediction only counts as a true positive
    when it points at the labeled reference product, not just any product.
    """
    if not cases:
        raise ValueError("No hay casos de calibración.")
    true_positive = false_positive = false_negative = true_negative = 0
    errors: list[dict[str, Any]] = []
    for case in cases:
        decision = rule.decide(case.incoming_id, case.candidates)
        correct_match = (
            not strict_match or decision.matched_product_id == case.reference_product_id
        )
        if decision.predicted_duplicate and case.is_duplicate and correct_match:
            true_positive += 1
        elif decision.predicted_duplicate:
            false_positive += 1
            errors.append({"incoming_id": case.incoming_id, "kind": "falso_positivo"})
        elif case.is_duplicate:
            false_negative += 1
            errors.append({"incoming_id": case.incoming_id, "kind": "falso_negativo"})
        else:
            true_negative += 1
    predicted_positive = true_positive + false_positive
    labeled_positive = true_positive + false_negative
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / labeled_positive if labeled_positive else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "rule": asdict(rule),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "errors": errors,
    }


def calibrate_rule(
    cases: Sequence[CalibrationCase],
    *,
    margin_candidates: Sequence[float] = (0.0, 0.005, 0.01, 0.02),
) -> dict[str, Any]:
    """Grid-search the rule on development cases and return the full report.

    The score grid is built from the observed candidate scores so every
    decision boundary between two consecutive cases is explored. Ties in F1
    prefer precision first (publishing a duplicate is reviewed by a person;
    flooding the review queue erodes trust) and a higher threshold second.
    """
    observed_scores = sorted(
        {case.candidates[0].native_score for case in cases if case.candidates}
    )
    if not observed_scores:
        raise ValueError("Ningún caso de calibración tiene candidatos.")
    score_grid = sorted(
        {round(score - 0.0005, 4) for score in observed_scores}
        | {round(score + 0.0005, 4) for score in observed_scores}
    )
    evaluations = []
    for margin_threshold in margin_candidates:
        for score_threshold in score_grid:
            if not 0.0 <= score_threshold <= 1.0:
                continue
            rule = DuplicateRule(
                score_threshold=score_threshold, margin_threshold=margin_threshold
            )
            evaluations.append(confusion_metrics(cases, rule))
    best = max(
        evaluations,
        key=lambda report: (
            report["f1"],
            report["precision"],
            report["rule"]["score_threshold"],
        ),
    )
    return {
        "best": best,
        "explored": len(evaluations),
        "score_grid": score_grid,
        "margin_candidates": list(margin_candidates),
    }


__all__ = [
    "CalibrationCase",
    "DuplicateRule",
    "calibrate_rule",
    "confusion_metrics",
]
