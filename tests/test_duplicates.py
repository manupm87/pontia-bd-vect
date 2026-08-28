"""Behavior of the duplicate rule, its metrics, and the calibration search."""

from __future__ import annotations

import pytest

from aurum_discovery.contracts import SearchHit
from aurum_discovery.duplicates import (
    CalibrationCase,
    DuplicateRule,
    calibrate_rule,
    confusion_metrics,
)


def build_hit(product_id: str, score: float, rank: int = 1) -> SearchHit:
    return SearchHit(
        rank=rank,
        record_id=f"id-{product_id}",
        product_id=product_id,
        title=f"Producto {product_id}",
        brand="",
        native_score=score,
        score_kind="similarity",
        higher_is_better=True,
    )


def build_case(
    incoming_id: str,
    *,
    is_duplicate: bool,
    top_score: float,
    second_score: float,
    reference: str = "REF",
) -> CalibrationCase:
    return CalibrationCase(
        incoming_id=incoming_id,
        is_duplicate=is_duplicate,
        reference_product_id=reference if is_duplicate else "",
        candidates=(
            build_hit("REF", top_score, rank=1),
            build_hit("OTRO", second_score, rank=2),
        ),
    )


def test_rule_flags_only_above_both_thresholds() -> None:
    rule = DuplicateRule(score_threshold=0.9, margin_threshold=0.05)
    candidates = (build_hit("REF", 0.95), build_hit("OTRO", 0.93, rank=2))
    decision = rule.decide("ALTA-1", candidates)
    assert not decision.predicted_duplicate
    confident = rule.decide(
        "ALTA-2", (build_hit("REF", 0.95), build_hit("OTRO", 0.80, rank=2))
    )
    assert confident.predicted_duplicate
    assert confident.matched_product_id == "REF"


def test_rule_without_candidates_predicts_negative() -> None:
    decision = DuplicateRule(score_threshold=0.5).decide("ALTA-3", ())
    assert not decision.predicted_duplicate
    assert decision.score == 0.0


def test_confusion_metrics_separate_error_kinds() -> None:
    cases = [
        build_case("TP", is_duplicate=True, top_score=0.97, second_score=0.6),
        build_case("FN", is_duplicate=True, top_score=0.70, second_score=0.6),
        build_case("FP", is_duplicate=False, top_score=0.96, second_score=0.6),
        build_case("TN", is_duplicate=False, top_score=0.50, second_score=0.4),
    ]
    report = confusion_metrics(cases, DuplicateRule(score_threshold=0.9))
    assert report["true_positive"] == 1
    assert report["false_negative"] == 1
    assert report["false_positive"] == 1
    assert report["true_negative"] == 1
    kinds = {error["incoming_id"]: error["kind"] for error in report["errors"]}
    assert kinds == {"FN": "falso_negativo", "FP": "falso_positivo"}


def test_strict_match_requires_the_labeled_reference() -> None:
    case = CalibrationCase(
        incoming_id="ALTA-4",
        is_duplicate=True,
        reference_product_id="ESPERADO",
        candidates=(build_hit("OTRO", 0.99),),
    )
    strict = confusion_metrics([case], DuplicateRule(score_threshold=0.9))
    relaxed = confusion_metrics(
        [case], DuplicateRule(score_threshold=0.9), strict_match=False
    )
    assert strict["false_positive"] == 1
    assert relaxed["true_positive"] == 1


def test_calibration_finds_a_separating_threshold() -> None:
    cases = [
        build_case("D1", is_duplicate=True, top_score=0.96, second_score=0.7),
        build_case("D2", is_duplicate=True, top_score=0.94, second_score=0.7),
        build_case("N1", is_duplicate=False, top_score=0.80, second_score=0.7),
        build_case("N2", is_duplicate=False, top_score=0.75, second_score=0.7),
    ]
    report = calibrate_rule(cases)
    assert report["best"]["f1"] == pytest.approx(1.0)
    threshold = report["best"]["rule"]["score_threshold"]
    assert 0.80 < threshold <= 0.94


def test_rule_validates_threshold_ranges() -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        DuplicateRule(score_threshold=1.5)
    with pytest.raises(ValueError, match="margin_threshold"):
        DuplicateRule(score_threshold=0.5, margin_threshold=-0.1)
