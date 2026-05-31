"""Deterministic answer comparison metrics."""
from __future__ import annotations

from collections import Counter
from math import isclose
from typing import Any

from .normalization import (
    canonical_text_variants,
    extract_numbers_with_units,
    normalize_answer_text,
    normalize_tokens,
)


def strict_exact_match(predicted: str, gold: str) -> int:
    return int((predicted or "").strip() == (gold or "").strip())


def normalized_exact_match(predicted: str, gold_answers: list[str]) -> int:
    pred_norm = normalize_answer_text(predicted or "")
    gold_norms = {normalize_answer_text(gold) for gold in gold_answers}
    return int(pred_norm in gold_norms)


def answer_contains_gold(predicted: str, gold_answers: list[str]) -> int:
    pred_norm = normalize_answer_text(predicted or "")
    return int(any(normalize_answer_text(gold) in pred_norm for gold in gold_answers))


def token_f1_score(predicted: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_tokens(predicted or "")
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_tokens(gold)
        if not gold_tokens:
            best = max(best, 1.0 if not pred_tokens else 0.0)
            continue
        if not pred_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _best_numeric_alignment(predicted: str, gold_answers: list[str], tolerance: float) -> dict[str, Any]:
    pred_numbers = extract_numbers_with_units(predicted or "")
    gold_numbers = []
    for gold in gold_answers:
        gold_numbers.extend(extract_numbers_with_units(gold))

    result = {
        "numeric_value_match": None,
        "numeric_value_match_tolerant": None,
        "unit_match": None,
        "scale_match": None,
        "predicted_numbers": pred_numbers,
        "gold_numbers": gold_numbers,
    }
    if not pred_numbers or not gold_numbers:
        return result

    value_match = 0
    tolerant_match = 0
    unit_match = 0
    scale_match = 0
    for pred_num in pred_numbers:
        for gold_num in gold_numbers:
            if pred_num["value"] == gold_num["value"]:
                value_match = 1
            if isclose(pred_num["value"], gold_num["value"], rel_tol=tolerance, abs_tol=tolerance):
                tolerant_match = 1
            if pred_num.get("unit") and gold_num.get("unit") and pred_num["unit"] == gold_num["unit"]:
                unit_match = 1
            if pred_num.get("scale") == gold_num.get("scale"):
                scale_match = 1

    result.update(
        {
            "numeric_value_match": value_match,
            "numeric_value_match_tolerant": tolerant_match,
            "unit_match": unit_match,
            "scale_match": scale_match,
        }
    )
    return result


def answer_type_match(predicted: str, gold_answers: list[str], answer_type: str | None) -> int | None:
    if not answer_type:
        return None
    predicted_numbers = extract_numbers_with_units(predicted or "")
    normalized_prediction = normalize_answer_text(predicted or "")
    answer_type = answer_type.lower()
    if answer_type in {"numeric", "number", "currency", "percentage"}:
        return int(bool(predicted_numbers))
    if answer_type in {"boolean", "bool"}:
        return int(normalized_prediction in {"yes", "no", "true", "false"})
    if answer_type in {"date", "datetime"}:
        return int(any("-" in variant and variant[:4].isdigit() for variant in canonical_text_variants([predicted])))
    return int(bool(normalized_prediction))


def compute_answer_metrics(
    predicted: str,
    gold_answers: list[str],
    *,
    answer_type: str | None = None,
    numeric_tolerance: float = 1e-3,
) -> dict[str, Any]:
    primary_gold = gold_answers[0] if gold_answers else ""
    metrics: dict[str, Any] = {
        "strict_exact_match": strict_exact_match(predicted, primary_gold),
        "normalized_exact_match": normalized_exact_match(predicted, gold_answers),
        "token_f1": token_f1_score(predicted, gold_answers),
        "answer_contains_gold": answer_contains_gold(predicted, gold_answers),
        "answer_type_match": answer_type_match(predicted, gold_answers, answer_type),
    }
    metrics.update(_best_numeric_alignment(predicted, gold_answers, numeric_tolerance))
    return metrics
