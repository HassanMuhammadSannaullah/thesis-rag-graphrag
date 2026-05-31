"""Hallucination-oriented evaluation metrics for RAG experiments."""
from __future__ import annotations

from typing import Any

from .normalization import normalize_answer_text
from .schemas import RetrievedContext


ABSTENTION_ANSWERS = {
    "i do not know",
    "don't know",
    "do not know",
    "unknown",
    "not enough information",
    "insufficient information",
}


def _joined_context(retrieved_contexts: list[RetrievedContext] | None) -> str:
    if not retrieved_contexts:
        return ""
    return "\n\n".join(ctx.text or ctx.id or "" for ctx in retrieved_contexts)


def compute_hallucination_metrics(
    *,
    predicted_answer: str,
    gold_answers: list[str],
    retrieved_contexts: list[RetrievedContext] | None,
    answer_metrics: dict[str, Any],
    judge_metrics: dict[str, Any],
) -> dict[str, Any]:
    normalized_prediction = normalize_answer_text(predicted_answer or "")
    normalized_context = normalize_answer_text(_joined_context(retrieved_contexts))
    normalized_gold = [normalize_answer_text(answer) for answer in gold_answers if answer]

    abstained = int(normalized_prediction in ABSTENTION_ANSWERS)
    prediction_in_context = int(bool(normalized_prediction) and normalized_prediction in normalized_context)
    gold_in_context = int(any(answer and answer in normalized_context for answer in normalized_gold))

    unsupported_prediction = int(
        bool(normalized_prediction)
        and not abstained
        and not prediction_in_context
        and bool(normalized_context)
    )

    faithfulness = judge_metrics.get("faithfulness")
    if faithfulness is None:
        faithfulness_gap = 1.0 - float(answer_metrics.get("normalized_exact_match", 0)) if unsupported_prediction else 0.0
    else:
        faithfulness_gap = max(0.0, min(1.0, 1.0 - float(faithfulness)))

    likely_hallucination = int(
        not abstained
        and float(answer_metrics.get("normalized_exact_match", 0)) == 0.0
        and (
            unsupported_prediction == 1
            or (gold_in_context == 1 and prediction_in_context == 0)
            or faithfulness_gap >= 0.5
        )
    )

    return {
        "abstention": abstained,
        "prediction_in_context": prediction_in_context,
        "gold_in_context": gold_in_context,
        "unsupported_prediction": unsupported_prediction,
        "likely_hallucination": likely_hallucination,
        "hallucination_score": faithfulness_gap,
    }
