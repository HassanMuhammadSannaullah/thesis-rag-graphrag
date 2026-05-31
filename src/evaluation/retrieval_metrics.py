"""Retrieval and evidence metrics."""
from __future__ import annotations

import math
from typing import Any

from .normalization import normalize_answer_text
from .schemas import RetrievedContext


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _ranked_ids(contexts: list[RetrievedContext]) -> list[str]:
    ordered = sorted(
        contexts,
        key=lambda ctx: (
            ctx.rank if ctx.rank is not None else 10**9,
            -(ctx.score if ctx.score is not None else float("-inf")),
        ),
    )
    ids = []
    for ctx in ordered:
        if ctx.id:
            ids.append(ctx.id)
    return ids


def _dcg(relevances: list[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel:
            score += rel / math.log2(idx + 1)
    return score


def _normalized_context_text(context: RetrievedContext) -> str:
    return normalize_answer_text(context.text or context.id or "")


def _answer_support_flags(
    contexts: list[RetrievedContext],
    gold_answers: list[str] | None,
) -> list[int] | None:
    if not gold_answers:
        return None
    normalized_answers = [normalize_answer_text(answer) for answer in gold_answers if normalize_answer_text(answer)]
    if not normalized_answers:
        return None
    flags: list[int] = []
    for context in contexts:
        context_text = _normalized_context_text(context)
        flags.append(int(any(answer in context_text for answer in normalized_answers)))
    return flags


def compute_retrieval_metrics(
    gold_evidence: list[str] | None,
    retrieved_contexts: list[RetrievedContext] | None,
    *,
    gold_answers: list[str] | None = None,
    k_values: list[int],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    if retrieved_contexts is None:
        warnings.append("missing_retrieved_contexts")
        return metrics, warnings
    if retrieved_contexts and all(ctx.id is None for ctx in retrieved_contexts):
        warnings.append("retrieved_context_ids_missing")
        ranked_contexts = sorted(
            retrieved_contexts,
            key=lambda ctx: (
                ctx.rank if ctx.rank is not None else 10**9,
                -(ctx.score if ctx.score is not None else float("-inf")),
            ),
        )
    else:
        ranked_contexts = sorted(
            retrieved_contexts,
            key=lambda ctx: (
                ctx.rank if ctx.rank is not None else 10**9,
                -(ctx.score if ctx.score is not None else float("-inf")),
            ),
        )

    answer_support_flags = _answer_support_flags(ranked_contexts, gold_answers)
    if answer_support_flags is None:
        warnings.append("missing_gold_answers_for_support_metrics")
    else:
        metrics["answer_support_coverage"] = float(any(answer_support_flags))
        reciprocal_rank = 0.0
        for idx, flag in enumerate(answer_support_flags, start=1):
            if flag:
                reciprocal_rank = 1.0 / idx
                break
        metrics["answer_support_mrr"] = reciprocal_rank
        for k in k_values:
            metrics[f"answer_support_hit_at_{k}"] = float(any(answer_support_flags[:k]))

    if not gold_evidence:
        warnings.append("missing_gold_evidence")
        return metrics, warnings

    gold_set = set(gold_evidence)
    ranked_ids = _ranked_ids(ranked_contexts)
    metrics["evidence_coverage"] = _safe_div(len(set(ranked_ids) & gold_set), len(gold_set))

    reciprocal_rank = 0.0
    for idx, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold_set:
            reciprocal_rank = 1.0 / idx
            break
    metrics["mrr"] = reciprocal_rank

    for k in k_values:
        top_k = ranked_ids[:k]
        hits = sum(1 for item_id in top_k if item_id in gold_set)
        precision = _safe_div(hits, len(top_k)) if top_k else None
        recall = _safe_div(hits, len(gold_set))
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        relevances = [1 if item_id in gold_set else 0 for item_id in top_k]
        ideal = [1] * min(len(gold_set), k)
        dcg = _dcg(relevances)
        idcg = _dcg(ideal)
        metrics[f"evidence_precision_at_{k}"] = precision
        metrics[f"evidence_recall_at_{k}"] = recall
        metrics[f"evidence_f1_at_{k}"] = f1
        metrics[f"ndcg_at_{k}"] = (dcg / idcg) if idcg else None

    return metrics, warnings
