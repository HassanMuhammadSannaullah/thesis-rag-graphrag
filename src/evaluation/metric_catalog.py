"""Shared metric definitions for classic RAG-vs-GraphRAG evaluation."""
from __future__ import annotations


def comparison_k_values(top_k: int) -> list[int]:
    return sorted({k for k in (1, 3, top_k) if k and k > 0})


def classic_canonical_metrics(top_k: int) -> list[str]:
    metrics = [
        "strict_exact_match",
        "normalized_exact_match",
        "token_f1",
        "answer_contains_gold",
        "numeric_value_match",
        "numeric_value_match_tolerant",
        "unit_match",
        "scale_match",
        "answer_type_match",
        "abstention",
        "prediction_in_context",
        "gold_in_context",
        "unsupported_prediction",
        "likely_hallucination",
        "hallucination_score",
        "answer_support_coverage",
        "answer_support_mrr",
        "evidence_coverage",
        "mrr",
        "answer_correctness",
        "answer_relevancy",
        "semantic_similarity",
        "factual_correctness",
        "faithfulness",
        "context_precision",
        "context_recall",
        "latency_seconds",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "retrieved_context_count",
    ]
    for k in comparison_k_values(top_k):
        metrics.extend(
            [
                f"answer_support_hit_at_{k}",
                f"evidence_precision_at_{k}",
                f"evidence_recall_at_{k}",
                f"evidence_f1_at_{k}",
                f"ndcg_at_{k}",
            ]
        )
    return metrics


def primary_statistical_metrics(top_k: int) -> list[str]:
    return [
        "normalized_exact_match",
        "token_f1",
        "answer_contains_gold",
        "gold_in_context",
        "prediction_in_context",
        "unsupported_prediction",
        "likely_hallucination",
        "answer_support_mrr",
        f"answer_support_hit_at_{top_k}",
        f"evidence_precision_at_{top_k}",
        f"evidence_recall_at_{top_k}",
        "latency_seconds",
    ]


def report_overall_columns(top_k: int) -> list[tuple[str, str]]:
    return [
        ("system_name", "System"),
        ("dataset_name", "Dataset"),
        ("generation_model", "Model"),
        ("count", "N"),
        ("normalized_exact_match", "Norm. EM"),
        ("token_f1", "Token F1"),
        ("answer_contains_gold", "Contains Gold"),
        ("numeric_value_match_tolerant", "Num Tol."),
        ("answer_type_match", "Type Match"),
        ("latency_seconds", "Avg Latency"),
    ]


def report_category_columns(top_k: int) -> list[tuple[str, str]]:
    return [
        ("group", "Question Type"),
        ("system_name", "System"),
        ("count", "N"),
        ("normalized_exact_match", "Norm. EM"),
        ("token_f1", "Token F1"),
        ("gold_in_context", "Gold In Ctx"),
        ("likely_hallucination", "Halluc."),
        (f"answer_support_hit_at_{top_k}", f"Hit@{top_k}"),
        ("latency_seconds", "Avg Latency"),
    ]


def report_grounding_columns(top_k: int) -> list[tuple[str, str]]:
    return [
        ("system_name", "System"),
        ("count", "N"),
        ("gold_in_context", "Gold In Ctx"),
        ("prediction_in_context", "Pred In Ctx"),
        ("unsupported_prediction", "Unsupported"),
        ("likely_hallucination", "Halluc."),
        ("abstention", "Abstain"),
        ("answer_support_mrr", "Ans MRR"),
        (f"answer_support_hit_at_{top_k}", f"Hit@{top_k}"),
        ("retrieved_context_count", "Ctx Count"),
    ]


def report_retrieval_columns(top_k: int) -> list[tuple[str, str]]:
    return [
        ("system_name", "System"),
        ("count", "N"),
        ("evidence_coverage", "Ev Cov."),
        ("mrr", "Ev MRR"),
        (f"evidence_precision_at_{top_k}", f"Ev P@{top_k}"),
        (f"evidence_recall_at_{top_k}", f"Ev R@{top_k}"),
        (f"evidence_f1_at_{top_k}", f"Ev F1@{top_k}"),
        (f"ndcg_at_{top_k}", f"nDCG@{top_k}"),
        ("total_tokens", "Avg Tokens"),
        ("latency_seconds", "Avg Latency"),
    ]


def comparison_report_columns(top_k: int) -> list[tuple[str, str]]:
    return [
        ("experiment_id", "Experiment"),
        ("system_name", "System"),
        ("dataset_name", "Dataset"),
        ("generation_model", "Model"),
        ("normalized_exact_match", "Norm. EM"),
        ("token_f1", "Token F1"),
        ("gold_in_context", "Gold In Ctx"),
        ("prediction_in_context", "Pred In Ctx"),
        ("likely_hallucination", "Halluc."),
        (f"answer_support_hit_at_{top_k}", f"Hit@{top_k}"),
        (f"evidence_recall_at_{top_k}", f"Ev R@{top_k}"),
        ("latency_seconds", "Latency"),
    ]
