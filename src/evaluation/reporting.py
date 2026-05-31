"""Markdown reporting for experiments and comparisons."""
from __future__ import annotations

from typing import Any

from src.config import settings as cfg

from .metric_catalog import (
    comparison_report_columns,
    report_category_columns,
    report_grounding_columns,
    report_overall_columns,
    report_retrieval_columns,
)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(columns: list[tuple[str, str]], rows: list[dict[str, Any]]) -> list[str]:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    return lines


def _metric_or_default(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def build_experiment_report(
    *,
    config: dict[str, Any],
    overall_rows: list[dict[str, Any]],
    by_type_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    statistical_report: dict[str, Any],
) -> str:
    top_k = cfg.TOP_K_RETRIEVAL
    overall_columns = report_overall_columns(top_k)
    category_columns = report_category_columns(top_k)
    grounding_columns = report_grounding_columns(top_k)
    retrieval_columns = report_retrieval_columns(top_k)
    evidence_metric_key = f"evidence_recall_at_{top_k}"
    evidence_annotated_count = sum(1 for row in metric_rows if row["metrics"].get(evidence_metric_key) is not None)

    lines = [
        "# Experiment Report",
        "",
        "## Overview",
        "",
        f"- Experiment ID: `{config['experiment_id']}`",
        f"- Dataset: `{config['dataset_name']}`",
        f"- System: `{config['system_name']}`",
        f"- Backend: `{config['model_backend']}`",
        f"- Generation model: `{config['generation_model']}`",
        f"- Embedding model: `{config['embedding_model']}`",
        f"- Comparison retrieval cutoff: `top-{top_k}`",
        f"- Questions with gold evidence annotations: `{evidence_annotated_count}/{len(metric_rows)}`",
        "",
        "## Answer Quality",
        "",
        *_table(overall_columns, overall_rows),
        "",
        "## Grounding And Hallucination",
        "",
        *_table(grounding_columns, overall_rows),
        "",
    ]
    lines.extend(["## Retrieval", ""])
    lines.extend(
        [
            (
                "- Evidence-based retrieval metrics require gold evidence annotations and will appear as `null` "
                "when the dataset does not provide them."
            ),
            (
                "- Answer-support metrics (`Hit@k`, `Ans MRR`, `Gold In Ctx`) are still computed for both systems "
                "from the retrieved contexts and gold answers."
            ),
            "",
        ]
    )
    lines.extend(_table(retrieval_columns, overall_rows))
    lines.extend(["", "## Category Breakdown", ""])
    lines.extend(_table(category_columns, by_type_rows or [{"group": "none"}]))

    lines.extend(["", "## Statistical Summary", ""])
    pairwise_tests = statistical_report.get("pairwise_tests", {})
    if not pairwise_tests:
        lines.append("- No pairwise statistical comparisons available.")
    else:
        stats_cols = [
            ("metric", "Metric"),
            ("system_a", "System A"),
            ("system_b", "System B"),
            ("paired_count", "Paired N"),
            ("mean_delta", "Mean Delta"),
            ("p_value", "P-Value"),
            ("win_rate_b_over_a", "Win Rate B>A"),
        ]
        flattened_stats: list[dict[str, Any]] = []
        for tests in pairwise_tests.values():
            flattened_stats.extend(tests)
        lines.extend(_table(stats_cols, flattened_stats))

    failures = {
        "empty_answers": sum(1 for row in metric_rows if not (row.get("predicted_answer") or "").strip()),
        "missing_retrieved_contexts": sum(
            1 for row in metric_rows if "missing_retrieved_contexts" in row.get("warnings", [])
        ),
        "parsing_errors": sum(1 for row in metric_rows if row.get("errors")),
    }
    lines.extend(["", "## Failures And Warnings", ""])
    lines.extend([f"- {key}: {value}" for key, value in failures.items()])

    def _examples(predicate) -> list[dict[str, Any]]:
        return [row for row in metric_rows if predicate(row)][:3]

    sections = [
        ("Strong Examples", _examples(lambda row: (row["metrics"].get("normalized_exact_match") or 0) >= 1.0)),
        (
            "Weak Examples",
            _examples(
                lambda row: (row["metrics"].get("normalized_exact_match") or 0) == 0
                and (row["metrics"].get("likely_hallucination") or 0) >= 1.0
            ),
        ),
    ]
    for title, rows in sections:
        lines.extend(["", f"## {title}", ""])
        if not rows:
            lines.append("- None")
            continue
        for row in rows:
            lines.extend(
                [
                    f"- `{row['question_id']}` {row.get('question')}",
                    f"  Gold: {row.get('gold_answer')}",
                    f"  Predicted: {row.get('predicted_answer')}",
                ]
            )
    return "\n".join(lines) + "\n"


def build_comparison_report(rows: list[dict[str, Any]]) -> str:
    top_k = cfg.TOP_K_RETRIEVAL
    lines = ["# Experiment Comparison", "", "## Summary", ""]
    lines.extend(
        _table(
            comparison_report_columns(top_k),
            rows,
        )
    )
    if rows:
        best_exact_match = max(rows, key=lambda row: _metric_or_default(row, "normalized_exact_match", -1.0))
        best_grounding = max(rows, key=lambda row: _metric_or_default(row, "gold_in_context", -1.0))
        lowest_hallucination = min(rows, key=lambda row: _metric_or_default(row, "likely_hallucination", 10**9))
        best_support = max(rows, key=lambda row: _metric_or_default(row, f"answer_support_hit_at_{top_k}", -1.0))
        evidence_candidates = [row for row in rows if row.get(f"evidence_recall_at_{top_k}") is not None]
        best_recall = (
            max(evidence_candidates, key=lambda row: _metric_or_default(row, f"evidence_recall_at_{top_k}", -1.0))
            if evidence_candidates
            else None
        )
        fastest = min(rows, key=lambda row: _metric_or_default(row, "latency_seconds", 10**9))
        lines.extend(
            [
                "",
                "## Winners",
                "",
                f"- Best by normalized exact match: `{best_exact_match.get('system_name')}` in `{best_exact_match.get('experiment_id')}`",
                f"- Best by grounded answer coverage: `{best_grounding.get('system_name')}` in `{best_grounding.get('experiment_id')}`",
                f"- Lowest hallucination rate: `{lowest_hallucination.get('system_name')}` in `{lowest_hallucination.get('experiment_id')}`",
                f"- Best by answer-support hit@{top_k}: `{best_support.get('system_name')}` in `{best_support.get('experiment_id')}`",
                f"- Best by latency: `{fastest.get('system_name')}` in `{fastest.get('experiment_id')}`",
            ]
        )
        if best_recall is not None:
            lines.append(
                f"- Best by evidence recall@{top_k}: `{best_recall.get('system_name')}` in `{best_recall.get('experiment_id')}`"
            )
    return "\n".join(lines) + "\n"
