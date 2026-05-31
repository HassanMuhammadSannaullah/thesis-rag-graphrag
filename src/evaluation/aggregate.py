"""Aggregation utilities for experiment metrics."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


NUMERIC_METRICS = {
    "strict_exact_match",
    "normalized_exact_match",
    "token_f1",
    "answer_contains_gold",
    "numeric_value_match",
    "numeric_value_match_tolerant",
    "unit_match",
    "scale_match",
    "answer_type_match",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "context_precision_llm",
    "context_recall_llm",
}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mean = _mean(values)
    assert mean is not None
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(rows),
        "failure_count": sum(1 for row in rows if row.get("error")),
        "warning_count": sum(len(row.get("warnings", [])) for row in rows),
        "empty_answer_count": sum(1 for row in rows if not (row.get("predicted_answer") or "").strip()),
        "judge_failure_count": sum(
            1 for row in rows if any("judge_" in err for err in row.get("errors", []))
        ),
    }
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_nulls: dict[str, int] = defaultdict(int)

    for row in rows:
        for metric_name, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                metric_values[metric_name].append(float(value))
            elif value is None:
                metric_nulls[metric_name] += 1

    for metric_name, values in metric_values.items():
        summary[metric_name] = _mean(values)
        summary[f"{metric_name}_std"] = _std(values)
        summary[f"{metric_name}_nulls"] = metric_nulls.get(metric_name, 0)
    for metric_name, null_count in metric_nulls.items():
        summary.setdefault(metric_name, None)
        summary.setdefault(f"{metric_name}_std", None)
        summary[f"{metric_name}_nulls"] = null_count
    return summary


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_fields: list[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["overall"].append(row)
        for field in group_fields:
            value = row.get(field)
            if value is None:
                continue
            grouped[f"{field}:{value}"].append(row)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, items in grouped.items():
        if key == "overall":
            output["overall"].append({"group": "overall", **summarize_group(items)})
            continue
        field, value = key.split(":", 1)
        output[field].append({"group": value, **summarize_group(items)})
    return dict(output)


def flatten_aggregate_tables(aggregate: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, entries in aggregate.items():
        for entry in entries:
            row = {"group_by": section, **entry}
            rows.append(row)
    return rows
