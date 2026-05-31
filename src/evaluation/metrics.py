"""Backward-compatible wrappers over the new evaluation pipeline."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .answer_metrics import (
    answer_contains_gold,
    normalized_exact_match,
    strict_exact_match,
    token_f1_score,
)


def normalize_answer(s: str) -> str:
    from .normalization import normalize_answer_text

    return normalize_answer_text(s)


def exact_match(pred: str, gold: str) -> bool:
    return bool(normalized_exact_match(pred, [gold]))


def contains_gold(pred: str, gold: str) -> bool:
    return bool(answer_contains_gold(pred, [gold]))


def token_f1(pred: str, gold: str) -> float:
    return token_f1_score(pred, [gold])


def evaluate_results(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    em_scores = []
    contains_scores = []
    f1_scores = []
    strict_scores = []
    for row in results:
        pred = row.get("predicted_answer", "")
        gold = row.get("gold_answer", "")
        strict_scores.append(strict_exact_match(pred, gold))
        em_scores.append(normalized_exact_match(pred, [gold]))
        contains_scores.append(answer_contains_gold(pred, [gold]))
        f1_scores.append(token_f1_score(pred, [gold]))
    return {
        "n": n,
        "exact_match": sum(em_scores) / n,
        "contains_gold": sum(contains_scores) / n,
        "token_f1": sum(f1_scores) / n,
        "em_count": sum(em_scores),
        "contains_count": sum(contains_scores),
        "strict_em_count": sum(strict_scores),
        "errors": sum(1 for row in results if row.get("error") or str(row.get("predicted_answer", "")).startswith("ERROR:")),
    }


def evaluate_by_question_type(results: list[dict], type_labels: dict[str, str] | None = None) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        qtype = (
            type_labels.get(row["question_id"], "unknown")
            if type_labels
            else row.get("question_type", "unknown")
        )
        groups[qtype].append(row)
    return {qtype: evaluate_results(rows) for qtype, rows in groups.items()}


def compare_systems(
    baseline_results: list[dict],
    graphrag_results: list[dict],
    type_labels: dict[str, str] | None = None,
) -> dict:
    baseline_metrics = evaluate_results(baseline_results)
    graphrag_metrics = evaluate_results(graphrag_results)
    output = {
        "baseline": baseline_metrics,
        "graphrag": graphrag_metrics,
        "deltas": {
            "exact_match": graphrag_metrics["exact_match"] - baseline_metrics["exact_match"],
            "contains_gold": graphrag_metrics["contains_gold"] - baseline_metrics["contains_gold"],
            "token_f1": graphrag_metrics["token_f1"] - baseline_metrics["token_f1"],
        },
    }
    if type_labels:
        output["baseline_by_type"] = evaluate_by_question_type(baseline_results, type_labels)
        output["graphrag_by_type"] = evaluate_by_question_type(graphrag_results, type_labels)
    return output


def print_metrics(metrics: dict, label: str = "") -> None:
    print(f"\n{'=' * 50}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 50}")
    print(f"  N questions:     {metrics['n']}")
    print(f"  Exact Match:     {metrics['exact_match']:.1%} ({metrics['em_count']}/{metrics['n']})")
    print(f"  Contains Gold:   {metrics['contains_gold']:.1%} ({metrics['contains_count']}/{metrics['n']})")
    print(f"  Token F1:        {metrics['token_f1']:.3f}")
    if metrics.get("errors"):
        print(f"  Errors:          {metrics['errors']}")
    print(f"{'=' * 50}")


def load_results(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
