"""Statistical summaries for experiment outputs without external dependencies."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from src.config import settings as cfg

from .metric_catalog import primary_statistical_metrics


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    fraction = position - lower
    return lower_value + (upper_value - lower_value) * fraction


def bootstrap_confidence_interval(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    if len(values) == 1:
        value = values[0]
        return {"mean": value, "ci_low": value, "ci_high": value}

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": sum(values) / len(values),
        "ci_low": _quantile(means, alpha),
        "ci_high": _quantile(means, 1.0 - alpha),
    }


def _paired_metric_values(metric_rows: list[dict[str, Any]], metric_name: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in metric_rows:
        system_name = row.get("system_name")
        question_id = row.get("question_id")
        value = row.get(metric_name)
        if system_name is None or question_id is None or not isinstance(value, (int, float)):
            continue
        grouped[str(question_id)][str(system_name)] = float(value)
    return grouped


def paired_randomization_test(
    metric_rows: list[dict[str, Any]],
    metric_name: str,
    *,
    n_permutations: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    paired = _paired_metric_values(metric_rows, metric_name)
    systems = sorted({system for per_q in paired.values() for system in per_q})
    results: list[dict[str, Any]] = []
    if len(systems) < 2:
        return results

    rng = random.Random(seed)
    for index, left_system in enumerate(systems):
        for right_system in systems[index + 1 :]:
            deltas = [
                per_q[right_system] - per_q[left_system]
                for per_q in paired.values()
                if left_system in per_q and right_system in per_q
            ]
            if not deltas:
                continue
            observed = sum(deltas) / len(deltas)
            ge_count = 0
            for _ in range(n_permutations):
                permuted = [delta if rng.random() < 0.5 else -delta for delta in deltas]
                permuted_mean = sum(permuted) / len(permuted)
                if abs(permuted_mean) >= abs(observed):
                    ge_count += 1
            p_value = (ge_count + 1) / (n_permutations + 1)
            positive = sum(1 for delta in deltas if delta > 0)
            negative = sum(1 for delta in deltas if delta < 0)
            total_non_zero = positive + negative
            sign_rate = (max(positive, negative) / total_non_zero) if total_non_zero else 0.5
            results.append(
                {
                    "metric": metric_name,
                    "system_a": left_system,
                    "system_b": right_system,
                    "paired_count": len(deltas),
                    "mean_delta": observed,
                    "p_value": p_value,
                    "win_rate_b_over_a": sign_rate if observed >= 0 else 1.0 - sign_rate,
                }
            )
    return results


def summarize_statistical_report(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"bootstrap": {}, "pairwise_tests": {}}
    for metric_name in primary_statistical_metrics(cfg.TOP_K_RETRIEVAL):
        by_system: dict[str, list[float]] = defaultdict(list)
        for row in metric_rows:
            system_name = row.get("system_name")
            value = row.get(metric_name)
            if system_name is None or not isinstance(value, (int, float)):
                continue
            by_system[str(system_name)].append(float(value))

        if by_system:
            report["bootstrap"][metric_name] = {
                system_name: bootstrap_confidence_interval(values)
                for system_name, values in by_system.items()
            }
        tests = paired_randomization_test(metric_rows, metric_name)
        if tests:
            report["pairwise_tests"][metric_name] = tests
    return report
