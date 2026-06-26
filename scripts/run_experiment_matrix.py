"""Run multiple benchmark configs from a single experiment matrix."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import snapshot_download

from src.benchmark.runner import LocalModelServerPreflightError, run_standard_benchmark
from src.config import settings as cfg
from src.evaluation.experiment_io import write_json
from src.evaluation.reporting import build_comparison_report


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = cfg.PROJECT_ROOT / value
    return value.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matrix of standard benchmark experiments.")
    parser.add_argument("--matrix", required=True, help="Path to a benchmark matrix JSON file.")
    parser.add_argument("--run-id", action="append", default=[], help="Run only selected experiment IDs.")
    parser.add_argument("--list", action="store_true", help="List configured experiments.")
    parser.add_argument("--download-only", action="store_true", help="Download/cache configured local Hugging Face models and exit.")
    return parser.parse_args()


def _load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("experiments"), list):
        raise ValueError("Matrix must contain an experiments list.")
    return payload


def _merge_config(defaults: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    config = dict(defaults)
    for key, value in experiment.items():
        if key in {"id", "enabled"}:
            continue
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def _looks_like_hf_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    if model_name.startswith(("http://", "https://")):
        return False
    return "/" in model_name


def _local_openai_models(config: dict[str, Any]) -> list[str]:
    models = dict(config.get("models", {}))
    if str(models.get("backend", cfg.MODEL_BACKEND)).lower() != "local_openai":
        return []
    names = [
        models.get("generation_model"),
        models.get("graphrag_index_model") or models.get("generation_model"),
        models.get("embedding_model"),
        models.get("reranker_model"),
    ]
    baseline = dict(config.get("baseline", {}))
    if baseline.get("use_reranker", True):
        names.append(baseline.get("reranker_model"))
    unique: list[str] = []
    for name in names:
        value = str(name).strip() if name else ""
        if _looks_like_hf_model(value) and value not in unique:
            unique.append(value)
    return unique


def _ensure_hf_models(config: dict[str, Any]) -> None:
    models = dict(config.get("models", {}))
    if not models.get("auto_download", False):
        return
    for model_name in _local_openai_models(config):
        try:
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(cfg.LOCAL_MODELS_DIR),
                local_files_only=True,
            )
            print(f"Hugging Face model already cached: {model_name}")
            continue
        except Exception:
            pass
        print(f"Downloading Hugging Face model: {model_name}")
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(cfg.LOCAL_MODELS_DIR),
            resume_download=True,
        )


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)


def _ensure_hybridqa_dataset(config: dict[str, Any]) -> None:
    dataset = dict(config.get("dataset", {}))
    if str(dataset.get("type", "hybridqa")).lower() != "hybridqa":
        return
    if not dataset.get("auto_download", False):
        return

    split = str(dataset.get("split", "dev"))
    raw_dir = cfg.RAW_DIR
    question_path = raw_dir / f"{split}.json"
    if not question_path.exists():
        _download_file(
            f"https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/{split}.json",
            question_path,
        )

    zip_path = raw_dir / "WikiTables-WithLinks.zip"
    if not zip_path.exists():
        _download_file(
            "https://github.com/wenhuchen/WikiTables-WithLinks/archive/refs/heads/master.zip",
            zip_path,
        )


def _overall_row(result: dict[str, Any]) -> dict[str, Any]:
    aggregate = result.get("aggregate_metrics", {})
    rows = aggregate.get("overall") or aggregate.get("system_name") or []
    return dict(rows[0]) if rows else {}


def _comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        run_id = result.get("id")
        summary = result.get("summary", {})
        output_dir = summary.get("output_dir")
        for system_name, metrics in summary.get("metrics_by_system", {}).items():
            row = _overall_row(metrics)
            if not row:
                continue
            metric_config = dict(metrics.get("config", {}))
            row["matrix_experiment_id"] = run_id
            row["run_output_dir"] = output_dir
            row.setdefault("system_name", system_name)
            row.setdefault("dataset_name", metric_config.get("dataset_name"))
            row.setdefault("generation_model", metric_config.get("generation_model"))
            row.setdefault("embedding_model", metric_config.get("embedding_model"))
            row.setdefault("model_backend", metric_config.get("model_backend"))
            if run_id:
                row["experiment_id"] = f"{run_id}:{row.get('system_name', system_name)}"
            rows.append(row)
    return rows


def _write_outputs(
    *,
    output_path: Path,
    report_path: Path,
    matrix_path: Path,
    results: list[dict[str, Any]],
) -> None:
    comparison_rows = _comparison_rows(results)
    write_json(
        output_path,
        {
            "matrix": str(matrix_path),
            "results": results,
            "comparison_rows": comparison_rows,
            "report_path": str(report_path),
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_comparison_report(comparison_rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    matrix_path = _resolve(args.matrix)
    matrix = _load_matrix(matrix_path)
    experiments = matrix["experiments"]
    if args.list:
        for row in experiments:
            state = "enabled" if row.get("enabled", True) else "disabled"
            print(f"{row.get('id')} [{state}]")
        return

    selected = [
        row for row in experiments
        if (not args.run_id or row.get("id") in set(args.run_id)) and row.get("enabled", True)
    ]
    defaults = dict(matrix.get("defaults", {}))
    if args.download_only:
        for row in selected:
            config = _merge_config(defaults, row)
            _ensure_hybridqa_dataset(config)
            _ensure_hf_models(config)
        print(json.dumps({"downloaded_or_cached": len(selected)}, indent=2))
        return

    output_path = _resolve(matrix.get("output_path", "results/experiments/matrix_summary.json"))
    report_path = _resolve(matrix.get("report_path", output_path.with_suffix(".md")))
    results = []
    for row in selected:
        config = _merge_config(defaults, row)
        try:
            _ensure_hybridqa_dataset(config)
            _ensure_hf_models(config)
            summary = run_standard_benchmark(config)
            results.append({"id": row.get("id"), "status": "completed", "summary": summary})
        except Exception as exc:
            results.append({"id": row.get("id"), "status": "failed", "error": repr(exc)})
            print(f"ERROR: experiment {row.get('id')} failed: {exc}", file=sys.stderr)
            if isinstance(exc, LocalModelServerPreflightError):
                _write_outputs(output_path=output_path, report_path=report_path, matrix_path=matrix_path, results=results)
                raise SystemExit(1)
        _write_outputs(output_path=output_path, report_path=report_path, matrix_path=matrix_path, results=results)
    completed = sum(1 for result in results if result.get("status") == "completed")
    failed = sum(1 for result in results if result.get("status") == "failed")
    print(json.dumps({"output_path": str(output_path), "completed": completed, "failed": failed}, indent=2))


if __name__ == "__main__":
    main()
