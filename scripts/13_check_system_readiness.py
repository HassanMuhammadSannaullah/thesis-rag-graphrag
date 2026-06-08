"""System readiness checks before running thesis experiments."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings as cfg
from src.utils.runtime import detect_hardware_snapshot, detect_runtime_environment


@dataclass
class CheckResult:
    name: str
    status: str
    details: str
    critical: bool
    final_run_required: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", default="configs/experiment_matrix.json")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if any critical check fails")
    return parser.parse_args()


def _exists_file(path: Path, *, critical: bool, label: str) -> CheckResult:
    if path.exists() and path.is_file():
        return CheckResult(label, "pass", f"Found file: {path}", critical)
    return CheckResult(label, "fail", f"Missing file: {path}", critical)


def _exists_dir(path: Path, *, critical: bool, label: str) -> CheckResult:
    if path.exists() and path.is_dir():
        return CheckResult(label, "pass", f"Found directory: {path}", critical)
    return CheckResult(label, "fail", f"Missing directory: {path}", critical)


def _disk_check(min_free_gb: float = 20.0) -> CheckResult:
    usage = shutil.disk_usage(cfg.PROJECT_ROOT)
    free_gb = usage.free / (1024**3)
    if free_gb >= min_free_gb:
        return CheckResult("disk_space", "pass", f"Free disk: {free_gb:.1f}GB", True, True)
    return CheckResult("disk_space", "fail", f"Free disk too low: {free_gb:.1f}GB (< {min_free_gb:.1f}GB)", True, True)


def _runtime_checks() -> list[CheckResult]:
    runtime_info = detect_runtime_environment()
    hardware_info = detect_hardware_snapshot(disk_path=cfg.PROJECT_ROOT)
    results: list[CheckResult] = []

    using_preferred_env = bool(runtime_info.get("using_preferred_conda_env"))
    current_env = runtime_info.get("current_conda_env") or "none"
    preferred_env = runtime_info.get("preferred_conda_env") or "unknown"
    results.append(
        CheckResult(
            "runtime_conda_env",
            "pass" if using_preferred_env else "fail",
            f"Current env: {current_env}; preferred env: {preferred_env}",
            True,
            True,
        )
    )

    python_matches = bool(runtime_info.get("current_python_matches_resolved"))
    results.append(
        CheckResult(
            "runtime_python_match",
            "pass" if python_matches else "fail",
            (
                f"Current python: {runtime_info.get('current_python')}; "
                f"resolved project python: {runtime_info.get('resolved_project_python')}"
            ),
            True,
            True,
        )
    )

    if cfg.MODEL_BACKEND == "local_openai":
        cuda_available = bool(hardware_info.get("torch_cuda_available"))
        gpu_name = hardware_info.get("primary_gpu_name") or "none"
        gpu_memory = hardware_info.get("primary_gpu_total_memory_gb")
        results.append(
            CheckResult(
                "torch_cuda_runtime",
                "pass" if cuda_available else "fail",
                f"torch CUDA available={cuda_available}; gpu={gpu_name}; vram_gb={gpu_memory}",
                False,
                True,
            )
        )

    total_ram_gb = hardware_info.get("ram_total_gb")
    available_ram_gb = hardware_info.get("ram_available_gb")
    if total_ram_gb is None or available_ram_gb is None:
        results.append(
            CheckResult(
                "system_ram",
                "warn",
                f"Unable to detect RAM reliably: {hardware_info.get('ram_error', 'unknown error')}",
                False,
                True,
            )
        )
    else:
        ram_status = "pass" if available_ram_gb >= 12.0 else "warn"
        results.append(
            CheckResult(
                "system_ram",
                ram_status,
                f"Total RAM: {total_ram_gb:.2f}GB; available RAM: {available_ram_gb:.2f}GB",
                False,
                True,
            )
        )

    detected_vram = hardware_info.get("primary_gpu_total_memory_gb")
    configured_vram = cfg.LOCAL_MODEL_VRAM_GB
    if detected_vram is not None:
        status = "pass" if abs(detected_vram - configured_vram) < 0.75 else "warn"
        results.append(
            CheckResult(
                "configured_vram_match",
                status,
                f"Configured VRAM={configured_vram}GB; detected VRAM={detected_vram}GB",
                False,
                False,
            )
        )
    return results


def _fairness_config_check() -> list[CheckResult]:
    results: list[CheckResult] = []
    if cfg.FAIR_BASELINE_MAX_CONTEXT_CHARS > 0:
        results.append(CheckResult("fair_context_chars", "pass", f"FAIR_BASELINE_MAX_CONTEXT_CHARS={cfg.FAIR_BASELINE_MAX_CONTEXT_CHARS}", True))
    else:
        results.append(CheckResult("fair_context_chars", "fail", "FAIR_BASELINE_MAX_CONTEXT_CHARS must be > 0", True))

    if cfg.FAIR_BASELINE_MAX_ANSWER_TOKENS > 0:
        results.append(CheckResult("fair_answer_tokens", "pass", f"FAIR_BASELINE_MAX_ANSWER_TOKENS={cfg.FAIR_BASELINE_MAX_ANSWER_TOKENS}", True))
    else:
        results.append(CheckResult("fair_answer_tokens", "fail", "FAIR_BASELINE_MAX_ANSWER_TOKENS must be > 0", True))
    return results


def _matrix_check(matrix_path: Path) -> CheckResult:
    if not matrix_path.exists():
        return CheckResult("experiment_matrix", "fail", f"Missing matrix file: {matrix_path}", True)
    try:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CheckResult("experiment_matrix", "fail", f"Invalid JSON: {exc}", True)

    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return CheckResult("experiment_matrix", "fail", "Matrix must include non-empty 'experiments' list", True)

    enabled = [row for row in experiments if row.get("enabled")]
    return CheckResult(
        "experiment_matrix",
        "pass",
        f"Matrix valid. Experiments: {len(experiments)}, enabled: {len(enabled)}",
        True,
    )


def _active_model_cache_hint() -> list[CheckResult]:
    results: list[CheckResult] = []
    if cfg.MODEL_BACKEND != "local_openai":
        results.append(CheckResult("local_model_backend", "pass", f"MODEL_BACKEND={cfg.MODEL_BACKEND} (no local cache requirement)", False))
        return results

    gen_hint = cfg.LOCAL_MODELS_DIR / f"models--{cfg.LOCAL_GENERATION_MODEL.replace('/', '--')}"
    emb_hint = cfg.LOCAL_MODELS_DIR / f"models--{cfg.LOCAL_EMBEDDING_MODEL.replace('/', '--')}"

    if gen_hint.exists():
        results.append(CheckResult("generation_model_cache", "pass", f"Found active generation cache hint: {gen_hint}", False))
    else:
        results.append(CheckResult("generation_model_cache", "warn", f"Missing active generation cache hint: {gen_hint}", False))

    if emb_hint.exists():
        results.append(CheckResult("embedding_model_cache", "pass", f"Found active embedding cache hint: {emb_hint}", False))
    else:
        results.append(CheckResult("embedding_model_cache", "warn", f"Missing active embedding cache hint: {emb_hint}", False))
    return results


def run_checks(matrix_path: Path) -> list[CheckResult]:
    checks: list[CheckResult] = [
        _exists_file(cfg.ORIGINAL_DIR / "dev.jsonl", critical=True, label="hybridqa_dev_data"),
        _exists_file(cfg.COMPLIANCE_DIR / "transactions.json", critical=False, label="compliance_transactions"),
        _exists_file(cfg.COMPLIANCE_DIR / "policies.json", critical=False, label="compliance_policies"),
        _exists_file(cfg.COMPLIANCE_DIR / "questions.json", critical=False, label="compliance_questions"),
        _exists_file(cfg.PROJECT_ROOT / "scripts" / "11_run_hybridqa_proper_compare.py", critical=True, label="hybridqa_runner_script"),
        _exists_file(cfg.PROJECT_ROOT / "scripts" / "08_run_compliance_full_experiment.py", critical=False, label="compliance_runner_script"),
        _exists_file(cfg.PROJECT_ROOT / "scripts" / "12_prepare_local_models.py", critical=False, label="model_prepare_script"),
        _exists_dir(cfg.RESULTS_DIR / "experiments", critical=True, label="results_experiments_dir"),
        _disk_check(min_free_gb=20.0),
        _matrix_check(matrix_path),
    ]
    checks.extend(_runtime_checks())
    checks.extend(_fairness_config_check())
    checks.extend(_active_model_cache_hint())
    return checks


def summarize(checks: list[CheckResult]) -> dict[str, Any]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in checks:
        counts[item.status] = counts.get(item.status, 0) + 1
    critical_failures = [item for item in checks if item.status == "fail" and item.critical]
    final_run_issues = [item for item in checks if item.final_run_required and item.status != "pass"]
    return {
        "summary": counts,
        "critical_failure_count": len(critical_failures),
        "ready": len(critical_failures) == 0,
        "final_run_issue_count": len(final_run_issues),
        "final_ready": len(critical_failures) == 0 and len(final_run_issues) == 0,
        "checks": [item.__dict__ for item in checks],
    }


def main() -> None:
    args = parse_args()
    matrix_path = (cfg.PROJECT_ROOT / args.matrix_path).resolve() if not Path(args.matrix_path).is_absolute() else Path(args.matrix_path)
    checks = run_checks(matrix_path)
    report = summarize(checks)

    print("=" * 72)
    print("SYSTEM READINESS")
    print("=" * 72)
    for item in checks:
        print(f"[{item.status.upper():4}] {item.name}: {item.details}")
    print("-" * 72)
    print(f"Summary: {report['summary']}")
    print(f"Critical failures: {report['critical_failure_count']}")
    print(f"Ready for smoke tests: {report['ready']}")
    print(f"Final-run issues: {report['final_run_issue_count']}")
    print(f"Ready for final runs: {report['final_ready']}")

    out_path = cfg.METRICS_DIR / "system_readiness.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved readiness report: {out_path}")

    if args.strict and not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
