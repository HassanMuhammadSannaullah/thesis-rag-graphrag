"""Run thesis experiments from an editable matrix configuration."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", default="configs/experiment_matrix.json")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run-id", action="append", default=[], help="Run only selected experiment id(s)")
    parser.add_argument("--run-enabled", action="store_true", help="Run all enabled experiments from matrix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--auto-compare", action="store_true", help="Run compare_experiments.py for completed runs")
    parser.add_argument("--log-file", help="Append full matrix stdout/stderr to this log file")
    return parser.parse_args()


def _load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Matrix must contain an 'experiments' list")
    return payload


def _bool_flag(name: str) -> str:
    return f"--{name}"


def _build_cli_args(args_map: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in args_map.items():
        flag = _bool_flag(key)
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                cli.extend([flag, str(item)])
            continue
        cli.extend([flag, str(value)])
    return cli


def _experiment_dirs_snapshot() -> set[Path]:
    root = cfg.RESULTS_DIR / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    return {p.resolve() for p in root.iterdir() if p.is_dir()}


def _pick_new_experiment_dir(before: set[Path], after: set[Path]) -> str | None:
    diff = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(diff[0]) if diff else None


def _select_experiments(matrix: dict[str, Any], run_ids: list[str], run_enabled: bool) -> list[dict[str, Any]]:
    rows = matrix.get("experiments", [])
    if run_ids:
        wanted = set(run_ids)
        selected = [row for row in rows if row.get("id") in wanted]
        missing = wanted - {row.get("id") for row in selected}
        if missing:
            raise ValueError(f"Unknown experiment ids: {sorted(missing)}")
        return selected
    if run_enabled:
        return [row for row in rows if row.get("enabled")]
    return []


def _resolve_output_root(matrix: dict[str, Any]) -> Path:
    root = str(matrix.get("defaults", {}).get("output_root", "results/experiments/matrix_runs"))
    path = Path(root)
    if not path.is_absolute():
        path = (cfg.PROJECT_ROOT / path).resolve()
    return path


def _append_log(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, log_path: Path | None = None) -> int:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        _append_log(log_path, line)
    return process.wait()


def _run_compare(
    experiment_dirs: list[str],
    python_exe: str,
    output_root: Path,
    *,
    log_path: Path | None = None,
) -> dict[str, str] | None:
    if len(experiment_dirs) < 2:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    md_out = output_root / "reports" / f"matrix_compare_{stamp}.md"
    csv_out = output_root / "reports" / f"matrix_compare_{stamp}.csv"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "scripts/compare_experiments.py",
        "--experiments",
        *experiment_dirs,
        "--output-md",
        str(md_out),
        "--output-csv",
        str(csv_out),
    ]
    return_code = _run_command(cmd, cwd=cfg.PROJECT_ROOT, log_path=log_path)
    if return_code != 0:
        return None
    return {"report_md": str(md_out), "report_csv": str(csv_out)}


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file).resolve() if args.log_file else None
    matrix_path = (cfg.PROJECT_ROOT / args.matrix_path).resolve() if not Path(args.matrix_path).is_absolute() else Path(args.matrix_path)
    matrix = _load_matrix(matrix_path)
    experiments = matrix.get("experiments", [])

    if args.list:
        print("=" * 72)
        print("EXPERIMENT MATRIX")
        print("=" * 72)
        for row in experiments:
            state = "enabled" if row.get("enabled") else "disabled"
            print(f"- {row.get('id')} [{state}] -> {row.get('script')}")
        return

    selected = _select_experiments(matrix, args.run_id, args.run_enabled)
    if not selected:
        print("No experiments selected. Use --list, --run-id, or --run-enabled.")
        return

    stop_on_failure = args.stop_on_failure or bool(matrix.get("defaults", {}).get("stop_on_failure", True))
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = _resolve_output_root(matrix)
    run_root.mkdir(parents=True, exist_ok=True)
    run_log_path = run_root / f"matrix_run_{run_stamp}.json"

    print("=" * 72)
    print("RUN EXPERIMENT MATRIX")
    print("=" * 72)
    print(f"Matrix: {matrix_path}")
    print(f"Selected: {len(selected)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Output root: {run_root}")
    if log_path is not None:
        print(f"Full log: {log_path}")
        _append_log(log_path, "=" * 72 + "\n")
        _append_log(log_path, "RUN EXPERIMENT MATRIX\n")
        _append_log(log_path, "=" * 72 + "\n")
        _append_log(log_path, f"Matrix: {matrix_path}\n")
        _append_log(log_path, f"Selected: {len(selected)}\n")
        _append_log(log_path, f"Dry run: {args.dry_run}\n")
        _append_log(log_path, f"Output root: {run_root}\n")

    results: list[dict[str, Any]] = []
    completed_experiment_dirs: list[str] = []

    for idx, row in enumerate(selected, start=1):
        exp_id = row.get("id", f"exp_{idx}")
        script_rel = row.get("script")
        args_map = dict(row.get("args", {}))
        default_env_map = dict(matrix.get("defaults", {}).get("env", {}))
        env_map = {**default_env_map, **dict(row.get("env", {}))}
        if not script_rel:
            results.append({"id": exp_id, "status": "failed", "error": "Missing script field"})
            if stop_on_failure:
                break
            continue

        cmd = [args.python, script_rel, *_build_cli_args(args_map)]
        cmd_preview = " ".join(shlex.quote(part) for part in cmd)
        print(f"\n[{idx}/{len(selected)}] {exp_id}")
        print(f"Command: {cmd_preview}")
        if log_path is not None:
            _append_log(log_path, f"\n[{idx}/{len(selected)}] {exp_id}\n")
            _append_log(log_path, f"Command: {cmd_preview}\n")

        record: dict[str, Any] = {
            "id": exp_id,
            "script": script_rel,
            "args": args_map,
            "env": env_map,
            "command": cmd,
            "command_preview": cmd_preview,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "planned",
            "return_code": None,
            "duration_seconds": None,
            "experiment_dir": None,
            "error": None,
        }

        if args.dry_run:
            record["status"] = "dry-run"
            results.append(record)
            continue

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in env_map.items()})
        before_dirs = _experiment_dirs_snapshot()
        started = time.perf_counter()
        return_code = _run_command(cmd, cwd=cfg.PROJECT_ROOT, env=env, log_path=log_path)
        duration = time.perf_counter() - started
        after_dirs = _experiment_dirs_snapshot()
        new_dir = _pick_new_experiment_dir(before_dirs, after_dirs)

        record["return_code"] = return_code
        record["duration_seconds"] = duration
        record["experiment_dir"] = new_dir
        record["status"] = "completed" if return_code == 0 else "failed"
        if return_code != 0:
            record["error"] = f"Exit code {return_code}"
        if new_dir and return_code == 0:
            completed_experiment_dirs.append(new_dir)

        results.append(record)
        if log_path is not None:
            _append_log(log_path, f"Status: {record['status']}\n")
            _append_log(log_path, f"Duration seconds: {duration:.3f}\n")
            if new_dir:
                _append_log(log_path, f"Experiment dir: {new_dir}\n")
        if return_code != 0 and stop_on_failure:
            print("Stopping due to failure and stop-on-failure policy.")
            break

    compare_output = None
    if args.auto_compare and not args.dry_run:
        compare_output = _run_compare(completed_experiment_dirs, args.python, run_root, log_path=log_path)

    payload = {
        "matrix_path": str(matrix_path),
        "run_stamp": run_stamp,
        "dry_run": args.dry_run,
        "log_file": str(log_path) if log_path is not None else None,
        "selected_count": len(selected),
        "results": results,
        "completed_experiment_dirs": completed_experiment_dirs,
        "compare_output": compare_output,
    }
    run_log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("MATRIX RUN COMPLETE")
    print("=" * 72)
    print(f"Run log: {run_log_path}")
    if log_path is not None:
        print(f"Full text log: {log_path}")
    if compare_output:
        print(f"Compare report (md): {compare_output['report_md']}")
        print(f"Compare report (csv): {compare_output['report_csv']}")


if __name__ == "__main__":
    main()
