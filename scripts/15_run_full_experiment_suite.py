"""One-command runner for shareable thesis experiments.

This script bootstraps the HybridQA dataset, parses the required split(s), and
then launches the selected experiment matrix entries.
"""
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
    parser.add_argument("--run-id", action="append", default=[], help="Run only selected experiment id(s)")
    parser.add_argument("--run-enabled", action="store_true", help="Run all enabled experiments from the matrix")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Matrix must contain an 'experiments' list")
    return payload


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


def _preview(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _run(cmd: list[str], *, dry_run: bool, log_path: Path, capture_to_log: bool = True) -> None:
    print(f"Command: {_preview(cmd)}")
    _append_log(log_path, f"Command: {_preview(cmd)}\n")
    if dry_run:
        return
    process = subprocess.Popen(
        cmd,
        cwd=str(cfg.PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if capture_to_log:
            _append_log(log_path, line)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)


def _required_splits(experiments: list[dict[str, Any]]) -> list[str]:
    splits: set[str] = set()
    for row in experiments:
        split = str(row.get("args", {}).get("split", "dev")).strip().lower()
        if split == "all":
            splits.update({"dev", "train"})
        elif split:
            splits.add(split)
    return sorted(splits or {"dev"})


def _ensure_dataset_downloaded(args: argparse.Namespace, *, log_path: Path) -> None:
    raw_targets = [
        cfg.RAW_DIR / "train.json",
        cfg.RAW_DIR / "dev.json",
        cfg.RAW_DIR / "WikiTables-WithLinks.zip",
    ]
    missing = [path for path in raw_targets if not path.exists()]
    if not missing or args.skip_download:
        return
    print("\n[1/3] Downloading HybridQA raw files")
    _append_log(log_path, "\n[1/3] Downloading HybridQA raw files\n")
    _run([args.python, "scripts/01_download_hybridqa.py"], dry_run=args.dry_run, log_path=log_path)


def _ensure_dataset_parsed(args: argparse.Namespace, splits: list[str], *, log_path: Path) -> None:
    if args.skip_parse:
        return
    for split in splits:
        parsed_path = cfg.ORIGINAL_DIR / f"{split}.jsonl"
        if parsed_path.exists():
            continue
        print(f"\n[2/3] Parsing HybridQA split: {split}")
        _append_log(log_path, f"\n[2/3] Parsing HybridQA split: {split}\n")
        _run([args.python, "scripts/02_parse_hybridqa.py", "--split", split], dry_run=args.dry_run, log_path=log_path)


def _run_matrix(args: argparse.Namespace, *, log_path: Path) -> None:
    cmd = [args.python, "scripts/14_run_experiment_matrix.py", "--matrix-path", args.matrix_path, "--python", args.python]
    if args.run_id:
        for run_id in args.run_id:
            cmd.extend(["--run-id", run_id])
    else:
        cmd.append("--run-enabled")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.stop_on_failure:
        cmd.append("--stop-on-failure")
    if not args.skip_compare:
        cmd.append("--auto-compare")
    cmd.extend(["--log-file", str(log_path)])
    print("\n[3/3] Running experiment matrix")
    _append_log(log_path, "\n[3/3] Running experiment matrix\n")
    _run(cmd, dry_run=args.dry_run, log_path=log_path, capture_to_log=False)


def main() -> None:
    args = parse_args()
    if not args.run_id and not args.run_enabled:
        args.run_enabled = True

    matrix_path = Path(args.matrix_path)
    if not matrix_path.is_absolute():
        matrix_path = (cfg.PROJECT_ROOT / matrix_path).resolve()
    matrix = _load_matrix(matrix_path)
    selected = _select_experiments(matrix, args.run_id, args.run_enabled)
    if not selected:
        print("No experiments selected. Use --run-id or enable experiments in the matrix.")
        return

    splits = _required_splits(selected)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = cfg.LOGS_DIR / f"full_experiment_suite_{run_stamp}.log"
    print("=" * 72)
    print("FULL EXPERIMENT SUITE")
    print("=" * 72)
    print(f"Matrix: {matrix_path}")
    print(f"Selected experiments: {len(selected)}")
    print(f"Required splits: {', '.join(splits)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Full log: {log_path}")
    _append_log(log_path, "=" * 72 + "\n")
    _append_log(log_path, "FULL EXPERIMENT SUITE\n")
    _append_log(log_path, "=" * 72 + "\n")
    _append_log(log_path, f"Matrix: {matrix_path}\n")
    _append_log(log_path, f"Selected experiments: {len(selected)}\n")
    _append_log(log_path, f"Required splits: {', '.join(splits)}\n")
    _append_log(log_path, f"Dry run: {args.dry_run}\n")
    _append_log(log_path, f"Full log: {log_path}\n")

    _ensure_dataset_downloaded(args, log_path=log_path)
    _ensure_dataset_parsed(args, splits, log_path=log_path)
    _run_matrix(args, log_path=log_path)


if __name__ == "__main__":
    main()
