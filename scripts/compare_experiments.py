"""Compare saved experiment bundles."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings as cfg
from src.evaluation.experiment_io import read_json, read_jsonl, write_csv
from src.evaluation.reporting import build_comparison_report


def _load_experiment_summary(exp_dir: Path) -> dict:
    config = read_json(exp_dir / "config.json")
    summary = read_json(exp_dir / "summary.json") if (exp_dir / "summary.json").exists() else {}
    if isinstance(summary, list):
        rows = []
        for item in summary:
            rows.append(
                {
                    "experiment_id": config.get("experiment_id"),
                    "system_name": item.get("system_name", config.get("system_name")),
                    "dataset_name": config.get("dataset_name"),
                    "generation_model": config.get("generation_model"),
                    "embedding_model": config.get("embedding_model"),
                    "model_backend": config.get("model_backend"),
                    "query_mode": config.get("query_mode"),
                    **item,
                }
            )
        return rows
    return {
        "experiment_id": config.get("experiment_id"),
        "system_name": config.get("system_name"),
        "dataset_name": config.get("dataset_name"),
        "generation_model": config.get("generation_model"),
        "embedding_model": config.get("embedding_model"),
        "model_backend": config.get("model_backend"),
        "query_mode": config.get("query_mode"),
        **summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiments:
        rows = []
        for path in args.experiments:
            loaded = _load_experiment_summary(Path(path))
            if isinstance(loaded, list):
                rows.extend(loaded)
            else:
                rows.append(loaded)
    else:
        index_path = cfg.RESULTS_DIR / "experiments" / "index.jsonl"
        rows = read_jsonl(index_path) if index_path.exists() else []
    report = build_comparison_report(rows)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    write_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
