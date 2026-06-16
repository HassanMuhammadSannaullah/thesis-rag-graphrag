"""Run one configured RAG vs GraphRAG benchmark experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmark.runner import load_config, run_standard_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standard RAG vs GraphRAG benchmark.")
    parser.add_argument("--config", required=True, help="Path to a benchmark JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_standard_benchmark(load_config(args.config))
    print(json.dumps({"output_dir": summary["output_dir"], "systems": summary["systems"]}, indent=2))


if __name__ == "__main__":
    main()
