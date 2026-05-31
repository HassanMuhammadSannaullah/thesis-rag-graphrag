"""Evaluate predictions against examples and save a reusable experiment bundle."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.evaluator import Evaluator
from src.evaluation.experiment_io import read_json, read_jsonl
from src.evaluation.schemas import EvaluationExample, SystemPrediction


def _read_examples(path: Path) -> list[EvaluationExample]:
    rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    return [EvaluationExample.from_dict(row) for row in rows]


def _read_predictions(path: Path, default_system_name: str) -> list[SystemPrediction]:
    rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    fixed = []
    for row in rows:
        row = dict(row)
        row.setdefault("system_name", row.get("system", default_system_name))
        fixed.append(SystemPrediction.from_dict(row))
    return fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--model-backend", required=True)
    parser.add_argument("--generation-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--dataset-version")
    parser.add_argument("--dataset-path")
    parser.add_argument("--query-mode")
    parser.add_argument("--experiment-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = _read_examples(Path(args.examples))
    predictions = _read_predictions(Path(args.predictions), args.system_name)
    experiment_id = args.experiment_id or Path(args.output_dir).name
    evaluator = Evaluator(
        dataset_name=args.dataset_name,
        system_name=args.system_name,
        model_backend=args.model_backend,
        generation_model=args.generation_model,
        embedding_model=args.embedding_model,
        experiment_id=experiment_id,
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        query_mode=args.query_mode,
        command=" ".join(sys.argv),
    )
    evaluator.evaluate(examples=examples, predictions=predictions, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
