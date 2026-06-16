"""Prepare canonical benchmark data from a configured dataset adapter."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmark.adapters import build_dataset_adapter
from src.benchmark.chunking import ChunkingConfig, chunk_documents
from src.benchmark.runner import load_config
from src.config import settings as cfg
from src.evaluation.experiment_io import ensure_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare canonical documents/questions from a dataset config without running models."
    )
    parser.add_argument("--config", required=True, help="Benchmark config containing a dataset section.")
    parser.add_argument("--output-dir", help="Where to write prepared canonical data.")
    return parser.parse_args()


def _resolve_output_dir(config: dict[str, Any], override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        dataset = config.get("dataset", {})
        name = str(dataset.get("type", "dataset"))
        split = str(dataset.get("split", "default"))
        path = cfg.DATA_DIR / "prepared" / f"{name}_{split}"
    if not path.is_absolute():
        path = cfg.PROJECT_ROOT / path
    return path.resolve()


def _availability_summary(dataset, index_units) -> dict[str, Any]:
    question_count = len(dataset.questions)
    with_answers = sum(bool(q.gold_answers) for q in dataset.questions)
    with_gold_evidence = sum(bool(q.gold_evidence_ids) for q in dataset.questions)
    with_proxy_evidence = sum(bool(q.proxy_evidence_ids) for q in dataset.questions)
    return {
        "dataset_name": dataset.name,
        "document_count": len(dataset.documents),
        "index_unit_count": len(index_units),
        "question_count": question_count,
        "questions_with_gold_answers": with_answers,
        "questions_with_official_gold_evidence_ids": with_gold_evidence,
        "questions_with_proxy_evidence_ids": with_proxy_evidence,
        "answer_quality_metrics_available": with_answers == question_count and question_count > 0,
        "official_id_retrieval_metrics_available": with_gold_evidence == question_count and question_count > 0,
        "proxy_retrieval_metrics_available": with_proxy_evidence > 0,
        "notes": [
            "Gold answers support answer quality metrics such as exact match and token F1.",
            "Official evidence IDs are required for official evidence recall/MRR/nDCG.",
            "Proxy evidence IDs are approximate support diagnostics inferred from answer-containing documents.",
        ],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = _resolve_output_dir(config, args.output_dir)
    ensure_dir(output_dir)

    dataset = build_dataset_adapter(config.get("dataset", {"type": "hybridqa"})).load()
    chunk_config = ChunkingConfig(**{
        key: value
        for key, value in dict(config.get("chunking", {})).items()
        if key in ChunkingConfig.__dataclass_fields__
    })
    index_units = chunk_documents(dataset.documents, chunk_config)
    summary = _availability_summary(dataset, index_units)

    write_json(output_dir / "dataset_metadata.json", dataset.metadata)
    write_jsonl(output_dir / "canonical_documents.jsonl", [document.to_dict() for document in dataset.documents])
    write_jsonl(output_dir / "index_units.jsonl", [document.to_dict() for document in index_units])
    write_jsonl(output_dir / "questions.jsonl", [question.to_dict() for question in dataset.questions])
    write_json(output_dir / "evaluation_availability.json", summary)

    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
