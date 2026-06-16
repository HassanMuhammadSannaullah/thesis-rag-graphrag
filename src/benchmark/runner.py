"""Standard configurable benchmark runner."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.benchmark.adapters import build_dataset_adapter
from src.benchmark.baseline_rag import StandardRagConfig, StandardRagPipeline
from src.benchmark.chunking import ChunkingConfig, chunk_documents
from src.benchmark.graphrag_pipeline import StandardGraphRagPipeline, default_graphrag_api_key
from src.config import settings as cfg
from src.evaluation.evaluator import Evaluator
from src.evaluation.experiment_io import ensure_dir, write_json, write_jsonl
from src.evaluation.schemas import SystemPrediction


def _apply_model_config(models: dict[str, Any]) -> None:
    """Apply per-experiment model settings to the existing model client config."""
    if not models:
        return
    if models.get("backend"):
        cfg.MODEL_BACKEND = str(models["backend"]).lower()
    if models.get("generation_model"):
        if cfg.MODEL_BACKEND == "local_openai":
            cfg.LOCAL_GENERATION_MODEL = str(models["generation_model"])
            cfg.LOCAL_GRAPHRAG_INDEX_MODEL = str(models.get("graphrag_index_model", models["generation_model"]))
        else:
            cfg.GENERATION_MODEL = str(models["generation_model"])
    if models.get("embedding_model"):
        if cfg.MODEL_BACKEND == "local_openai":
            cfg.LOCAL_EMBEDDING_MODEL = str(models["embedding_model"])
        else:
            cfg.EMBEDDING_MODEL = str(models["embedding_model"])
    if models.get("embedding_dimension") is not None:
        if cfg.MODEL_BACKEND == "local_openai":
            cfg.LOCAL_EMBEDDING_DIMENSION = int(models["embedding_dimension"])
        else:
            cfg.EMBEDDING_DIMENSION = int(models["embedding_dimension"])
    if models.get("base_url"):
        cfg.LOCAL_LLM_BASE_URL = str(models["base_url"])
        cfg.LOCAL_LLM_BASE_URLS = [cfg.LOCAL_LLM_BASE_URL]
    if models.get("base_urls"):
        values = models["base_urls"]
        if isinstance(values, str):
            cfg.LOCAL_LLM_BASE_URLS = [value.strip() for value in values.split(",") if value.strip()]
        else:
            cfg.LOCAL_LLM_BASE_URLS = [str(value).strip() for value in values if str(value).strip()]
        if cfg.LOCAL_LLM_BASE_URLS:
            cfg.LOCAL_LLM_BASE_URL = cfg.LOCAL_LLM_BASE_URLS[0]
    if models.get("api_key"):
        if cfg.MODEL_BACKEND == "local_openai":
            cfg.LOCAL_LLM_API_KEY = str(models["api_key"])
        else:
            cfg.GOOGLE_API_KEY = str(models["api_key"])

    # Recreate clients on the next call if a config changed inside this process.
    try:
        from src.utils import model_client

        model_client._openai_clients.clear()
        model_client._openai_url_index = 0
        model_client._gemini_client = None
    except Exception:
        pass


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _apply_parallelism_config(parallelism: dict[str, Any]) -> None:
    """Apply per-experiment client-side LLM request concurrency settings."""
    shared = parallelism.get("llm_concurrent_requests")
    cfg.EMBEDDING_CONCURRENT_REQUESTS = _positive_int(
        parallelism.get("embedding_concurrent_requests", shared),
        cfg.DEFAULT_EMBEDDING_CONCURRENT_REQUESTS,
    )
    cfg.RETRIEVAL_CONCURRENT_REQUESTS = _positive_int(
        parallelism.get("retrieval_concurrent_requests", shared),
        cfg.DEFAULT_RETRIEVAL_CONCURRENT_REQUESTS,
    )
    cfg.EVALUATION_CONCURRENT_REQUESTS = _positive_int(
        parallelism.get("evaluation_concurrent_requests", shared),
        cfg.DEFAULT_EVALUATION_CONCURRENT_REQUESTS,
    )
    cfg.GRAPHRAG_CONCURRENT_REQUESTS = _positive_int(
        parallelism.get("graphrag_concurrent_requests", shared),
        cfg.DEFAULT_GRAPHRAG_CONCURRENT_REQUESTS,
    )


def _apply_graphrag_config(graphrag: dict[str, Any]) -> None:
    """Apply per-experiment GraphRAG runtime settings that live in cfg."""
    if graphrag.get("max_context_tokens") is not None:
        cfg.GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS = _positive_int(
            graphrag.get("max_context_tokens"),
            cfg.GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS,
        )


def _run_queries(questions, query_fn, max_workers: int) -> list[SystemPrediction]:
    if max_workers <= 1 or len(questions) <= 1:
        return [query_fn(question) for question in questions]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(query_fn, questions))


def _resolve_output_dir(config: dict[str, Any]) -> Path:
    output = config.get("output_dir")
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = cfg.PROJECT_ROOT / path
        return path.resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return cfg.RESULTS_DIR / "experiments" / f"standard_benchmark_{stamp}"


def _config_dataclass(cls, payload: dict[str, Any]):
    fields = cls.__dataclass_fields__.keys()
    return cls(**{key: value for key, value in payload.items() if key in fields})


def _failed_predictions(
    questions,
    *,
    system_name: str,
    error: str,
    metadata: dict[str, Any],
) -> list[SystemPrediction]:
    return [
        SystemPrediction(
            question_id=question.question_id,
            system_name=system_name,
            predicted_answer="ERROR",
            retrieved_contexts=[],
            error=error,
            metadata=metadata,
        )
        for question in questions
    ]


def run_standard_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    _apply_model_config(dict(config.get("models", {})))
    _apply_parallelism_config(dict(config.get("parallelism") or {}))
    _apply_graphrag_config(dict(config.get("graphrag") or {}))
    output_dir = _resolve_output_dir(config)
    ensure_dir(output_dir)
    write_json(output_dir / "benchmark_config.json", config)

    dataset = build_dataset_adapter(config.get("dataset", {"type": "hybridqa"})).load()
    chunk_config = _config_dataclass(ChunkingConfig, config.get("chunking", {}))
    index_units = chunk_documents(dataset.documents, chunk_config)

    write_jsonl(output_dir / "canonical_documents.jsonl", [document.to_dict() for document in dataset.documents])
    write_jsonl(output_dir / "index_units.jsonl", [document.to_dict() for document in index_units])
    write_jsonl(output_dir / "questions.jsonl", [question.to_dict() for question in dataset.questions])

    systems = config.get("systems", ["baseline"])
    predictions = []
    system_results: dict[str, Any] = {}

    if "baseline" in systems:
        rag_config = _config_dataclass(StandardRagConfig, config.get("baseline", {}))
        baseline = StandardRagPipeline(index_units, rag_config)
        baseline.build()
        baseline_predictions = _run_queries(
            dataset.questions,
            baseline.query,
            cfg.RETRIEVAL_CONCURRENT_REQUESTS,
        )
        predictions.extend(baseline_predictions)
        system_results["baseline"] = {
            "prediction_count": len(baseline_predictions),
            "backend": baseline.backend_metadata,
            "config": asdict(rag_config),
            "retrieval_concurrent_requests": cfg.RETRIEVAL_CONCURRENT_REQUESTS,
        }

    if "graphrag" in systems:
        graphrag_config = dict(config.get("graphrag", {}))
        workspace_dir = Path(graphrag_config.get("workspace_dir", output_dir / "graphrag_workspace"))
        if not workspace_dir.is_absolute():
            workspace_dir = (cfg.PROJECT_ROOT / workspace_dir).resolve()
        graphrag = StandardGraphRagPipeline(
            workspace_dir=workspace_dir,
            api_key=graphrag_config.get("api_key") or default_graphrag_api_key(),
            query_method=graphrag_config.get("query_method", "local"),
            response_type=graphrag_config.get("response_type", "Single sentence"),
            force_rebuild=bool(graphrag_config.get("force_rebuild", False)),
            index_method=graphrag_config.get("index_method", "standard"),
            extract_graph_max_gleanings=int(graphrag_config.get("extract_graph_max_gleanings", 0)),
        )
        try:
            graphrag.build(index_units if graphrag_config.get("use_chunked_units", True) else dataset.documents)
            graphrag_predictions = graphrag.query_many(
                dataset.questions,
                max_concurrent=cfg.RETRIEVAL_CONCURRENT_REQUESTS,
            )
            graph_status = "completed"
            graph_error = None
        except Exception as exc:
            graph_status = "failed"
            graph_error = str(exc)
            graphrag_predictions = _failed_predictions(
                dataset.questions,
                system_name="standard_graphrag",
                error=f"GraphRAG indexing/query failed: {graph_error}",
                metadata={
                    "workspace_dir": str(workspace_dir),
                    "query_method": graphrag.query_method,
                    "response_type": graphrag.response_type,
                    "failure_stage": "graphrag_build_or_query",
                },
            )
        predictions.extend(graphrag_predictions)
        system_results["graphrag"] = {
            "prediction_count": len(graphrag_predictions),
            "status": graph_status,
            "error": graph_error,
            "workspace_dir": str(workspace_dir),
            "config": graphrag_config,
            "retrieval_concurrent_requests": cfg.RETRIEVAL_CONCURRENT_REQUESTS,
            "graphrag_concurrent_requests": cfg.GRAPHRAG_CONCURRENT_REQUESTS,
        }

    write_jsonl(output_dir / "predictions.jsonl", [prediction.to_dict() for prediction in predictions])

    metrics_by_system: dict[str, Any] = {}
    examples = dataset.evaluation_examples()
    for system_name in sorted({prediction.system_name for prediction in predictions}):
        system_predictions = [prediction for prediction in predictions if prediction.system_name == system_name]
        result_key = "baseline" if system_name == "standard_hybrid_rag" else "graphrag"
        evaluator = Evaluator(
            dataset_name=dataset.name,
            system_name=system_name,
            model_backend=cfg.MODEL_BACKEND,
            generation_model=cfg.LOCAL_GENERATION_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL,
            embedding_model=cfg.LOCAL_EMBEDDING_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.EMBEDDING_MODEL,
            experiment_id=f"{output_dir.name}_{system_name}",
            dataset_version=dataset.metadata.get("split"),
            dataset_path=str(config.get("dataset", {}).get("path", "")),
            query_mode="standard_benchmark",
            command="scripts/run_benchmark.py",
            run_metadata={
                "benchmark_dataset": dataset.metadata,
                "chunking": asdict(chunk_config),
                "system_results": system_results.get(result_key, {}),
            },
        )
        metrics_by_system[system_name] = evaluator.evaluate(
            examples=examples,
            predictions=system_predictions,
            output_dir=output_dir / system_name,
        )

    summary = {
        "output_dir": str(output_dir),
        "dataset": dataset.metadata,
        "document_count": len(dataset.documents),
        "index_unit_count": len(index_units),
        "question_count": len(dataset.questions),
        "systems": system_results,
        "metrics_by_system": metrics_by_system,
    }
    write_json(output_dir / "benchmark_summary.json", summary)
    _write_comparison_report(output_dir, metrics_by_system)
    return summary


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = cfg.PROJECT_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def _first_overall_row(result: dict[str, Any]) -> dict[str, Any]:
    aggregate = result.get("aggregate_metrics", {})
    rows = aggregate.get("overall") or aggregate.get("system_name") or []
    return rows[0] if rows else {}


def _metric_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    mean_key = f"{key}_mean"
    return row.get(mean_key)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_comparison_report(output_dir: Path, metrics_by_system: dict[str, Any]) -> None:
    if not metrics_by_system:
        return
    rows = {name: _first_overall_row(result) for name, result in metrics_by_system.items()}
    payload = {"systems": rows}
    baseline = rows.get("standard_hybrid_rag")
    graphrag = rows.get("standard_graphrag")
    if baseline and graphrag:
        deltas = {}
        for key, value in graphrag.items():
            base_value = baseline.get(key)
            if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                deltas[key] = value - base_value
        payload["graphrag_minus_baseline"] = deltas
    write_json(output_dir / "comparison_summary.json", payload)

    display_metrics = [
        "count",
        "failure_count",
        "normalized_exact_match",
        "token_f1",
        "answer_contains_gold",
        "answer_support_mrr",
        "answer_support_hit_at_5",
        "proxy_evidence_recall_at_5",
        "unsupported_prediction",
        "likely_hallucination",
        "latency_seconds",
        "retrieved_context_count",
    ]

    lines = ["# Benchmark Comparison", ""]
    if baseline and graphrag:
        lines.extend(
            [
                "| Metric | Baseline | GraphRAG | GraphRAG - Baseline |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key in display_metrics:
            base_value = _metric_value(baseline, key)
            graph_value = _metric_value(graphrag, key)
            delta_value = payload.get("graphrag_minus_baseline", {}).get(key)
            lines.append(
                f"| `{key}` | {_format_metric(base_value)} | "
                f"{_format_metric(graph_value)} | {_format_metric(delta_value)} |"
            )
        lines.append("")

    for system_name, row in rows.items():
        lines.append(f"## {system_name}")
        for key in display_metrics:
            value = _metric_value(row, key)
            if value is not None:
                lines.append(f"- `{key}`: {_format_metric(value)}")
        lines.append("")
        failure_examples = [
            metric_row
            for metric_row in metrics_by_system.get(system_name, {}).get("per_question_metrics", [])
            if metric_row.get("error")
        ][:3]
        if failure_examples:
            lines.append("### Failure Examples")
            for metric_row in failure_examples:
                lines.append(f"- `{metric_row.get('question_id')}` {metric_row.get('error')}")
            lines.append("")
    if payload.get("graphrag_minus_baseline"):
        lines.append("## GraphRAG Minus Baseline")
        for key in display_metrics:
            value = payload["graphrag_minus_baseline"].get(key)
            if value is not None:
                lines.append(f"- `{key}`: {_format_metric(value)}")
    (output_dir / "comparison_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
