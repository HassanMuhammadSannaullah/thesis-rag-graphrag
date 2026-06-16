"""Main evaluation pipeline with selectable metric frameworks."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from src.config import settings as cfg

from .aggregate import aggregate_rows, flatten_aggregate_tables, summarize_group
from .answer_metrics import compute_answer_metrics
from .experiment_io import append_jsonl, ensure_dir, timestamp_utc, write_csv, write_json, write_jsonl
from .hallucination_metrics import compute_hallucination_metrics
from .metric_catalog import classic_canonical_metrics, comparison_k_values
from .ragas_support import (
    RAGAS_CANONICAL_METRICS,
    build_default_ragas_metrics,
    build_ragas_embeddings,
    build_ragas_llm,
    canonicalize_ragas_metrics,
    load_ragas_runtime,
)
from .reporting import build_experiment_report
from .retrieval_metrics import compute_answer_support_metrics, compute_id_retrieval_metrics
from .schemas import EvaluationExample, MetricResult, SystemPrediction
from .statistics import summarize_statistical_report
from src.utils.runtime import detect_hardware_snapshot, detect_runtime_environment


class Evaluator:
    def __init__(
        self,
        *,
        dataset_name: str,
        system_name: str,
        model_backend: str,
        generation_model: str,
        embedding_model: str,
        experiment_id: str,
        dataset_version: str | None = None,
        dataset_path: str | None = None,
        query_mode: str | None = None,
        command: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.system_name = system_name
        self.model_backend = model_backend
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.experiment_id = experiment_id
        self.dataset_version = dataset_version
        self.dataset_path = dataset_path
        self.query_mode = query_mode
        self.command = command
        self.run_metadata = run_metadata or {}
        self.evaluation_framework = cfg.EVALUATION_FRAMEWORK
        self._ragas_runtime = load_ragas_runtime() if self.evaluation_framework == "ragas" else None

    def _git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cfg.PROJECT_ROOT,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def config_dict(self) -> dict[str, Any]:
        runtime_environment = detect_runtime_environment()
        hardware_snapshot = detect_hardware_snapshot(disk_path=cfg.PROJECT_ROOT)
        if self.evaluation_framework == "ragas":
            metric_config = {
                "framework": "ragas",
                "ragas_version": self._ragas_runtime.version if self._ragas_runtime else None,
                "metrics": RAGAS_CANONICAL_METRICS,
                "evaluation_backend": cfg.RAGAS_EVAL_BACKEND,
                "evaluation_generation_model": cfg.RAGAS_EVAL_GENERATION_MODEL,
                "evaluation_embedding_model": cfg.RAGAS_EVAL_EMBEDDING_MODEL,
            }
        else:
            metric_config = {
                "framework": "classic",
                "metrics": classic_canonical_metrics(cfg.TOP_K_RETRIEVAL),
                "judge_metrics_enabled": False,
                "retrieval_k": cfg.TOP_K_RETRIEVAL,
            }

        return {
            "experiment_id": self.experiment_id,
            "timestamp": timestamp_utc(),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_path": self.dataset_path,
            "system_name": self.system_name,
            "model_backend": self.model_backend,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "query_mode": self.query_mode,
            "git_commit_hash": self._git_commit(),
            "command": self.command,
            "runtime_environment": runtime_environment,
            "hardware_snapshot": hardware_snapshot,
            "parallelism": {
                "embedding_concurrent_requests": cfg.EMBEDDING_CONCURRENT_REQUESTS,
                "retrieval_concurrent_requests": cfg.RETRIEVAL_CONCURRENT_REQUESTS,
                "evaluation_concurrent_requests": cfg.EVALUATION_CONCURRENT_REQUESTS,
                "graphrag_concurrent_requests": cfg.GRAPHRAG_CONCURRENT_REQUESTS,
            },
            "run_metadata": self.run_metadata,
            "metric_config": metric_config,
        }

    def _classic_metrics_for_pair(
        self,
        *,
        example: EvaluationExample,
        prediction: SystemPrediction,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        if prediction.retrieved_contexts is None:
            warnings.append("missing_retrieved_contexts")
        elif not prediction.retrieved_contexts:
            warnings.append("empty_retrieved_contexts")
        if prediction.error:
            errors.append(prediction.error)

        answer_metrics = compute_answer_metrics(
            prediction.predicted_answer or "",
            example.all_gold_answers(),
            answer_type=example.answer_type,
        )
        retrieval_k_values = comparison_k_values(cfg.TOP_K_RETRIEVAL)
        answer_support_metrics, answer_support_warnings = compute_answer_support_metrics(
            prediction.retrieved_contexts,
            gold_answers=example.all_gold_answers(),
            k_values=retrieval_k_values,
        )
        strict_retrieval_metrics, retrieval_warnings = compute_id_retrieval_metrics(
            example.gold_evidence,
            prediction.retrieved_contexts,
            k_values=retrieval_k_values,
            prefix="evidence",
            missing_warning="missing_gold_evidence",
        )
        proxy_retrieval_metrics, proxy_retrieval_warnings = compute_id_retrieval_metrics(
            example.proxy_evidence,
            prediction.retrieved_contexts,
            k_values=retrieval_k_values,
            prefix="proxy_evidence",
            missing_warning="missing_proxy_evidence",
        )
        warnings.extend(answer_support_warnings)
        warnings.extend(retrieval_warnings)
        warnings.extend(proxy_retrieval_warnings)
        grounding_metrics = compute_hallucination_metrics(
            predicted_answer=prediction.predicted_answer or "",
            gold_answers=example.all_gold_answers(),
            retrieved_contexts=prediction.retrieved_contexts,
            answer_metrics=answer_metrics,
            judge_metrics={},
        )

        precision_key = f"evidence_precision_at_{cfg.TOP_K_RETRIEVAL}"
        recall_key = f"evidence_recall_at_{cfg.TOP_K_RETRIEVAL}"
        metrics = {
            **answer_metrics,
            **answer_support_metrics,
            **strict_retrieval_metrics,
            **proxy_retrieval_metrics,
            **grounding_metrics,
            # Compatibility fields expected by report/statistics code paths.
            "answer_correctness": float(answer_metrics.get("normalized_exact_match", 0)),
            "answer_relevancy": float(answer_metrics.get("answer_contains_gold", 0)),
            "semantic_similarity": None,
            "factual_correctness": (
                float(answer_metrics["numeric_value_match_tolerant"])
                if answer_metrics.get("numeric_value_match_tolerant") is not None
                else None
            ),
            "faithfulness": None,
            "context_precision": strict_retrieval_metrics.get(precision_key),
            "context_recall": strict_retrieval_metrics.get(recall_key),
            "latency_seconds": prediction.latency_seconds,
            "prompt_tokens": prediction.prompt_tokens,
            "output_tokens": prediction.output_tokens,
            "total_tokens": prediction.total_tokens,
            "retrieved_context_count": len(prediction.retrieved_contexts or []),
        }
        return metrics, warnings, errors

    def evaluate(
        self,
        *,
        examples: list[EvaluationExample],
        predictions: list[SystemPrediction],
        output_dir: Path,
        append_index: bool = True,
    ) -> dict[str, Any]:
        ensure_dir(output_dir)
        example_by_id = {example.question_id: example for example in examples}
        predictions_rows: list[dict[str, Any]] = []
        paired_examples: list[tuple[EvaluationExample, SystemPrediction]] = []

        for prediction in predictions:
            example = example_by_id.get(prediction.question_id)
            if example is None:
                continue

            predictions_rows.append(
                {
                    "question_id": example.question_id,
                    "question": example.question,
                    "gold_answer": example.gold_answer,
                    "gold_evidence": example.gold_evidence,
                    "proxy_evidence": example.proxy_evidence,
                    "evidence_label_mode": example.evidence_label_mode,
                    "predicted_answer": prediction.predicted_answer,
                    "system_name": prediction.system_name,
                    "dataset_name": self.dataset_name,
                    "model_backend": self.model_backend,
                    "generation_model": self.generation_model,
                    "embedding_model": self.embedding_model,
                    "question_type": example.question_type,
                    "operation_type": example.operation_type,
                    "difficulty": example.difficulty,
                    "retrieved_contexts": [ctx.to_dict() for ctx in (prediction.retrieved_contexts or [])],
                    "raw_system_metadata": prediction.metadata,
                    "latency_seconds": prediction.latency_seconds,
                    "prompt_tokens": prediction.prompt_tokens,
                    "output_tokens": prediction.output_tokens,
                    "total_tokens": prediction.total_tokens,
                    "error": prediction.error,
                    "warnings": [],
                }
            )
            paired_examples.append((example, prediction))

        ragas_rows: list[dict[str, Any]] = []
        if self.evaluation_framework == "ragas":
            if self._ragas_runtime is None:
                raise RuntimeError("Ragas runtime is unavailable while evaluation framework is 'ragas'.")
            ragas_samples = [
                self._ragas_runtime.SingleTurnSample(
                    user_input=example.question,
                    response=prediction.predicted_answer or "",
                    retrieved_contexts=[
                        ctx.text or ctx.id or ""
                        for ctx in (prediction.retrieved_contexts or [])
                        if (ctx.text or ctx.id)
                    ],
                    reference=str(example.all_gold_answers()[0]) if example.all_gold_answers() else "",
                )
                for example, prediction in paired_examples
            ]

            ragas_llm = build_ragas_llm(
                self._ragas_runtime,
                generation_model=cfg.RAGAS_EVAL_GENERATION_MODEL,
            )
            ragas_embeddings = build_ragas_embeddings(
                self._ragas_runtime,
                embedding_model=cfg.RAGAS_EVAL_EMBEDDING_MODEL,
            )
            ragas_run_config = self._ragas_runtime.RunConfig(
                timeout=600,
                max_retries=2,
                max_wait=30,
                max_workers=cfg.EVALUATION_CONCURRENT_REQUESTS,
            )
            ragas_dataset = self._ragas_runtime.EvaluationDataset(samples=ragas_samples)
            ragas_result = self._ragas_runtime.evaluate(
                dataset=ragas_dataset,
                metrics=build_default_ragas_metrics(
                    self._ragas_runtime,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                ),
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=ragas_run_config,
                raise_exceptions=False,
                show_progress=False,
                batch_size=cfg.EVALUATION_CONCURRENT_REQUESTS,
            )
            ragas_rows = ragas_result.to_pandas().to_dict(orient="records")

        metric_results: list[MetricResult] = []
        if self.evaluation_framework == "ragas":
            paired_rows = zip(paired_examples, ragas_rows)
        else:
            paired_rows = ((pair, None) for pair in paired_examples)

        for (example, prediction), ragas_row in paired_rows:
            if self.evaluation_framework == "ragas":
                warnings: list[str] = []
                errors: list[str] = []
                if prediction.retrieved_contexts is None:
                    warnings.append("missing_retrieved_contexts")
                elif not prediction.retrieved_contexts:
                    warnings.append("empty_retrieved_contexts")
                if prediction.error:
                    errors.append(prediction.error)
                metrics = {
                    **canonicalize_ragas_metrics(ragas_row or {}),
                    "latency_seconds": prediction.latency_seconds,
                    "prompt_tokens": prediction.prompt_tokens,
                    "output_tokens": prediction.output_tokens,
                    "total_tokens": prediction.total_tokens,
                    "retrieved_context_count": len(prediction.retrieved_contexts or []),
                }
            else:
                metrics, warnings, errors = self._classic_metrics_for_pair(
                    example=example,
                    prediction=prediction,
                )
            metric_results.append(
                MetricResult(
                    question_id=example.question_id,
                    system_name=prediction.system_name,
                    dataset_name=self.dataset_name,
                    metrics=metrics,
                    warnings=warnings,
                    errors=errors,
                    metadata={
                        "question_type": example.question_type,
                        "operation_type": example.operation_type,
                        "difficulty": example.difficulty,
                    },
                )
            )

        metric_rows = []
        for result, prediction_row in zip(metric_results, predictions_rows):
            row = {
                "experiment_id": self.experiment_id,
                "dataset_name": self.dataset_name,
                "system_name": result.system_name,
                "model_backend": self.model_backend,
                "generation_model": self.generation_model,
                "embedding_model": self.embedding_model,
                "question_id": result.question_id,
                "question": prediction_row["question"],
                "gold_answer": prediction_row["gold_answer"],
                "gold_evidence": prediction_row["gold_evidence"],
                "proxy_evidence": prediction_row["proxy_evidence"],
                "evidence_label_mode": prediction_row["evidence_label_mode"],
                "predicted_answer": prediction_row["predicted_answer"],
                "question_type": result.metadata.get("question_type"),
                "operation_type": result.metadata.get("operation_type"),
                "difficulty": result.metadata.get("difficulty"),
                "warnings": result.warnings,
                "errors": result.errors,
                "error": prediction_row["error"],
                "metrics": result.metrics,
                **result.metrics,
            }
            metric_rows.append(row)

        aggregate = aggregate_rows(
            metric_rows,
            group_fields=["system_name", "dataset_name", "generation_model", "question_type", "operation_type", "difficulty"],
        )
        statistical_report = summarize_statistical_report(metric_rows)
        flat_aggregate = flatten_aggregate_tables(aggregate)
        config = self.config_dict()

        write_json(output_dir / "config.json", config)
        write_jsonl(output_dir / "predictions.jsonl", predictions_rows)
        write_jsonl(output_dir / "per_question_metrics.jsonl", metric_rows)
        write_json(output_dir / "aggregate_metrics.json", aggregate)
        write_json(output_dir / "statistical_report.json", statistical_report)
        write_csv(output_dir / "aggregate_metrics.csv", flat_aggregate)

        overall_rows = []
        if aggregate.get("system_name"):
            for row in aggregate["system_name"]:
                overall_rows.append(
                    {
                        "system_name": row.get("group"),
                        "dataset_name": self.dataset_name,
                        "generation_model": self.generation_model,
                        **row,
                    }
                )
        elif aggregate.get("overall"):
            overall_rows = [
                {
                    "system_name": self.system_name,
                    "dataset_name": self.dataset_name,
                    "generation_model": self.generation_model,
                    **aggregate["overall"][0],
                }
            ]

        by_type_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in metric_rows:
            qtype = row.get("question_type")
            if qtype is None:
                continue
            key = (row.get("system_name"), qtype)
            by_type_groups.setdefault(key, []).append(row)
        by_type_rows = []
        for (system_name, qtype), rows in by_type_groups.items():
            by_type_rows.append(
                {
                    "system_name": system_name,
                    "group": qtype,
                    **summarize_group(rows),
                }
            )

        retrieval_rows = [{"group": "overall", **aggregate["overall"][0]}] if aggregate.get("overall") else []
        report = build_experiment_report(
            config=config,
            overall_rows=overall_rows,
            by_type_rows=by_type_rows,
            retrieval_rows=retrieval_rows,
            metric_rows=metric_rows,
            statistical_report=statistical_report,
        )
        write_json(
            output_dir / "summary.json",
            overall_rows if len(overall_rows) != 1 else (overall_rows[0] if overall_rows else {}),
        )
        (output_dir / "report.md").write_text(report, encoding="utf-8")

        if append_index:
            for summary in overall_rows or [{}]:
                summary_row = {
                    "experiment_id": self.experiment_id,
                    "timestamp": config["timestamp"],
                    "dataset_name": self.dataset_name,
                    "system_name": summary.get("system_name", self.system_name),
                    "model_backend": self.model_backend,
                    "generation_model": self.generation_model,
                    "embedding_model": self.embedding_model,
                    "query_mode": self.query_mode,
                    **summary,
                }
                append_jsonl(cfg.RESULTS_DIR / "experiments" / "index.jsonl", summary_row)

        return {
            "config": config,
            "predictions": predictions_rows,
            "per_question_metrics": metric_rows,
            "aggregate_metrics": aggregate,
            "statistical_report": statistical_report,
            "report_path": output_dir / "report.md",
        }
