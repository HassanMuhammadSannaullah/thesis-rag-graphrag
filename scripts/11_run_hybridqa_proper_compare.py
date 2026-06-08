"""
Fair HybridQA comparison runner.

Comparison design:
  - baseline RAG: vector retrieval over the exact GraphRAG text units
  - GraphRAG: full GraphRAG indexing + local query mode

This keeps both systems grounded in the same underlying evidence units while
still comparing plain vector retrieval against graph-aware retrieval.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jsonlines
import pandas as pd
import requests
from graphrag.config.load_config import load_config
from graphrag.data_model.data_reader import DataReader
from graphrag.query.factory import get_basic_search_engine, get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.base import SearchResult
from graphrag.utils.api import get_embedding_store, load_search_prompt
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baseline.vector_index import LocalVectorIndex
from src.config.model_registry import build_local_model_registry_snapshot
from src.config import settings as cfg
from src.data_pipeline.hybridqa_evidence import attach_proxy_evidence
from src.evaluation.evaluator import Evaluator
from src.evaluation.schemas import EvaluationExample, RetrievedContext, SystemPrediction
from src.graphrag_system.corpus_prep import hybridqa_record_to_text, hybridqa_to_graphrag_docs
from src.graphrag_system.runner import create_graphrag_config, has_graphrag_index, run_graphrag_index
from src.utils.model_client import generate_text, probe_local_backend
from src.utils.runtime import resolve_project_python


UNIFIED_QA_PROMPT = """---Role---

You answer HybridQA-style questions using only the provided context.

---Rules---

- Use only facts present in the provided table content, linked passages, and retrieved evidence.
- Many questions require combining table facts with linked-entity descriptions.
- Return the shortest exact answer you can.
- Do not add citations, bullets, markdown, or commentary.
- If the answer is a number, date, name, nationality, or title, output just that field.
- If the context does not support a confident answer, say: I do not know.

---Target response length and format---

{response_type}

---Data tables---

{context_data}
"""


LOCAL_GRAPHRAG_SEARCH_KWARGS = {
    "community_prop": cfg.GRAPHRAG_LOCAL_COMMUNITY_PROP,
    "text_unit_prop": cfg.GRAPHRAG_LOCAL_TEXT_UNIT_PROP,
    "top_k_mapped_entities": cfg.GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES,
    "top_k_relationships": cfg.GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS,
    "max_context_tokens": cfg.GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS,
    "include_relationship_weight": cfg.GRAPHRAG_LOCAL_INCLUDE_RELATIONSHIP_WEIGHT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--systems", default="baseline,graphrag")
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--force-query", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--graphrag-query-method", default="local", choices=["local", "basic"])
    parser.add_argument("--baseline-top-k", type=int, default=8)
    parser.add_argument("--baseline-max-context-chars", type=int, default=cfg.FAIR_BASELINE_MAX_CONTEXT_CHARS)
    parser.add_argument("--baseline-max-answer-tokens", type=int, default=cfg.FAIR_BASELINE_MAX_ANSWER_TOKENS)
    return parser.parse_args()


def active_generation_model() -> str:
    return cfg.LOCAL_GENERATION_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL


def active_embedding_model() -> str:
    return cfg.LOCAL_EMBEDDING_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.EMBEDDING_MODEL


def active_model_api_key() -> str:
    return cfg.LOCAL_LLM_API_KEY if cfg.MODEL_BACKEND == "local_openai" else cfg.GOOGLE_API_KEY


def local_server_health() -> dict[str, Any] | None:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(cfg.LOCAL_LLM_BASE_URL)
        base_root = f"{parsed.scheme}://{parsed.netloc}/"
    except Exception:
        base_root = f"http://{cfg.LOCAL_SERVER_HOST}:{cfg.LOCAL_SERVER_PORT}/"

    # 1. Check custom HF local server health endpoint on the target host
    try:
        response = requests.get(
            base_root.rstrip("/") + "/health",
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    # 2. Check Ollama root endpoint on the target host
    try:
        response = requests.get(
            base_root,
            timeout=5,
        )
        if response.status_code == 200 and "Ollama" in response.text:
            return {
                "status": "ok",
                "generation_model": cfg.LOCAL_GENERATION_MODEL,
                "embedding_model": cfg.LOCAL_EMBEDDING_MODEL,
                "device": "external_ollama",
            }
    except Exception:
        pass

    # 3. Check standard /v1/models endpoint using LOCAL_LLM_BASE_URL
    try:
        url = cfg.LOCAL_LLM_BASE_URL.rstrip("/") + "/models"
        headers = {}
        if cfg.LOCAL_LLM_API_KEY:
            headers["Authorization"] = f"Bearer {cfg.LOCAL_LLM_API_KEY}"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return {
                "status": "ok",
                "generation_model": cfg.LOCAL_GENERATION_MODEL,
                "embedding_model": cfg.LOCAL_EMBEDDING_MODEL,
                "device": "external_openai",
            }
    except Exception:
        pass

    return None


def ensure_local_server() -> subprocess.Popen[str] | None:
    if cfg.MODEL_BACKEND != "local_openai":
        return None
    if local_server_health() is not None:
        print("[Local Server] Existing server detected. Warming generation and embedding models ...")
        probe = probe_local_backend()
        print(
            "[Local Server] Probe OK "
            f"({probe['generation_model']} / {probe['embedding_model']})"
        )
        return None

    print("[Local Server] Starting local OpenAI-compatible server ...")
    out_log = cfg.LOGS_DIR / "fair_compare_local_server.out.log"
    err_log = cfg.LOGS_DIR / "fair_compare_local_server.err.log"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with out_log.open("w", encoding="utf-8") as stdout_handle, err_log.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            [str(resolve_project_python()), "scripts/local_openai_server.py"],
            cwd=str(cfg.PROJECT_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )

    for _ in range(60):
        if process.poll() is not None:
            break
        health = local_server_health()
        if health is not None:
            print(f"[Local Server] Ready on {cfg.LOCAL_SERVER_HOST}:{cfg.LOCAL_SERVER_PORT}")
            print("[Local Server] Warming generation and embedding models ...")
            probe = probe_local_backend()
            print(
                "[Local Server] Probe OK "
                f"({probe['generation_model']} / {probe['embedding_model']})"
            )
            return process
        time.sleep(2)

    raise RuntimeError(
        "Local server failed to start. Check "
        f"{out_log} and {err_log}."
    )


def stop_local_server(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()


def load_hybridqa_records(split: str, question_limit: int | None = None) -> list[dict]:
    path = cfg.ORIGINAL_DIR / f"{split}.jsonl"
    records: list[dict] = []
    with jsonlines.open(str(path)) as reader:
        for record in reader:
            records.append(attach_proxy_evidence(record))
            if question_limit is not None and len(records) >= question_limit:
                break
    return records


def unique_table_records(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for record in records:
        table_id = record["table_id"]
        if table_id in seen:
            continue
        seen.add(table_id)
        unique.append(record)
    return unique


def generate_baseline_answer(
    question: str,
    contexts: list[dict],
    *,
    max_context_chars: int,
    max_answer_tokens: int,
) -> str:
    context_text = "\n\n".join(
        f"[Text Unit {idx}] {ctx['text']}" for idx, ctx in enumerate(contexts, start=1)
    )
    prompt = UNIFIED_QA_PROMPT.format(
        context_data=context_text[:max_context_chars],
        response_type="Single sentence",
    )
    prompt = f"{prompt}\n\nQuestion: {question}\nAnswer:"
    return generate_text(prompt, max_tokens=max_answer_tokens).strip()


def build_fairness_protocol(args: argparse.Namespace, records: list[dict]) -> dict[str, Any]:
    protocol = {
        "protocol_version": "v1",
        "systems": sorted({part.strip().lower() for part in args.systems.split(",") if part.strip()}),
        "query_budget": {
            "question_count": len(records),
            "question_limit_arg": args.question_limit,
        },
        "baseline": {
            "top_k": args.baseline_top_k,
            "max_context_chars": args.baseline_max_context_chars,
            "max_answer_tokens": args.baseline_max_answer_tokens,
        },
        "graphrag": {
            "mode": "pure",
            "query_method": args.graphrag_query_method,
            "report_community_level": cfg.GRAPHRAG_LOCAL_REPORT_COMMUNITY_LEVEL,
            "entity_community_level": None,
            "local_search_params": LOCAL_GRAPHRAG_SEARCH_KWARGS,
        },
        "shared_input_mode": "shared_graphrag_text_units_for_baseline",
    }
    checks: list[str] = []
    if args.baseline_top_k <= 0:
        checks.append("baseline_top_k must be > 0")
    if args.baseline_max_context_chars <= 0:
        checks.append("baseline_max_context_chars must be > 0")
    if args.baseline_max_answer_tokens <= 0:
        checks.append("baseline_max_answer_tokens must be > 0")
    if "baseline" not in protocol["systems"] and "graphrag" not in protocol["systems"]:
        checks.append("At least one system must be selected")
    if checks:
        raise ValueError("Fairness protocol validation failed: " + "; ".join(checks))
    return protocol


def write_hybridqa_graphrag_prompts(project_dir: Path) -> None:
    prompts_dir = project_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for prompt_name in ("local_search_system_prompt.txt", "basic_search_system_prompt.txt"):
        (prompts_dir / prompt_name).write_text(UNIFIED_QA_PROMPT, encoding="utf-8")


def prepare_graphrag_project(records: list[dict], *, split: str, force_reindex: bool) -> Path:
    project_dir = cfg.PROJECT_ROOT / "graphrag_workspace" / f"hybridqa_{split}_fair_compare_full"
    if force_reindex:
        for path in (project_dir / "output", project_dir / "cache", project_dir / "logs"):
            if path.exists():
                shutil.rmtree(path)

    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for old_file in input_dir.glob("*.txt"):
        old_file.unlink()

    hybridqa_to_graphrag_docs(
        unique_table_records(records),
        input_dir,
        max_passages_per_record=cfg.MAX_LINKED_PASSAGES,
    )
    create_graphrag_config(project_dir, api_key=active_model_api_key(), force=False)
    write_hybridqa_graphrag_prompts(project_dir)
    return project_dir


async def load_graphrag_dataframes(project_dir: Path, query_method: str) -> tuple[object, dict[str, object]]:
    config = load_config(root_dir=project_dir)
    storage_obj = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage_obj)
    reader = DataReader(table_provider)

    required = ["text_units"]
    optional: list[str] = []
    if query_method == "local":
        required = ["communities", "community_reports", "text_units", "relationships", "entities"]
        optional = ["covariates"]

    dataframes: dict[str, object] = {}
    for name in required:
        dataframes[name] = await getattr(reader, name)()

    for name in optional:
        if await table_provider.has(name):
            dataframes[name] = await getattr(reader, name)()
        else:
            dataframes[name] = None
    return config, dataframes


async def build_graphrag_search_engine(project_dir: Path, query_method: str):
    config, dataframes = await load_graphrag_dataframes(project_dir, query_method)

    if query_method == "basic":
        embedding_store = get_embedding_store(config.vector_store, "text_unit_text")
        prompt = load_search_prompt(config.basic_search.prompt)
        return get_basic_search_engine(
            config=config,
            text_units=read_indexer_text_units(dataframes["text_units"]),
            text_unit_embeddings=embedding_store,
            response_type="Single sentence",
            system_prompt=prompt,
            callbacks=[],
        )

    if query_method == "local":
        description_embedding_store = get_embedding_store(config.vector_store, "entity_description")
        prompt = load_search_prompt(config.local_search.prompt)
        communities = dataframes["communities"]
        community_reports = dataframes["community_reports"]
        entities = dataframes["entities"]
        relationships = dataframes["relationships"]
        covariates = dataframes.get("covariates")
        engine = get_local_search_engine(
            config=config,
            reports=read_indexer_reports(
                community_reports,
                communities,
                community_level=cfg.GRAPHRAG_LOCAL_REPORT_COMMUNITY_LEVEL,
            ),
            text_units=read_indexer_text_units(dataframes["text_units"]),
            entities=read_indexer_entities(entities, communities, community_level=None),
            relationships=read_indexer_relationships(relationships),
            covariates={"claims": read_indexer_covariates(covariates) if covariates is not None else []},
            description_embedding_store=description_embedding_store,
            response_type="Single sentence",
            system_prompt=prompt,
            callbacks=[],
        )
        engine.context_builder_params.update(LOCAL_GRAPHRAG_SEARCH_KWARGS)
        return engine

    raise RuntimeError(f"Unsupported GraphRAG query method for this script: {query_method}")


def ensure_graphrag_index(project_dir: Path, records: list[dict], force_reindex: bool) -> None:
    if force_reindex or not has_graphrag_index(project_dir):
        # GraphRAG CLI 3.x currently emits a logging-format exception on
        # successful dry runs (`logger.info(..., True)`), which makes the
        # validation step noisy and unreliable for unattended experiment runs.
        # Go straight to the full index invocation instead.
        print("\n[GraphRAG] Skipping dry-run validation due to GraphRAG CLI logging bug ...")
        print("\n[GraphRAG] Full indexing ...")
        timeout_seconds = 86400  # always allow up to 24h; local 3B model is slow
        if not run_graphrag_index(
            project_dir,
            dry_run=False,
            verbose=True,
            method="standard",
            timeout_seconds=timeout_seconds,
        ):
            raise RuntimeError("GraphRAG indexing failed.")
    else:
        print("\n[GraphRAG] Existing full index found, skipping indexing.")


def load_shared_text_units(project_dir: Path) -> list[dict[str, Any]]:
    text_units_path = project_dir / "output" / "text_units.parquet"
    frame = pd.read_parquet(text_units_path)
    units: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        units.append(
            {
                "id": str(row["id"]),
                "text": str(row["text"]),
                "document_id": str(row.get("document_id")),
                "human_readable_id": row.get("human_readable_id"),
                "n_tokens": row.get("n_tokens"),
                "type": "graphrag_text_unit",
            }
        )
    return units


def build_shared_text_unit_index(project_dir: Path, *, split: str, force_reindex: bool) -> LocalVectorIndex:
    index_path = cfg.INDEX_CACHE_DIR / f"hybridqa_{split}_shared_text_unit_rag"
    if force_reindex and index_path.exists():
        shutil.rmtree(index_path)

    index = LocalVectorIndex(index_path)
    shared_units = load_shared_text_units(project_dir)
    texts = [unit["text"] for unit in shared_units]
    metadatas = [
        {
            "id": unit["id"],
            "type": unit["type"],
            "text": unit["text"],
            "document_id": unit["document_id"],
            "human_readable_id": unit["human_readable_id"],
            "n_tokens": unit["n_tokens"],
        }
        for unit in shared_units
    ]
    index.add(texts, metadatas, batch_size=8)
    return index


async def search_graphrag(engine, question: str) -> SearchResult:
    return await engine.search(question)


def serialize_graphrag_contexts(search_result: SearchResult) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    context_text = search_result.context_text

    if isinstance(context_text, dict):
        for rank, (section_name, text_value) in enumerate(context_text.items(), start=1):
            text = str(text_value).strip()
            if not text:
                continue
            contexts.append(
                {
                    "id": f"{section_name}:{rank}",
                    "text": text,
                    "score": None,
                    "rank": rank,
                    "source_type": section_name,
                    "metadata": {},
                }
            )
        return contexts

    if isinstance(context_text, list):
        for rank, text_value in enumerate(context_text, start=1):
            text = str(text_value).strip()
            if not text:
                continue
            contexts.append(
                {
                    "id": f"context:{rank}",
                    "text": text,
                    "score": None,
                    "rank": rank,
                    "source_type": "context_chunk",
                    "metadata": {},
                }
            )
        return contexts

    text = str(context_text).strip()
    if text:
        contexts.append(
            {
                "id": "context:1",
                "text": text,
                "score": None,
                "rank": 1,
                "source_type": "context_chunk",
                "metadata": {},
            }
        )
    return contexts


def normalize_graphrag_response(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, (dict, list)):
        return json.dumps(response, ensure_ascii=False).strip()
    return str(response).strip()


async def run_standard_rag(
    records: list[dict],
    *,
    project_dir: Path,
    split: str,
    force_query: bool,
    force_reindex: bool,
    top_k: int,
    max_context_chars: int,
    max_answer_tokens: int,
) -> list[dict]:
    output_path = cfg.OUTPUTS_DIR / f"hybridqa_{split}_fair_standard_rag_results.json"
    index = build_shared_text_unit_index(project_dir, split=split, force_reindex=force_reindex)

    existing = {}
    if output_path.exists() and not force_query:
        with open(output_path, encoding="utf-8") as f:
            for row in json.load(f):
                existing[row["question_id"]] = row

    results_dict = {}
    # Prefill existing results
    for record in records:
        qid = record["question_id"]
        if qid in existing:
            results_dict[qid] = existing[qid]

    to_run = [r for r in records if r["question_id"] not in results_dict]
    total_to_run = len(to_run)

    if total_to_run > 0:
        sem = asyncio.Semaphore(8)
        completed_count = 0

        async def worker(record: dict) -> dict:
            nonlocal completed_count
            qid = record["question_id"]
            async with sem:
                started = time.perf_counter()
                retrieved = index.search(record["question"], top_k=min(top_k, index.size))
                try:
                    answer = await asyncio.to_thread(
                        generate_baseline_answer,
                        record["question"],
                        retrieved,
                        max_context_chars=max_context_chars,
                        max_answer_tokens=max_answer_tokens,
                    )
                    error = None
                except Exception as exc:
                    answer = f"ERROR: {exc}"
                    error = str(exc)
                latency = time.perf_counter() - started

            res = {
                "question_id": qid,
                "question": record["question"],
                "gold_answer": record["answer"],
                "predicted_answer": answer,
                "system": "baseline",
                "table_id": record["table_id"],
                "retrieved_contexts": [
                    {
                        "id": item["id"],
                        "text": item.get("text"),
                        "score": item.get("score"),
                        "rank": rank,
                        "source_type": item.get("type"),
                        "metadata": {
                            "document_id": item.get("document_id"),
                            "human_readable_id": item.get("human_readable_id"),
                            "n_tokens": item.get("n_tokens"),
                        },
                    }
                    for rank, item in enumerate(retrieved, start=1)
                ],
                "latency_seconds": latency,
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": error,
            }
            results_dict[qid] = res
            completed_count += 1
            if completed_count % 25 == 0 or completed_count == total_to_run:
                print(f"[Shared-Unit RAG] Answered {completed_count}/{total_to_run} new questions (total: {len(results_dict)}/{len(records)})")
                ordered_results = [results_dict[r["question_id"]] for r in records if r["question_id"] in results_dict]
                output_path.write_text(json.dumps(ordered_results, indent=2, ensure_ascii=False), encoding="utf-8")
            return res

        await asyncio.gather(*(worker(r) for r in to_run))

    ordered_results = [results_dict[r["question_id"]] for r in records if r["question_id"] in results_dict]
    output_path.write_text(json.dumps(ordered_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return ordered_results


async def run_full_graphrag(
    records: list[dict],
    *,
    project_dir: Path,
    split: str,
    force_query: bool,
    query_method: str,
) -> list[dict]:
    output_path = cfg.OUTPUTS_DIR / f"hybridqa_{split}_fair_full_graphrag_{query_method}_results.json"
    search_engine = await build_graphrag_search_engine(project_dir, query_method)

    existing = {}
    if output_path.exists() and not force_query:
        with open(output_path, encoding="utf-8") as f:
            for row in json.load(f):
                existing[row["question_id"]] = row

    results_dict = {}
    # Prefill existing results
    for record in records:
        qid = record["question_id"]
        if qid in existing:
            results_dict[qid] = existing[qid]

    to_run = [r for r in records if r["question_id"] not in results_dict]
    total_to_run = len(to_run)

    if total_to_run > 0:
        sem = asyncio.Semaphore(8)
        completed_count = 0

        async def worker(record: dict) -> dict:
            nonlocal completed_count
            qid = record["question_id"]
            async with sem:
                try:
                    search_result = await search_graphrag(search_engine, record["question"])
                    answer = normalize_graphrag_response(search_result.response)
                    retrieved_contexts = serialize_graphrag_contexts(search_result)
                    latency = search_result.completion_time
                    prompt_tokens = search_result.prompt_tokens
                    output_tokens = search_result.output_tokens
                    total_tokens = (prompt_tokens or 0) + (output_tokens or 0)
                    error = None
                except Exception as exc:
                    answer = f"ERROR: {exc}"
                    retrieved_contexts = []
                    latency = None
                    prompt_tokens = None
                    output_tokens = None
                    total_tokens = None
                    error = str(exc)

            res = {
                "question_id": qid,
                "question": record["question"],
                "gold_answer": record["answer"],
                "predicted_answer": answer,
                "system": "graphrag",
                "table_id": record["table_id"],
                "workspace": str(project_dir),
                "method": query_method,
                "retrieved_contexts": retrieved_contexts,
                "latency_seconds": latency,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "error": error,
            }
            results_dict[qid] = res
            completed_count += 1
            if completed_count % 25 == 0 or completed_count == total_to_run:
                print(f"[GraphRAG {query_method}] Answered {completed_count}/{total_to_run} new questions (total: {len(results_dict)}/{len(records)})")
                ordered_results = [results_dict[r["question_id"]] for r in records if r["question_id"] in results_dict]
                output_path.write_text(json.dumps(ordered_results, indent=2, ensure_ascii=False), encoding="utf-8")
            return res

        await asyncio.gather(*(worker(r) for r in to_run))

    ordered_results = [results_dict[r["question_id"]] for r in records if r["question_id"] in results_dict]
    output_path.write_text(json.dumps(ordered_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return ordered_results


def build_examples(records: list[dict], split: str) -> list[EvaluationExample]:
    return [
        EvaluationExample(
            question_id=record["question_id"],
            question=record["question"],
            gold_answer=record["answer"],
            gold_evidence=record.get("gold_evidence"),
            proxy_evidence=record.get("proxy_evidence"),
            evidence_label_mode=(record.get("evidence_alignment") or {}).get("label_mode"),
            question_type="hybridqa",
            operation_type="table_plus_text",
            difficulty=None,
            answer_type=None,
            metadata={
                "source_dataset": "hybridqa",
                "split": split,
                "table_id": record["table_id"],
                "evidence_alignment": record.get("evidence_alignment"),
            },
        )
        for record in records
    ]


def to_predictions(rows: list[dict], system_name: str) -> list[SystemPrediction]:
    predictions = []
    for row in rows:
        contexts = row.get("retrieved_contexts")
        predictions.append(
            SystemPrediction(
                question_id=row["question_id"],
                system_name=system_name,
                predicted_answer=row.get("predicted_answer", ""),
                retrieved_contexts=(
                    [RetrievedContext.from_dict(ctx) for ctx in contexts]
                    if contexts is not None
                    else None
                ),
                latency_seconds=row.get("latency_seconds"),
                prompt_tokens=row.get("prompt_tokens"),
                output_tokens=row.get("output_tokens"),
                total_tokens=row.get("total_tokens"),
                error=row.get("error"),
                metadata={"workspace": row.get("workspace"), "method": row.get("method")},
            )
        )
    return predictions


def evaluate_run(
    *,
    split: str,
    records: list[dict],
    baseline_rows: list[dict],
    graphrag_rows: list[dict],
    query_method: str,
    fairness_protocol: dict[str, Any],
    model_registry_snapshot: dict[str, Any],
) -> Path:
    experiment_id = f"hybridqa_{split}_fair_compare_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = cfg.RESULTS_DIR / "experiments" / experiment_id
    predictions = to_predictions(baseline_rows, "baseline") + to_predictions(graphrag_rows, "graphrag")
    evaluator = Evaluator(
        dataset_name=f"hybridqa_{split}_fair_compare",
        system_name="shared_text_unit_rag_vs_full_graphrag",
        model_backend=cfg.MODEL_BACKEND,
        generation_model=active_generation_model(),
        embedding_model=active_embedding_model(),
        experiment_id=experiment_id,
        dataset_version=split,
        dataset_path=str(cfg.ORIGINAL_DIR / f"{split}.jsonl"),
        query_mode=query_method,
        command=" ".join(sys.argv),
        run_metadata={
            "fairness_protocol": fairness_protocol,
            "local_model_registry": model_registry_snapshot,
        },
    )
    evaluator.evaluate(examples=build_examples(records, split), predictions=predictions, output_dir=output_dir)
    return output_dir


def main() -> None:
    args = parse_args()
    systems = {part.strip().lower() for part in args.systems.split(",") if part.strip()}
    local_server_process = ensure_local_server()

    try:
        records = load_hybridqa_records(args.split, args.question_limit)
        fairness_protocol = build_fairness_protocol(args, records)
        model_registry_snapshot = build_local_model_registry_snapshot()

        print("=" * 60)
        print("FAIR HYBRIDQA COMPARISON")
        print("=" * 60)
        print(f"Split: {args.split}")
        print(f"Backend: {cfg.MODEL_BACKEND}")
        print(f"Generation model: {active_generation_model()}")
        print(f"Embedding model: {active_embedding_model()}")
        print(f"Questions: {len(records)}")
        print(f"Unique tables: {len(unique_table_records(records))}")
        print(f"GraphRAG query mode: {args.graphrag_query_method}")
        print("GraphRAG mode: pure graph-only local search")
        print(
            "GraphRAG local params: "
            f"community_prop={cfg.GRAPHRAG_LOCAL_COMMUNITY_PROP}, "
            f"text_unit_prop={cfg.GRAPHRAG_LOCAL_TEXT_UNIT_PROP}, "
            f"top_k_entities={cfg.GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES}, "
            f"top_k_relationships={cfg.GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS}, "
            f"max_context_tokens={cfg.GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS}"
        )
        print("Fairness mode: shared GraphRAG text units for baseline RAG")
        print(f"Baseline context cap: {args.baseline_max_context_chars} chars")
        print(f"Baseline answer cap: {args.baseline_max_answer_tokens} tokens")
        print(f"Evaluation metrics: {cfg.EVALUATION_FRAMEWORK}")

        project_dir = prepare_graphrag_project(records, split=args.split, force_reindex=args.force_reindex)

        if args.prepare_only:
            print("[GraphRAG] Project prepared.")
            return

        ensure_graphrag_index(project_dir, records, args.force_reindex)

        baseline_rows: list[dict] = []
        graphrag_rows: list[dict] = []

        if "baseline" in systems:
            baseline_rows = asyncio.run(
                run_standard_rag(
                    records,
                    project_dir=project_dir,
                    split=args.split,
                    force_query=args.force_query,
                    force_reindex=args.force_reindex,
                    top_k=args.baseline_top_k,
                    max_context_chars=args.baseline_max_context_chars,
                    max_answer_tokens=args.baseline_max_answer_tokens,
                )
            )

        if "graphrag" in systems:
            graphrag_rows = asyncio.run(
                run_full_graphrag(
                    records,
                    project_dir=project_dir,
                    split=args.split,
                    force_query=args.force_query,
                    query_method=args.graphrag_query_method,
                )
            )

        experiment_dir = evaluate_run(
            split=args.split,
            records=records,
            baseline_rows=baseline_rows,
            graphrag_rows=graphrag_rows,
            query_method=args.graphrag_query_method,
            fairness_protocol=fairness_protocol,
            model_registry_snapshot=model_registry_snapshot,
        )
        print(f"\nExperiment bundle saved to {experiment_dir}")
    finally:
        stop_local_server(local_server_process)


if __name__ == "__main__":
    main()
