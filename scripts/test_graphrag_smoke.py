"""
GraphRAG smoke test — mirrors the EXACT steps of the production pipeline
(scripts/11_run_hybridqa_proper_compare.py) on 5 synthetic HybridQA records.

Pipeline steps verified (in order):
  1. Local server start / health check      (ensure_local_server equivalent)
  2. Records → GraphRAG text-file documents  (hybridqa_to_graphrag_docs)
  3. GraphRAG settings.yaml + prompts        (create_graphrag_config)
  4. GraphRAG dry-run validation             (run_graphrag_index dry_run=True)
  5. Full GraphRAG indexing                  (run_graphrag_index dry_run=False)
  6. Load all output parquet tables          (communities, entities, relationships,
                                              text_units, community_reports)
  7. Build local search engine               (get_local_search_engine)
  8. Local search query per question         (engine.search per record)
  9. Response normalization                  (normalize_graphrag_response)
 10. Result dict shape validation            (same keys as production output rows)
 11. Ragas evaluation metrics                (Evaluator.evaluate on 5 examples)

Exit 0 = all steps pass. Exit 1 = something is broken.

Usage:
    $env:MODEL_BACKEND="local_openai"
    $env:LOCAL_GENERATION_MODEL="Qwen/Qwen2.5-3B-Instruct"
    $env:LOCAL_EMBEDDING_MODEL="intfloat/e5-base-v2"
    python scripts/test_graphrag_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings as cfg
from src.evaluation.evaluator import Evaluator
from src.evaluation.schemas import EvaluationExample, RetrievedContext, SystemPrediction
from src.graphrag_system.corpus_prep import hybridqa_to_graphrag_docs
from src.graphrag_system.runner import create_graphrag_config, run_graphrag_index
from src.utils.model_client import probe_local_backend
from src.utils.runtime import resolve_project_python

# ---------------------------------------------------------------------------
# Identical 5 synthetic records used by the baseline smoke test
# (must stay in sync — both tests share the same GraphRAG workspace)
# ---------------------------------------------------------------------------
SMOKE_RECORDS: list[dict] = [
    {
        "question_id": "smoke_001",
        "question": "What is the nationality of the player who scored 42 goals?",
        "answer": "Brazilian",
        "table_id": "smoke_scorers",
        "split": "smoke",
        "table": {
            "title": "Top Scorers",
            "section_title": "Season Statistics",
            "section_text": "",
            "intro": "Top goal scorers of the season.",
            "headers": ["Player", "Goals", "Nationality"],
            "rows": [
                {"Player": "Luciano Silva", "Goals": "42", "Nationality": "Brazilian", "_links": ["/wiki/Luciano_Silva"]},
                {"Player": "Marc Dubois", "Goals": "31", "Nationality": "French", "_links": []},
                {"Player": "Kenji Watanabe", "Goals": "28", "Nationality": "Japanese", "_links": []},
            ],
            "num_rows": 3,
            "all_links": ["/wiki/Luciano_Silva"],
        },
        "linked_passages": [
            {"link": "/wiki/Luciano_Silva", "text": "Luciano Silva is a Brazilian professional footballer known for his prolific goal-scoring ability in the top division."},
        ],
        "num_linked_passages": 1,
    },
    {
        "question_id": "smoke_002",
        "question": "Which company was founded in 1994 and is headquartered in Seattle?",
        "answer": "Amazon",
        "table_id": "smoke_tech",
        "split": "smoke",
        "table": {
            "title": "Major Technology Companies",
            "section_title": "Founded",
            "section_text": "",
            "intro": "Overview of major technology companies and their founding details.",
            "headers": ["Company", "Year Founded", "Headquarters"],
            "rows": [
                {"Company": "Amazon", "Year Founded": "1994", "Headquarters": "Seattle", "_links": ["/wiki/Amazon_(company)"]},
                {"Company": "Google", "Year Founded": "1998", "Headquarters": "Mountain View", "_links": []},
                {"Company": "Meta", "Year Founded": "2004", "Headquarters": "Menlo Park", "_links": []},
            ],
            "num_rows": 3,
            "all_links": ["/wiki/Amazon_(company)"],
        },
        "linked_passages": [
            {"link": "/wiki/Amazon_(company)", "text": "Amazon was founded by Jeff Bezos on July 5, 1994, and is headquartered in Seattle, Washington. It started as an online bookstore."},
        ],
        "num_linked_passages": 1,
    },
    {
        "question_id": "smoke_003",
        "question": "In what city were the 2012 Summer Olympics held?",
        "answer": "London",
        "table_id": "smoke_olympics",
        "split": "smoke",
        "table": {
            "title": "Summer Olympics Host Cities",
            "section_title": "Recent Games",
            "section_text": "",
            "intro": "Cities that hosted the Summer Olympic Games.",
            "headers": ["Year", "Host City", "Country"],
            "rows": [
                {"Year": "2012", "Host City": "London", "Country": "United Kingdom", "_links": ["/wiki/2012_Summer_Olympics"]},
                {"Year": "2016", "Host City": "Rio de Janeiro", "Country": "Brazil", "_links": []},
                {"Year": "2020", "Host City": "Tokyo", "Country": "Japan", "_links": []},
            ],
            "num_rows": 3,
            "all_links": ["/wiki/2012_Summer_Olympics"],
        },
        "linked_passages": [
            {"link": "/wiki/2012_Summer_Olympics", "text": "The 2012 Summer Olympics, officially the Games of the XXX Olympiad, were held in London, United Kingdom, from 27 July to 12 August 2012."},
        ],
        "num_linked_passages": 1,
    },
    {
        "question_id": "smoke_004",
        "question": "Who became CEO of Microsoft in 2014?",
        "answer": "Satya Nadella",
        "table_id": "smoke_ceos",
        "split": "smoke",
        "table": {
            "title": "Technology Company CEOs",
            "section_title": "Leadership Changes",
            "section_text": "",
            "intro": "Chief Executive Officers of major technology companies and the year they assumed the role.",
            "headers": ["Company", "CEO", "Year Appointed"],
            "rows": [
                {"Company": "Microsoft", "CEO": "Satya Nadella", "Year Appointed": "2014", "_links": ["/wiki/Satya_Nadella"]},
                {"Company": "Apple", "CEO": "Tim Cook", "Year Appointed": "2011", "_links": []},
                {"Company": "Google", "CEO": "Sundar Pichai", "Year Appointed": "2015", "_links": []},
            ],
            "num_rows": 3,
            "all_links": ["/wiki/Satya_Nadella"],
        },
        "linked_passages": [
            {"link": "/wiki/Satya_Nadella", "text": "Satya Narayana Nadella is an Indian-American business executive who became the CEO of Microsoft Corporation in February 2014, succeeding Steve Ballmer."},
        ],
        "num_linked_passages": 1,
    },
    {
        "question_id": "smoke_005",
        "question": "What programming language was created by Guido van Rossum?",
        "answer": "Python",
        "table_id": "smoke_languages",
        "split": "smoke",
        "table": {
            "title": "Programming Languages and Their Creators",
            "section_title": "Origins",
            "section_text": "",
            "intro": "Notable programming languages and the people who created them.",
            "headers": ["Language", "Creator", "Year"],
            "rows": [
                {"Language": "Python", "Creator": "Guido van Rossum", "Year": "1991", "_links": ["/wiki/Guido_van_Rossum"]},
                {"Language": "Java", "Creator": "James Gosling", "Year": "1995", "_links": []},
                {"Language": "C++", "Creator": "Bjarne Stroustrup", "Year": "1985", "_links": []},
            ],
            "num_rows": 3,
            "all_links": ["/wiki/Guido_van_Rossum"],
        },
        "linked_passages": [
            {"link": "/wiki/Guido_van_Rossum", "text": "Guido van Rossum is a Dutch programmer best known as the creator of the Python programming language, which he began developing in the late 1980s and first released in 1991."},
        ],
        "num_linked_passages": 1,
    },
]

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

SMOKE_PROJECT_DIR = cfg.PROJECT_ROOT / "graphrag_workspace" / "smoke_graphrag"
LOCAL_GRAPHRAG_SEARCH_KWARGS = {
    "community_prop": cfg.GRAPHRAG_LOCAL_COMMUNITY_PROP,
    "text_unit_prop": cfg.GRAPHRAG_LOCAL_TEXT_UNIT_PROP,
    "top_k_mapped_entities": cfg.GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES,
    "top_k_relationships": cfg.GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS,
    "include_relationship_weight": cfg.GRAPHRAG_LOCAL_INCLUDE_RELATIONSHIP_WEIGHT,
}


# ---------------------------------------------------------------------------
# Exact copies of helpers from 11_run_hybridqa_proper_compare.py
# ---------------------------------------------------------------------------

def normalize_graphrag_response(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, (dict, list)):
        return json.dumps(response, ensure_ascii=False).strip()
    return str(response).strip()


def serialize_graphrag_contexts(search_result) -> list[dict]:
    contexts: list[dict] = []
    context_text = search_result.context_text

    if isinstance(context_text, dict):
        for rank, (section_name, text_value) in enumerate(context_text.items(), start=1):
            text = str(text_value).strip()
            if text:
                contexts.append({"id": f"{section_name}:{rank}", "text": text,
                                  "score": None, "rank": rank,
                                  "source_type": section_name, "metadata": {}})
        return contexts

    if isinstance(context_text, list):
        for rank, text_value in enumerate(context_text, start=1):
            text = str(text_value).strip()
            if text:
                contexts.append({"id": f"context:{rank}", "text": text,
                                  "score": None, "rank": rank,
                                  "source_type": "context_chunk", "metadata": {}})
        return contexts

    text = str(context_text).strip()
    if text:
        contexts.append({"id": "context:1", "text": text, "score": None,
                          "rank": 1, "source_type": "context_chunk", "metadata": {}})
    return contexts


def check_server() -> bool:
    try:
        resp = requests.get(
            f"http://{cfg.LOCAL_SERVER_HOST}:{cfg.LOCAL_SERVER_PORT}/health",
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


def start_server() -> subprocess.Popen:
    print("  Starting local server (loading 3B model - may take 1-3 min)...")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "local_openai_server.py")],
        env={**os.environ},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(90):
        time.sleep(2)
        if check_server():
            print("  Server ready.")
            return proc
    proc.terminate()
    raise RuntimeError("Local server did not become ready in 3 minutes.")


def load_graphrag_dataframes(project_dir: Path) -> tuple:
    from graphrag.config.load_config import load_config
    from graphrag.data_model.data_reader import DataReader
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider

    config = load_config(root_dir=project_dir)
    storage_obj = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage_obj)
    reader = DataReader(table_provider)

    dataframes = {
        "communities": asyncio.run(reader.communities()),
        "community_reports": asyncio.run(reader.community_reports()),
        "entities": asyncio.run(reader.entities()),
        "relationships": asyncio.run(reader.relationships()),
        "text_units": asyncio.run(reader.text_units()),
    }
    if asyncio.run(table_provider.has("covariates")):
        dataframes["covariates"] = asyncio.run(reader.covariates())
    else:
        dataframes["covariates"] = None

    return config, dataframes


def build_local_search_engine(project_dir: Path):
    from graphrag.query.factory import get_local_search_engine
    from graphrag.query.indexer_adapters import (
        read_indexer_covariates, read_indexer_entities,
        read_indexer_relationships, read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.utils.api import get_embedding_store, load_search_prompt

    config, dataframes = load_graphrag_dataframes(project_dir)
    description_embedding_store = get_embedding_store(config.vector_store, "entity_description")
    prompt = load_search_prompt(config.local_search.prompt)
    covariates = dataframes.get("covariates")

    engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(
            dataframes["community_reports"],
            dataframes["communities"],
            community_level=cfg.GRAPHRAG_LOCAL_REPORT_COMMUNITY_LEVEL,
        ),
        text_units=read_indexer_text_units(dataframes["text_units"]),
        entities=read_indexer_entities(dataframes["entities"], dataframes["communities"], community_level=None),
        relationships=read_indexer_relationships(dataframes["relationships"]),
        covariates={"claims": read_indexer_covariates(covariates) if covariates is not None else []},
        description_embedding_store=description_embedding_store,
        response_type="Single sentence",
        system_prompt=prompt,
        callbacks=[],
    )
    engine.context_builder_params.update(LOCAL_GRAPHRAG_SEARCH_KWARGS)
    return engine


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        stdout_handle = getattr(proc, "_codex_stdout_handle", None)
        stderr_handle = getattr(proc, "_codex_stderr_handle", None)
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def warm_local_server() -> None:
    print("  Running local generation + embedding probe ...")
    probe = probe_local_backend()
    print(
        "  Probe OK "
        f"({probe['generation_model']} / {probe['embedding_model']}, "
        f"{probe['embedding_dimension']} dims)"
    )


def start_server() -> subprocess.Popen:
    print("  Starting local server (model warm-up may take a few minutes)...")
    out_log = cfg.LOGS_DIR / "graphrag_smoke_local_server.out.log"
    err_log = cfg.LOGS_DIR / "graphrag_smoke_local_server.err.log"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    out_handle = out_log.open("w", encoding="utf-8")
    err_handle = err_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(resolve_project_python()), str(Path(__file__).parent / "local_openai_server.py")],
        env={**os.environ},
        stdout=out_handle,
        stderr=err_handle,
        creationflags=creationflags,
    )
    proc._codex_stdout_handle = out_handle  # type: ignore[attr-defined]
    proc._codex_stderr_handle = err_handle  # type: ignore[attr-defined]
    for _ in range(90):
        time.sleep(2)
        if check_server():
            print("  Server ready.")
            return proc
    proc.terminate()
    out_handle.close()
    err_handle.close()
    raise RuntimeError(
        "Local server did not become ready in 3 minutes. "
        f"Check {out_log} and {err_log}."
    )


def run_smoke_tests() -> bool:
    all_passed = True
    server_proc = None

    # ---- STEP 1: Server -------------------------------------------------------
    print("\n[Step 1/11] Local server health check")
    if check_server():
        print(f"  {PASS} Server already running at {cfg.LOCAL_SERVER_HOST}:{cfg.LOCAL_SERVER_PORT}")
    else:
        try:
            server_proc = start_server()
            print(f"  {PASS} Server started.")
        except RuntimeError as exc:
            print(f"  {FAIL} {exc}")
            return False
    try:
        warm_local_server()
        print(f"  {PASS} Generation and embedding probe passed.")
    except Exception as exc:
        print(f"  {FAIL} Probe failed: {exc}")
        stop_server(server_proc)
        return False

    # ---- STEP 2: Prepare GraphRAG project ------------------------------------
    print(f"\n[Step 2/11] Prepare GraphRAG project - {len(SMOKE_RECORDS)} documents")
    try:
        if SMOKE_PROJECT_DIR.exists():
            shutil.rmtree(SMOKE_PROJECT_DIR)
        input_dir = SMOKE_PROJECT_DIR / "input"
        input_dir.mkdir(parents=True)
        hybridqa_to_graphrag_docs(
            SMOKE_RECORDS,
            input_dir,
            max_passages_per_record=cfg.MAX_LINKED_PASSAGES,
        )
        api_key = cfg.LOCAL_LLM_API_KEY if cfg.MODEL_BACKEND == "local_openai" else cfg.GOOGLE_API_KEY
        create_graphrag_config(SMOKE_PROJECT_DIR, api_key=api_key, force=True)
        doc_count = len(list(input_dir.glob("*.txt")))
        print(f"  {PASS} {doc_count} text documents written")
        print(f"  {PASS} settings.yaml written")
    except Exception as exc:
        print(f"  {FAIL} Project setup failed: {exc}")
        stop_server(server_proc)
        return False

    # ---- STEP 3: GraphRAG prompts check --------------------------------------
    print("\n[Step 3/11] Verify required prompt files exist")
    required_prompts = [
        "extract_graph.txt",
        "summarize_descriptions.txt",
        "community_report_graph.txt",
        "community_report_text.txt",
        "local_search_system_prompt.txt",
    ]
    prompts_dir = SMOKE_PROJECT_DIR / "prompts"
    missing_prompts = [p for p in required_prompts if not (prompts_dir / p).exists()]
    if missing_prompts:
        print(f"  {FAIL} Missing prompts: {missing_prompts}")
        all_passed = False
    else:
        print(f"  {PASS} All {len(required_prompts)} prompt files present")

    # ---- STEP 4: GraphRAG dry-run validation ---------------------------------
    print("\n[Step 4/11] GraphRAG dry-run validation")
    try:
        dry_ok = run_graphrag_index(
            SMOKE_PROJECT_DIR, dry_run=True, verbose=True,
            method="standard", timeout_seconds=900,
        )
        if dry_ok:
            print(f"  {PASS} Dry-run passed.")
        else:
            print(f"  {FAIL} Dry-run returned failure status.")
            all_passed = False
    except Exception as exc:
        print(f"  {FAIL} Dry-run error: {exc}")
        all_passed = False

    # ---- STEP 5: Full GraphRAG indexing --------------------------------------
    print("\n[Step 5/11] Full GraphRAG indexing (entity + relationship extraction)")
    try:
        ok = run_graphrag_index(
            SMOKE_PROJECT_DIR, dry_run=False, verbose=True,
            method="standard", timeout_seconds=3600,
        )
        if ok:
            print(f"  {PASS} Indexing completed.")
        else:
            print(f"  {FAIL} Indexing returned failure status.")
            print(f"  Check: {SMOKE_PROJECT_DIR / 'logs'}")
            all_passed = False
    except Exception as exc:
        print(f"  {FAIL} Indexing raised: {exc}")
        all_passed = False

    if not all_passed:
        stop_server(server_proc)
        return False

    # ---- STEP 6: Verify output parquet tables --------------------------------
    print("\n[Step 6/11] Verify output parquet tables")
    output_dir = SMOKE_PROJECT_DIR / "output"
    required_tables = ["text_units.parquet", "entities.parquet", "relationships.parquet",
                        "communities.parquet", "community_reports.parquet"]
    for table_file in required_tables:
        path = output_dir / table_file
        if path.exists() and path.stat().st_size > 0:
            import pandas as pd
            df = pd.read_parquet(path)
            print(f"  {PASS} {table_file} - {len(df)} rows")
        else:
            print(f"  {FAIL} {table_file} - missing or empty")
            all_passed = False

    if not all_passed:
        stop_server(server_proc)
        return False

    # ---- STEP 7: Load parquet tables + build local search engine  -----------
    print("\n[Step 7/11] Load output tables + build local search engine")
    search_engine = None
    try:
        search_engine = build_local_search_engine(SMOKE_PROJECT_DIR)
        print(f"  {PASS} Local search engine built successfully")
        print(
            "  Pure GraphRAG local params: "
            f"community_prop={cfg.GRAPHRAG_LOCAL_COMMUNITY_PROP}, "
            f"text_unit_prop={cfg.GRAPHRAG_LOCAL_TEXT_UNIT_PROP}, "
            f"top_k_entities={cfg.GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES}, "
            f"top_k_relationships={cfg.GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS}"
        )
    except Exception as exc:
        print(f"  {FAIL} Failed to build search engine: {exc}")
        all_passed = False
        stop_server(server_proc)
        return False

    # ---- STEP 8: Local search query per question -----------------------------
    print("\n[Step 8/11] Local search queries - one per record")
    raw_search_results: dict[str, Any] = {}
    for rec in SMOKE_RECORDS:
        qid = rec["question_id"]
        q = rec["question"]
        try:
            result = asyncio.run(search_engine.search(q))
            raw_search_results[qid] = result
            resp_text = str(result.response)[:80]
            print(f"  {PASS} [{qid}] response: '{resp_text}...'")
        except Exception as exc:
            print(f"  {FAIL} [{qid}] Search error: {exc}")
            all_passed = False

    # ---- STEP 9: Response normalization + result dict shape ------------------
    print("\n[Step 9/11] Response normalization + result dict shape validation")
    result_rows: list[dict] = []
    required_keys = {"question_id", "question", "gold_answer", "predicted_answer",
                     "system", "table_id", "workspace", "method", "retrieved_contexts",
                     "latency_seconds", "prompt_tokens", "output_tokens",
                     "total_tokens", "error"}
    for rec in SMOKE_RECORDS:
        qid = rec["question_id"]
        raw = raw_search_results.get(qid)
        if raw is None:
            continue
        answer = normalize_graphrag_response(raw.response)
        contexts = serialize_graphrag_contexts(raw)
        row = {
            "question_id": qid,
            "question": rec["question"],
            "gold_answer": rec["answer"],
            "predicted_answer": answer,
            "system": "graphrag",
            "table_id": rec["table_id"],
            "workspace": str(SMOKE_PROJECT_DIR),
            "method": "local",
            "retrieved_contexts": contexts,
            "latency_seconds": raw.completion_time,
            "prompt_tokens": raw.prompt_tokens,
            "output_tokens": raw.output_tokens,
            "total_tokens": (raw.prompt_tokens or 0) + (raw.output_tokens or 0),
            "error": None,
        }
        missing = required_keys - set(row.keys())
        if missing:
            print(f"  {FAIL} [{qid}] Missing keys: {missing}")
            all_passed = False
        else:
            gold = rec["answer"].lower()
            match = gold in answer.lower() or answer.lower() in gold
            status = PASS if match else WARN
            print(f"  {status} [{qid}]  pred='{answer[:60]}'  gold='{rec['answer']}'")
            result_rows.append(row)

    # ---- STEP 10: Answer quality summary -------------------------------------
    print("\n[Step 10/11] Answer quality summary")
    correct = sum(
        1 for row in result_rows
        if row["gold_answer"].lower() in row["predicted_answer"].lower()
        or row["predicted_answer"].lower() in row["gold_answer"].lower()
    )
    total = len(result_rows)
    print(f"  Correct (substring match): {correct}/{total}")
    if correct == 0 and total > 0:
        print(f"  {WARN} No exact substring matches - model may need more context. Check WARN lines above.")
    elif correct >= total // 2:
        print(f"  {PASS} More than half answered correctly - pipeline is functioning.")
    else:
        print(f"  {WARN} Low accuracy on smoke set - review model output above.")

    # ---- STEP 11: Evaluator metrics ------------------------------------------
    print("\n[Step 11/11] Evaluator - ragas metrics on 5 examples")
    try:
        examples = [
            EvaluationExample(
                question_id=rec["question_id"],
                question=rec["question"],
                gold_answer=rec["answer"],
                gold_evidence=None,
                question_type="hybridqa",
                operation_type="table_plus_text",
                difficulty=None,
                answer_type=None,
                metadata={"source_dataset": "hybridqa_smoke", "split": "smoke",
                           "table_id": rec["table_id"]},
            )
            for rec in SMOKE_RECORDS
        ]
        predictions = [
            SystemPrediction(
                question_id=row["question_id"],
                system_name="graphrag",
                predicted_answer=row["predicted_answer"],
                retrieved_contexts=[RetrievedContext.from_dict(ctx) for ctx in row["retrieved_contexts"]],
                latency_seconds=row.get("latency_seconds"),
                prompt_tokens=row.get("prompt_tokens"),
                output_tokens=row.get("output_tokens"),
                total_tokens=row.get("total_tokens"),
                error=row.get("error"),
                metadata={"workspace": row.get("workspace"), "method": row.get("method")},
            )
            for row in result_rows
        ]
        eval_out_dir = cfg.RESULTS_DIR / "experiments" / "smoke_graphrag"
        evaluator = Evaluator(
            dataset_name="hybridqa_smoke",
            system_name="graphrag",
            model_backend=cfg.MODEL_BACKEND,
            generation_model=cfg.LOCAL_GENERATION_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL,
            embedding_model=cfg.LOCAL_EMBEDDING_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.EMBEDDING_MODEL,
            experiment_id="smoke_graphrag",
            dataset_version="smoke",
            query_mode="local",
            command=" ".join(sys.argv),
        )
        evaluator.evaluate(examples=examples, predictions=predictions, output_dir=eval_out_dir)
        print(f"  {PASS} Evaluator ran successfully -> {eval_out_dir}")
    except Exception as exc:
        print(f"  {FAIL} Evaluator failed: {exc}")
        all_passed = False

    stop_server(server_proc)

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPHRAG SMOKE TEST (mirrors full pipeline)")
    print(f"Generation model : {cfg.LOCAL_GENERATION_MODEL}")
    print(f"Embedding model  : {cfg.LOCAL_EMBEDDING_MODEL}")
    print(f"Backend          : {cfg.MODEL_BACKEND}")
    print(f"Records          : {len(SMOKE_RECORDS)} synthetic HybridQA records")
    print(f"Query method     : local")
    print("=" * 60)

    ok = run_smoke_tests()

    print("\n" + "=" * 60)
    if ok:
        print("\033[92mALL 11 STEPS PASSED - GraphRAG pipeline is ready for the full run.\033[0m")
        sys.exit(0)
    else:
        print("\033[91mSOME STEPS FAILED - fix errors above before running the full pipeline.\033[0m")
        sys.exit(1)
