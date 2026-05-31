"""
Baseline RAG smoke test — mirrors the EXACT steps of the production pipeline
(scripts/11_run_hybridqa_proper_compare.py) on 5 synthetic HybridQA records.

Pipeline steps verified (in order):
  1. Local server start / health check  (ensure_local_server equivalent)
  2. Records → GraphRAG text-file documents  (hybridqa_to_graphrag_docs)
  3. GraphRAG settings.yaml written        (create_graphrag_config)
  4. GraphRAG indexing                     (run_graphrag_index → text_units.parquet)
  5. text_units.parquet loaded + embedded  (load_shared_text_units + LocalVectorIndex.add)
  6. Vector retrieval per question         (index.search)
  7. Answer generation per question        (generate_baseline_answer with UNIFIED_QA_PROMPT)
  8. Result dict shape validated           (same keys as production output rows)
  9. Ragas evaluation metrics              (Evaluator.evaluate on the 5 examples)

Exit 0 = all steps pass. Exit 1 = something is broken.

Usage:
    $env:MODEL_BACKEND="local_openai"
    $env:LOCAL_GENERATION_MODEL="Qwen/Qwen2.5-3B-Instruct"
    $env:LOCAL_EMBEDDING_MODEL="intfloat/e5-base-v2"
    python scripts/test_baseline_smoke.py
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
from src.baseline.vector_index import LocalVectorIndex
from src.evaluation.evaluator import Evaluator
from src.evaluation.schemas import EvaluationExample, RetrievedContext, SystemPrediction
from src.graphrag_system.corpus_prep import hybridqa_to_graphrag_docs
from src.graphrag_system.runner import create_graphrag_config, run_graphrag_index
from src.utils.model_client import generate_text
from src.utils.runtime import resolve_project_python

# ---------------------------------------------------------------------------
# 5 synthetic records in the EXACT format produced by 02_parse_hybridqa.py
# (question_id, question, answer, table_id, table{title,section_title,headers,
#  rows,num_rows,all_links}, linked_passages, split)
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

# Mirrors the UNIFIED_QA_PROMPT from 11_run_hybridqa_proper_compare.py
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

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

SMOKE_PROJECT_DIR = cfg.PROJECT_ROOT / "graphrag_workspace" / "smoke_baseline"
SMOKE_INDEX_DIR = cfg.INDEX_CACHE_DIR / "smoke_baseline_text_unit_rag"


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
        [str(resolve_project_python()), str(Path(__file__).parent / "local_openai_server.py")],
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


def generate_baseline_answer(question: str, contexts: list[dict], max_context_chars: int, max_answer_tokens: int) -> str:
    """Exact same function as in 11_run_hybridqa_proper_compare.py."""
    context_text = "\n\n".join(
        f"[Text Unit {idx}] {ctx['text']}" for idx, ctx in enumerate(contexts, start=1)
    )
    prompt = UNIFIED_QA_PROMPT.format(
        context_data=context_text[:max_context_chars],
        response_type="Single sentence",
    )
    prompt = f"{prompt}\n\nQuestion: {question}\nAnswer:"
    return generate_text(prompt, max_tokens=max_answer_tokens).strip()


def load_shared_text_units(project_dir: Path) -> list[dict]:
    """Exact same function as in 11_run_hybridqa_proper_compare.py."""
    import pandas as pd
    text_units_path = project_dir / "output" / "text_units.parquet"
    frame = pd.read_parquet(text_units_path)
    units = []
    for row in frame.to_dict(orient="records"):
        units.append({
            "id": str(row["id"]),
            "text": str(row["text"]),
            "document_id": str(row.get("document_id")),
            "human_readable_id": row.get("human_readable_id"),
            "n_tokens": row.get("n_tokens"),
            "type": "graphrag_text_unit",
        })
    return units


def build_shared_text_unit_index(project_dir: Path, index_dir: Path) -> LocalVectorIndex:
    """Exact same logic as build_shared_text_unit_index() in 11_run_hybridqa_proper_compare.py."""
    if index_dir.exists():
        shutil.rmtree(index_dir)
    shared_units = load_shared_text_units(project_dir)
    texts = [u["text"] for u in shared_units]
    metadatas = [
        {"id": u["id"], "type": u["type"], "text": u["text"],
         "document_id": u["document_id"], "human_readable_id": u["human_readable_id"],
         "n_tokens": u["n_tokens"]}
        for u in shared_units
    ]
    index = LocalVectorIndex(index_dir)
    index.add(texts, metadatas, batch_size=8)
    return index


def run_smoke_tests() -> bool:
    all_passed = True
    server_proc = None

    # ---- STEP 1: Server -------------------------------------------------------
    print("\n[Step 1/9] Local server health check")
    if check_server():
        print(f"  {PASS} Server already running at {cfg.LOCAL_SERVER_HOST}:{cfg.LOCAL_SERVER_PORT}")
    else:
        try:
            server_proc = start_server()
            print(f"  {PASS} Server started.")
        except RuntimeError as exc:
            print(f"  {FAIL} {exc}")
            return False

    # ---- STEP 2: Prepare GraphRAG project (hybridqa_to_graphrag_docs + config) -
    print(f"\n[Step 2/9] Prepare GraphRAG project - {len(SMOKE_RECORDS)} documents")
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
        print(f"  {PASS} {doc_count} text documents written to {input_dir}")
        print(f"  {PASS} settings.yaml written")
    except Exception as exc:
        print(f"  {FAIL} Project setup failed: {exc}")
        if server_proc:
            server_proc.terminate()
        return False

    # ---- STEP 3: GraphRAG indexing -------------------------------------------
    print("\n[Step 3/9] GraphRAG indexing (entity extraction + knowledge graph)")
    try:
        ok = run_graphrag_index(
            SMOKE_PROJECT_DIR, dry_run=False, verbose=True,
            method="standard", timeout_seconds=3600,
        )
        if ok:
            print(f"  {PASS} GraphRAG indexing completed.")
        else:
            print(f"  {FAIL} GraphRAG indexing returned failure status.")
            print(f"  Check: {SMOKE_PROJECT_DIR / 'logs'}")
            all_passed = False
    except Exception as exc:
        print(f"  {FAIL} Indexing raised: {exc}")
        all_passed = False

    if not all_passed:
        if server_proc:
            server_proc.terminate()
        return False

    # ---- STEP 4: Load text_units.parquet ------------------------------------
    print("\n[Step 4/9] Load text_units.parquet produced by GraphRAG")
    try:
        shared_units = load_shared_text_units(SMOKE_PROJECT_DIR)
        print(f"  {PASS} Loaded {len(shared_units)} text units from text_units.parquet")
        if not shared_units:
            print(f"  {FAIL} No text units found - parquet may be empty")
            all_passed = False
    except Exception as exc:
        print(f"  {FAIL} Failed to load text_units.parquet: {exc}")
        all_passed = False
        if server_proc:
            server_proc.terminate()
        return False

    # ---- STEP 5: Embed text units into vector index -------------------------
    print("\n[Step 5/9] Embed text units -> vector index")
    index = None
    try:
        index = build_shared_text_unit_index(SMOKE_PROJECT_DIR, SMOKE_INDEX_DIR)
        index.save()
        print(f"  {PASS} Vector index built - {len(index.metadata)} vectors")
    except Exception as exc:
        print(f"  {FAIL} Indexing failed: {exc}")
        all_passed = False
        if server_proc:
            server_proc.terminate()
        return False

    # ---- STEP 6: Vector retrieval per question -------------------------------
    print("\n[Step 6/9] Vector retrieval (top-8, same as production)")
    retrieval_map: dict[str, list[dict]] = {}
    for rec in SMOKE_RECORDS:
        qid = rec["question_id"]
        q = rec["question"]
        try:
            retrieved = index.search(q, top_k=min(8, index.size))
            retrieval_map[qid] = retrieved
            if retrieved:
                print(f"  {PASS} [{qid}] Retrieved {len(retrieved)} units  |  q: '{q[:55]}'")
            else:
                print(f"  {FAIL} [{qid}] No results for: '{q[:55]}'")
                all_passed = False
        except Exception as exc:
            print(f"  {FAIL} [{qid}] Retrieval error: {exc}")
            all_passed = False

    # ---- STEP 7: Answer generation per question ------------------------------
    print("\n[Step 7/9] Answer generation (UNIFIED_QA_PROMPT, max 128 tokens)")
    result_rows: list[dict] = []
    for rec in SMOKE_RECORDS:
        qid = rec["question_id"]
        q = rec["question"]
        gold = rec["answer"]
        contexts = retrieval_map.get(qid, [])
        try:
            pred = generate_baseline_answer(q, contexts, max_context_chars=16000, max_answer_tokens=128)
            match = gold.lower() in pred.lower() or pred.lower() in gold.lower()
            status = PASS if match else WARN
            print(f"  {status} [{qid}]  pred='{pred}'  gold='{gold}'")
            result_rows.append({
                "question_id": qid,
                "question": q,
                "gold_answer": gold,
                "predicted_answer": pred,
                "system": "baseline",
                "table_id": rec["table_id"],
                "retrieved_contexts": [
                    {"id": item["id"], "text": item.get("text"), "score": item.get("score"),
                     "rank": rank, "source_type": item.get("type"),
                     "metadata": {"document_id": item.get("document_id"),
                                  "human_readable_id": item.get("human_readable_id"),
                                  "n_tokens": item.get("n_tokens")}}
                    for rank, item in enumerate(contexts, start=1)
                ],
                "latency_seconds": None,
                "prompt_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "error": None,
            })
        except Exception as exc:
            print(f"  {FAIL} [{qid}] Generation error: {exc}")
            all_passed = False

    # ---- STEP 8: Result dict shape validation --------------------------------
    print("\n[Step 8/9] Validate output row structure")
    required_keys = {"question_id", "question", "gold_answer", "predicted_answer",
                     "system", "table_id", "retrieved_contexts", "latency_seconds",
                     "prompt_tokens", "output_tokens", "total_tokens", "error"}
    for row in result_rows:
        missing = required_keys - set(row.keys())
        if missing:
            print(f"  {FAIL} [{row['question_id']}] Missing keys: {missing}")
            all_passed = False
        else:
            if not row["predicted_answer"] or not isinstance(row["retrieved_contexts"], list):
                print(f"  {FAIL} [{row['question_id']}] Empty answer or bad contexts type")
                all_passed = False
    if all_passed:
        print(f"  {PASS} All {len(result_rows)} rows have correct structure")

    # ---- STEP 9: Evaluator metrics -------------------------------------------
    print("\n[Step 9/9] Evaluator - ragas metrics on 5 examples")
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
                system_name="baseline",
                predicted_answer=row["predicted_answer"],
                retrieved_contexts=[RetrievedContext.from_dict(ctx) for ctx in row["retrieved_contexts"]],
                latency_seconds=None,
                prompt_tokens=None,
                output_tokens=None,
                total_tokens=None,
                error=None,
                metadata={},
            )
            for row in result_rows
        ]
        eval_out_dir = cfg.RESULTS_DIR / "experiments" / "smoke_baseline"
        evaluator = Evaluator(
            dataset_name="hybridqa_smoke",
            system_name="baseline",
            model_backend=cfg.MODEL_BACKEND,
            generation_model=cfg.LOCAL_GENERATION_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL,
            embedding_model=cfg.LOCAL_EMBEDDING_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.EMBEDDING_MODEL,
            experiment_id="smoke_baseline",
            dataset_version="smoke",
            query_mode="shared_text_unit_rag",
            command=" ".join(sys.argv),
        )
        evaluator.evaluate(examples=examples, predictions=predictions, output_dir=eval_out_dir)
        print(f"  {PASS} Evaluator ran successfully -> {eval_out_dir}")
    except Exception as exc:
        print(f"  {FAIL} Evaluator failed: {exc}")
        all_passed = False

    if server_proc is not None:
        server_proc.terminate()

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE RAG SMOKE TEST (mirrors full pipeline)")
    print(f"Generation model : {cfg.LOCAL_GENERATION_MODEL}")
    print(f"Embedding model  : {cfg.LOCAL_EMBEDDING_MODEL}")
    print(f"Backend          : {cfg.MODEL_BACKEND}")
    print(f"Records          : {len(SMOKE_RECORDS)} synthetic HybridQA records")
    print("=" * 60)

    ok = run_smoke_tests()

    print("\n" + "=" * 60)
    if ok:
        print("\033[92mALL 9 STEPS PASSED - baseline RAG pipeline is ready for the full run.\033[0m")
        sys.exit(0)
    else:
        print("\033[91mSOME STEPS FAILED - fix the issues above before running the full pipeline.\033[0m")
        sys.exit(1)
