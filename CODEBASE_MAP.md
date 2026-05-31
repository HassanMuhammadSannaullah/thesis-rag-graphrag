# CODEBASE MAP — thesis-rag-graphrag
> **Purpose:** Single-file reference for AI assistants and collaborators. Read this instead of re-scanning the full codebase.
> **Maintained:** Update this file whenever a module, function, schema, path, or key behaviour changes.

---

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [Top-Level Layout](#2-top-level-layout)
3. [Configuration — `src/config/`](#3-configuration--srcconfig)
4. [Utilities — `src/utils/`](#4-utilities--srcutils)
5. [Baseline RAG System — `src/baseline/`](#5-baseline-rag-system--srcbaseline)
6. [GraphRAG System — `src/graphrag_system/`](#6-graphrag-system--srcgraphrag_system)
7. [Evaluation Infrastructure — `src/evaluation/`](#7-evaluation-infrastructure--srcevaluation)
8. [Data Pipeline — `src/data_pipeline/`](#8-data-pipeline--srcdata_pipeline)
9. [Scripts — `scripts/`](#9-scripts--scripts)
10. [Datasets](#10-datasets)
11. [Experiment Configuration](#11-experiment-configuration)
12. [Caching Strategy](#12-caching-strategy)
13. [Key Data Flows (End-to-End)](#13-key-data-flows-end-to-end)
14. [Environment Variables & `.env`](#14-environment-variables--env)
15. [Outputs & Results Structure](#15-outputs--results-structure)
16. [Known Issues & TODOs](#16-known-issues--todos)
17. [Changelog](#17-changelog)

---

## 1. Project Goal

Master thesis comparing two retrieval-augmented generation (RAG) strategies on hybrid QA tasks:

| System | Strategy |
|---|---|
| **Baseline** | Flat vector search (cosine similarity over numpy embeddings) + 2-stage link expansion |
| **GraphRAG** | Microsoft GraphRAG (knowledge graph + community reports + local/basic search) |

Both systems use the **same underlying text units** so differences are purely due to retrieval strategy, not document coverage. Primary dataset: **HybridQA** (Wikipedia tables + linked passages). Secondary dataset: **Compliance** (synthetic, custom thesis dataset).

---

## 2. Top-Level Layout

```
thesis-rag-graphrag/
├── CODEBASE_MAP.md          ← THIS FILE — update on every change
├── README.md
├── requirements.txt         ← Python deps (see §14 for extra runtime deps)
├── .env                     ← NOT committed; holds API keys and overrides
│
├── configs/
│   └── experiment_matrix.json   ← Experiment run definitions (see §11)
│
├── src/                     ← All importable library code
│   ├── config/
│   ├── utils/
│   ├── baseline/
│   ├── graphrag_system/
│   ├── evaluation/
│   └── data_pipeline/
│
├── scripts/                 ← CLI entry points (numbered = pipeline order)
│
├── data/                    ← Raw + processed datasets (see §10)
├── graphrag_workspace/      ← Per-experiment GraphRAG project dirs
├── cache/                   ← Disk caches (embeddings, generations, indexes)
├── local_models/            ← HuggingFace model weights (HF_HOME)
├── results/                 ← Experiment outputs, logs, metrics, reports
└── notebooks/               ← Exploratory notebooks (not in main pipeline)
```

---

## 3. Configuration — `src/config/`

### `src/config/settings.py`
**Central config module.** Loaded by every other module via `from src.config import settings as cfg`.

- Loads `.env` from project root via `python-dotenv`.
- Provides all paths as `pathlib.Path` objects (auto-creates cache/results dirs on import).
- Switches behaviour based on `MODEL_BACKEND` env var.

**Key variables:**

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_BACKEND` | `"gemini"` | `"gemini"` or `"local_openai"` |
| `GOOGLE_API_KEY` | `""` | Gemini API key |
| `LOCAL_LLM_BASE_URL` | `"http://127.0.0.1:8001/v1"` | Local server URL |
| `GENERATION_MODEL` | `"gemini-2.5-flash-lite"` | Gemini generation model |
| `EMBEDDING_MODEL` | `"gemini-embedding-001"` | Gemini embedding model |
| `LOCAL_GENERATION_MODEL` | `"Qwen/Qwen2.5-14B-Instruct"` | Local generation model |
| `LOCAL_GRAPHRAG_INDEX_MODEL` | same as generation | Model used during GraphRAG indexing |
| `LOCAL_EMBEDDING_MODEL` | `"intfloat/e5-small-v2"` | Local embedding model |
| `MAX_RPM` / `MAX_RPD` | `10` / `500` | Gemini rate limits |
| `SLEEP_BETWEEN_REQ` | `4.0` s | Sleep between generation calls |
| `SLEEP_BETWEEN_EMB` | `0.5` s | Sleep between embedding calls |
| `DEV_SAMPLE_SIZE` | `5` | Default sample size for dev |
| `EVAL_SAMPLE_SIZE` | `10` | Default sample for evaluation |
| `TOP_K_RETRIEVAL` | `5` | Default top-k for vector search |
| `FAIR_BASELINE_MAX_CONTEXT_CHARS` | `16000` | Max context chars in fair comparison |
| `FAIR_BASELINE_MAX_ANSWER_TOKENS` | `128` | Max answer tokens in fair comparison |
| `GRAPHRAG_LOCAL_*` | various | GraphRAG local search tuning params, including `GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS` |

**Key path constants** (all relative to `PROJECT_ROOT`):

| Constant | Path |
|---|---|
| `DATA_DIR` | `data/` |
| `RAW_DIR` | `data/raw/hybridqa/` |
| `HYBRIDQA_DIR` | `data/hybridqa/` |
| `ORIGINAL_DIR` | `data/hybridqa/original/` |
| `SAMPLES_DIR` | `data/hybridqa/samples/` |
| `PARSED_DIR` | `data/hybridqa/parsed/` |
| `COMPLIANCE_DIR` | `data/compliance/` |
| `CACHE_DIR` | `cache/` |
| `EMB_CACHE_DIR` | `cache/embeddings/` |
| `INDEX_CACHE_DIR` | `cache/indexes/` |
| `GEN_CACHE_DIR` | `cache/generations/` |
| `RESULTS_DIR` | `results/` |
| `LOGS_DIR` | `results/logs/` |
| `METRICS_DIR` | `results/metrics/` |
| `OUTPUTS_DIR` | `results/outputs/` |
| `LOCAL_MODELS_DIR` | `local_models/` |

---

### `src/config/model_registry.py`
Provides `ModelSpec` dataclass and ranked lists of generation/embedding models with hardware metadata. Used by `scripts/12_prepare_local_models.py` to decide what to download.

**Generation models (tier A):** Qwen2.5-14B-Instruct (fp16), Qwen2.5-7B-Instruct (int4), Mistral-7B-Instruct-v0.3 (fp16), Llama-3.1-8B-Instruct (int4)
**Embedding models (tier A):** e5-base-v2, bge-base-en-v1.5 | **(tier B):** e5-small-v2, bge-small-en-v1.5, gte-base

Function `build_local_model_registry_snapshot()` → returns hardware-aware dict used in experiment metadata.

---

## 4. Utilities — `src/utils/`

### `src/utils/model_client.py`
**The single interface for all LLM/embedding calls.** Both backends (Gemini and local OpenAI) go through here.

**Public functions:**

| Function | Signature | Notes |
|---|---|---|
| `generate_text` | `(prompt, model=None, temperature=0.1, max_tokens=512, use_cache=True) → str` | Disk-cached. Retries on 429/503. |
| `embed_texts` | `(texts, model=None, use_cache=True) → list[list[float]]` | Per-text disk cache. Batch-aware. |
| `probe_local_backend` | `(generation_model=None, embedding_model=None) → dict` | Warms up local server, verifies both model paths work. |

**Caching:** Both functions cache to `cache/generations/` using SHA-256 of `backend::model::...::prompt`. Cache files are JSON with `{"backend", "model", "text"}` or `{"backend", "model", "vec"}`.

**Gemini retry logic:** Up to 5 retries; 60s×attempt on 429, 30s×attempt on 503.

**Backend selection:** Governed by `cfg.MODEL_BACKEND`. `"local_openai"` → OpenAI client pointed at `LOCAL_LLM_BASE_URL`. `"gemini"` → google-genai client.

---

### `src/utils/gemini_api.py`
Thin backward-compat shim. Just re-exports `generate_text` and `embed_texts` from `model_client`.

---

### `src/utils/runtime.py`
Resolves the correct Python executable and GraphRAG CLI path for the project conda environment (`thesis_rag_gpu` by default, overridable via `THESIS_RAG_CONDA_ENV`).

| Function | Returns |
|---|---|
| `resolve_project_python()` | `Path` to the correct `python.exe` |
| `resolve_graphrag_cli()` | `str` path to `graphrag` CLI executable |

---

## 5. Baseline RAG System — `src/baseline/`

### `src/baseline/corpus_builder.py`
Converts parsed HybridQA records into flat retrieval units (dicts).

**Unit types and schemas:**

| Type | ID pattern | Fields |
|---|---|---|
| `table_summary` | `summary_{table_id}` | `id, type, text, table_id, question_id` |
| `table_row` | `row_{table_id}_{i}` | `id, type, text, table_id, question_id, row_index, row_links` |
| `linked_passage` | `passage_{table_id}_{j}` | `id, type, text, table_id, question_id, link` |

**Key functions:**

| Function | Purpose |
|---|---|
| `build_corpus_for_record(record)` | Builds all units for one HybridQA record |
| `build_corpus(records, max_passages=30)` | Full corpus from all records |
| `save_corpus(corpus, path)` | Saves to JSONL |
| `load_corpus(path)` | Loads from JSONL |

Text limits: passages truncated at 1000 chars; row texts have no truncation.

---

### `src/baseline/vector_index.py`
**`LocalVectorIndex`** — flat numpy cosine similarity index with persistent disk storage.

**Storage files (under `index_path/`):**
- `embeddings.npy` — float32 numpy array shape `(N, D)`
- `metadata.jsonl` — one JSON dict per line, corresponding to embedding rows

**Key methods:**
- `add(texts, metadatas, batch_size=5)` — skips already-indexed IDs; embeds in batches of 5; truncates to 2000 chars for embedding
- `search(query, top_k=5)` → list of metadata dicts with added `"score"` field (cosine similarity)
- `save()` / `_load_if_exists()` — automatic load on construction if files exist
- `size` property → number of indexed items

---

### `src/baseline/retriever.py`
Implements two-stage hybrid retrieval.

**`hybrid_retrieve(question, index, passage_lookup, top_k=5, max_expansion=5, max_total=10)`:**
1. **Stage 1:** Vector search → top `top_k` candidates (summaries + rows)
2. **Stage 2:** For retrieved `table_row` candidates, follow `row_links` → look up passages in `passage_lookup` dict
3. Returns combined list capped at `max_total`

**`build_passage_lookup(records)`:** Builds `{wiki_link → passage_unit}` dict from all records. Passages truncated at 1000 chars. Each entry has `score: -0.1` (expansion items rank below vector results).

> **Known bug:** There is unreachable code after the `return` statement on the last two lines of `hybrid_retrieve`. Harmless but should be cleaned up.

---

### `src/baseline/answer_generator.py`
Generates natural-language answers from question + evidence units.

**`generate_answer(question, evidence_units)`** → dict with `{question, generated_answer, num_evidence_units, evidence_types}`.

Uses `ANSWER_PROMPT` — instructs model to:
- Use only evidence facts
- Return shortest exact answer
- Say "INSUFFICIENT EVIDENCE" if context is insufficient
Context is capped at 6000 chars before truncation.

> **Note:** This module is used for standalone baseline runs. The fair-comparison script (`11_run_hybridqa_proper_compare.py`) uses a different `UNIFIED_QA_PROMPT` defined inline.

---

## 6. GraphRAG System — `src/graphrag_system/`

### `src/graphrag_system/corpus_prep.py`
Converts structured datasets into text files for GraphRAG's `input/` directory.

| Function | Input → Output |
|---|---|
| `hybridqa_to_graphrag_docs(records, output_dir)` | N records → N txt files named `hybridqa_doc_{i:04d}.txt` |
| `compliance_to_graphrag_docs(transactions_path, policies_path, output_dir)` | JSON files → per-policy txts + per-department txts + `transactions_all.txt` |
| `hybridqa_record_to_text(record, max_passages=10)` | single record → markdown-ish text string |

GraphRAG doc format: markdown-like with `# Table: ...`, `## Table Data`, row-by-row text, `## Related Entities` sections.

---

### `src/graphrag_system/runner.py`
Wraps the Microsoft GraphRAG CLI.

**`create_graphrag_config(project_dir, api_key, model=None, embedding_model=None, api_base=None, force=False)`:**
- Initialises GraphRAG project (`graphrag init -r ...`)
- Writes `settings.yaml` auto-generated from active backend config
- Always overwrites `settings.yaml` to enforce correct backend settings

**`_settings_yaml(...)`** generates GraphRAG 3.x compatible YAML with:
- `concurrent_requests: 1` (required for local server)
- Separate `graph_index_completion_model` and `default_completion_model`
- LanceDB vector store at `output/lancedb`
- Chunking: 4000 tokens, 0 overlap, cl100k_base encoding
- Claims extraction **disabled** (`extract_claims.enabled: false`)
- `local_search` and `basic_search` use custom `LOOKUP_SEARCH_PROMPT` (injected via `prompts/` dir)

**`run_graphrag_index(project_dir, timeout=7200)`** — runs `graphrag index -r <project_dir>`.

**`has_graphrag_index(project_dir)`** — checks whether `output/lancedb` exists.

**Other helpers:** `index_graphrag(...)`, `query_graphrag_local(...)`, `query_graphrag_basic(...)`.

**`LOOKUP_SEARCH_PROMPT`** — custom system prompt emphasising exact lookups and discouraging markdown/commentary.

---

## 7. Evaluation Infrastructure — `src/evaluation/`

### Core Schemas — `schemas.py`

```python
@dataclass
class RetrievedContext:
    id: str | None
    text: str | None
    score: float | None
    rank: int | None
    source_type: str | None
    metadata: dict

@dataclass
class EvaluationExample:
    question_id: str
    question: str
    gold_answer: str | list[str]
    gold_evidence: list[str] | None   # evidence IDs for retrieval metrics
    question_type: str | None         # e.g. "bridge", "comparison"
    operation_type: str | None
    difficulty: str | None
    answer_type: str | None           # "numeric", "date", "entity"...
    aliases: list[str] | None
    metadata: dict

@dataclass
class SystemPrediction:
    question_id: str
    system_name: str
    predicted_answer: str
    retrieved_contexts: list[RetrievedContext] | None
    latency_seconds: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None
    metadata: dict

@dataclass
class MetricResult:          # (also in schemas.py)
    question_id: str
    system_name: str
    metrics: dict
    warnings: list[str]
    error: str | None
```

All dataclasses have `.from_dict()` and `.to_dict()` / `asdict()`.

---

### Answer Metrics — `answer_metrics.py`

| Function | Returns |
|---|---|
| `strict_exact_match(pred, gold)` | `int` — string equality |
| `normalized_exact_match(pred, gold_answers)` | `int` — after normalization |
| `answer_contains_gold(pred, gold_answers)` | `int` — substring check |
| `token_f1_score(pred, gold_answers)` | `float` — best F1 across gold variants |
| `compute_answer_metrics(pred, gold_answers, answer_type, numeric_tolerance)` | `dict` — all metrics combined |

Also includes: `answer_type_match`, `_best_numeric_alignment` (with tolerant numeric matching).

---

### Normalization — `normalization.py`
Normalises before comparison: lowercase, remove articles (`a/an/the`), strip currency symbols, unify date formats, split into tokens.

| Function | Purpose |
|---|---|
| `normalize_answer_text(text)` | Full normalization → string |
| `normalize_tokens(text)` | Normalization → list of tokens |
| `try_normalize_date(text)` | Parse date → `"YYYY-MM-DD"` or `None` |
| `extract_numbers_with_units(text)` | Returns `[{value, unit, scale, raw}]` |
| `canonical_text_variants(text)` | Returns set of equivalent string forms |

---

### Retrieval Metrics — `retrieval_metrics.py`
Compares retrieved context IDs against `gold_evidence` IDs.

**`compute_retrieval_metrics(gold_evidence, retrieved_contexts, k_values=[1,3,5,10])`** → `(metrics_dict, warnings_list)`

Metrics computed: `evidence_coverage`, `mrr`, `precision@k`, `recall@k`, `f1@k`, `ndcg@k` for each k in `k_values`.

Ranking: sorted by `ctx.rank` first, then by `-ctx.score` (descending).

---

### Hallucination Metrics — `hallucination_metrics.py`

**`compute_hallucination_metrics(predicted_answer, gold_answers, retrieved_contexts, answer_metrics, judge_metrics)`** → dict

Produces:
- `abstained` — predicted answer is in the abstention set ("I do not know" etc.)
- `prediction_in_context` — normalized prediction appears in normalized joined context
- `gold_in_context` — gold answer appears in context
- `unsupported_prediction` — non-abstained prediction not found in context
- `faithfulness_gap` — 1.0 - faithfulness (or heuristic if no judge)
- `likely_hallucination` — heuristic flag: wrong + unsupported OR gold in context but prediction not

---

### Judge Metrics — `judge_metrics.py`
Optional LLM-as-judge evaluation.

**`compute_judge_metrics(question, gold_answer, predicted_answer, retrieved_contexts, model=None)`** → dict

Prompts the LLM with `JUDGE_TEMPLATE` → parses JSON response with keys: `faithfulness`, `answer_relevance`, `context_relevance`, `context_precision_llm`, `context_recall_llm`, `rationale`.

Uses `json_repair` if standard parsing fails. **Disabled by default** (requires `--compute-judge-metrics` flag or `compute_judge_metrics=True` in `Evaluator`).

---

### Aggregation — `aggregate.py`

**`summarize_group(rows)`** — aggregates a list of per-question metric dicts into means, std devs, null counts.

**`aggregate_rows(rows, group_fields)`** — groups rows by field values, applies `summarize_group` per group. Returns `{"overall": [...], "question_type": [...], ...}`.

**`flatten_aggregate_tables(agg_result)`** — flattens into list of dicts for reporting.

---

### Statistics — `statistics.py`

| Function | Purpose |
|---|---|
| `bootstrap_confidence_interval(values, confidence=0.95, n_resamples=1000, seed=42)` | Returns `{mean, ci_low, ci_high}` |
| `summarize_statistical_report(metric_rows, systems, metrics)` | Per-metric CIs and pairwise differences |

`PRIMARY_METRICS` list: `normalized_exact_match`, `token_f1`, `answer_contains_gold`, `context_contains_gold_answer`, `faithfulness`, `likely_hallucination`, `unsupported_prediction`, `evidence_recall_at_5`.

---

### Evaluator — `evaluator.py`
**`Evaluator` class** — orchestrates the full per-experiment evaluation and saves results.

Constructor parameters: `dataset_name`, `system_name`, `model_backend`, `generation_model`, `embedding_model`, `experiment_id`, `dataset_version`, `dataset_path`, `query_mode`, `k_values=[1,3,5,10]`, `numeric_tolerance=1e-3`, `compute_judge_metrics=False`, `judge_model=None`, `command=None`, `run_metadata={}`.

**`evaluate(examples, predictions, output_dir)`** — main method:
1. Matches predictions to examples by `question_id`
2. Computes answer metrics, retrieval metrics, hallucination metrics, (optionally) judge metrics per question
3. Aggregates by group fields (`question_type`, `answer_type`, `difficulty`)
4. Computes stats and CIs
5. Saves output bundle → `output_dir/`

**Output bundle structure:**
```
output_dir/
  config.json         ← evaluator config + git commit
  predictions.jsonl   ← raw SystemPrediction dicts
  metrics.jsonl       ← per-question MetricResult dicts
  summary.json        ← aggregated metrics
  by_type.json        ← per-question-type aggregation
  report.md           ← markdown report
  comparison.csv      ← flat CSV
```

---

### Reporting — `reporting.py`

**`build_experiment_report(...)`** → markdown string with Overview, Overall Results table, Category Breakdown, Retrieval Metrics, Judge Metrics, Statistical Summary sections.

**`build_comparison_report(rows)`** → markdown table comparing multiple experiment bundles.

Report columns (MAIN_COLUMNS): `system_name`, `dataset_name`, `generation_model`, count, `normalized_exact_match`, `token_f1`, `numeric_value_match`, `context_contains_gold_answer`, `retrieved_context_count`, `faithfulness`, `likely_hallucination`, `unsupported_prediction`, `evidence_recall_at_5`, `evidence_precision_at_5`, `latency_seconds`.

---

### I/O Helpers — `experiment_io.py`
Stateless helpers: `write_json`, `read_json`, `write_jsonl`, `read_jsonl`, `write_csv`, `append_jsonl`, `ensure_dir`, `timestamp_utc`.

---

### Legacy Wrapper — `metrics.py`
Backward-compat wrappers: `normalize_answer`, `exact_match`, `contains_gold`, `token_f1`, `evaluate_results`, `evaluate_by_question_type`, `compare_systems`. These delegate to the new metric modules.

---

## 8. Data Pipeline — `src/data_pipeline/`

### `src/data_pipeline/compliance_generator.py`
Generates the synthetic **Compliance** dataset. Run once; outputs committed to `data/compliance/`.

**Generated files:**
- `data/compliance/transactions.json` — 50 transaction records
- `data/compliance/policies.json` — 5 policy documents
- `data/compliance/questions.json` — 10 evaluation questions

**Transaction schema fields:** `transaction_id`, `date`, `department`, `vendor`, `category`, `amount`, `currency`, `status`, `required_approval`, `actual_approver`, `compliant`, `invoice_ref`, `description`.

**Compliance rules (from generated policy POL-001):**
- `< $1,000` → Department Manager
- `$1,000–$5,000` → Department Head
- `$5,000–$25,000` → Finance Director
- `> $25,000` → CFO + Board Approval

**15% violation rate** (actual_approver lower than required). `random.seed(42)` ensures reproducibility.

**Question types:** `simple_lookup`, `hybrid_lookup`, `multi_hop`, `compliance_reasoning`.

---

## 9. Scripts — `scripts/`

Scripts are numbered by pipeline stage. Always run from project root with `python scripts/<name>.py`.

### Download & Parse Pipeline

| Script | Purpose | Key args |
|---|---|---|
| `01_download_hybridqa.py` | Downloads `train.json`, `dev.json`, `WikiTables-WithLinks.zip` from GitHub | none |
| `02_parse_hybridqa.py` | Parses ZIP → JSONL records in `data/hybridqa/original/` and `samples/` | `--split dev/train/all`, `--sample-size N` |

**Parsed record schema (JSONL):**
```json
{
  "question_id": "...",
  "question": "...",
  "answer": "...",
  "table_id": "...",
  "table": {
    "title": "...", "section_title": "...", "intro": "...",
    "headers": [...], "rows": [{...,"_links": [...]}], "num_rows": N
  },
  "linked_passages": [{"link": "...", "text": "..."}]
}
```

---

### Experiment Scripts

| Script | Purpose | Key args |
|---|---|---|
| `11_run_hybridqa_proper_compare.py` | **MAIN**: Fair comparison of baseline vs GraphRAG | `--split`, `--systems`, `--question-limit`, `--force-query`, `--force-reindex`, `--graphrag-query-method local/basic`, `--baseline-top-k`, `--baseline-max-context-chars`, `--baseline-max-answer-tokens`, `--compute-judge-metrics` |
| `14_run_experiment_matrix.py` | Runs experiments from `configs/experiment_matrix.json` | `--list`, `--run-id ID`, `--run-enabled`, `--dry-run`, `--stop-on-failure`, `--auto-compare` |
| `compare_experiments.py` | Loads saved experiment bundles and produces comparison report | `--experiments dir1 dir2 ...`, `--output-md`, `--output-csv` |
| `evaluate_predictions.py` | Standalone evaluator: takes examples + predictions files, saves bundle | `--examples`, `--predictions`, `--output-dir`, `--dataset-name`, `--system-name`, etc. |

---

### Infrastructure Scripts

| Script | Purpose |
|---|---|
| `12_prepare_local_models.py` | Inspect/download local HuggingFace models. Flags: `--download`, `--generation-tiers A,B`, `--embedding-tiers A,B`, `--max-models N` |
| `13_check_system_readiness.py` | Pre-flight checks: disk space, required files, matrix validity, fairness config |
| `local_openai_server.py` | FastAPI OpenAI-compatible server backed by HuggingFace transformers. Serves generation (`/v1/chat/completions`) and embedding (`/v1/embeddings`) on `LOCAL_SERVER_PORT` (default 8001). Uses `Lock` per model type for thread safety. |
| `test_baseline_smoke.py` | Quick smoke test of baseline retrieval pipeline |
| `test_graphrag_smoke.py` | Quick smoke test of GraphRAG pipeline |
| `verify_corpus.py` | Checks corpus row_links and passage linkage for a sample |

---

### `scripts/11_run_hybridqa_proper_compare.py` — Detailed Flow

This is the **authoritative experiment script**. Its flow:

1. **Parse args** (`--split`, `--systems`, `--question-limit`, etc.)
2. **Start local server** if `MODEL_BACKEND == local_openai` and no server running (spawns `local_openai_server.py`)
3. **Load parsed HybridQA records** from `data/hybridqa/original/{split}.jsonl`; apply `--question-limit`
4. **Build GraphRAG input docs** via `hybridqa_to_graphrag_docs` → `graphrag_workspace/{workspace_name}/input/`
5. **Run GraphRAG index** if not already indexed (or `--force-reindex`)
6. **Build baseline vector index** — loads corpus from text units produced for GraphRAG (same docs) → `LocalVectorIndex` at `cache/indexes/{workspace_name}/`
7. **For each question:**
   - **Baseline:** vector search top-k → truncate to `max_context_chars` → `UNIFIED_QA_PROMPT` → `generate_text`
   - **GraphRAG:** `local` or `basic` search via Python API (`get_local_search_engine` / `get_basic_search_engine`) → format response as `SystemPrediction`
8. **Evaluate** via `Evaluator.evaluate(examples, predictions, output_dir)`
9. Save results to `results/experiments/{experiment_id}/`

**GraphRAG query uses Python API** (not CLI subprocess) for local and basic search modes. Loaded via `graphrag.query.factory`.

**Baseline uses `UNIFIED_QA_PROMPT`** (defined in script, not in `answer_generator.py`).

**Workspace naming:** `graphrag_workspace/hybridqa_{split}_proper_full/` (or similar based on experiment config).

---

### `scripts/local_openai_server.py` — Server Details

- **Framework:** FastAPI + uvicorn
- **Endpoints:** `GET /health`, `POST /v1/chat/completions`, `POST /v1/embeddings`
- **Models loaded lazily** on first request, then cached globally
- **Threading:** `_generation_lock` and `_embedding_lock` prevent concurrent model use
- **HF model storage:** `HF_HOME` and `HUGGINGFACE_HUB_CACHE` set to `LOCAL_MODELS_DIR`
- **Device resolution:** CUDA if available, otherwise CPU
- **dtype:** `float16` on CUDA, `float32` on CPU
- **Embedding method:** Mean pooling over last hidden states (no CLS token assumed)

---

## 10. Datasets

### HybridQA
- **Source:** GitHub `wenhuchen/HybridQA` + `wenhuchen/WikiTables-WithLinks`
- **Raw files:** `data/raw/hybridqa/train.json`, `dev.json`, `WikiTables-WithLinks.zip`
- **Parsed files:** `data/hybridqa/original/dev.jsonl`, `train.jsonl`
- **Sample files:** `data/hybridqa/samples/dev_sample.jsonl`, `train_sample.jsonl`
- **Dev set size:** ~3000 questions; typical experiment uses `--question-limit 300`
- **Structure:** Each record links a Wikipedia table with entity passages via cell hyperlinks
- **No category/difficulty labels** in the original dataset

### Compliance (Synthetic)
- **Files:** `data/compliance/transactions.json`, `policies.json`, `questions.json`
- **Generated by:** `src/data_pipeline/compliance_generator.py` with `seed=42`
- **Size:** 50 transactions, 5 policies, 10 questions
- **Violation rate:** 15% of transactions non-compliant
- **Question types:** `simple_lookup`, `hybrid_lookup`, `multi_hop`, `compliance_reasoning`

---

## 11. Experiment Configuration

### `configs/experiment_matrix.json`
Defines experiments as a list. Each experiment has:
```json
{
  "id": "unique_string",
  "enabled": true/false,
  "script": "scripts/11_run_hybridqa_proper_compare.py",
  "args": { "split": "dev", "systems": "baseline,graphrag", "question-limit": 300, ... },
  "env": { "MODEL_BACKEND": "local_openai", "LOCAL_GENERATION_MODEL": "...", ... },
  "notes": "Human description"
}
```

**Current active experiment (version 2.0):**
- ID: `hybridqa_final_qwen14b_e5base_local`
- 300 questions, dev split, both systems, local GraphRAG query mode
- Model: `Qwen/Qwen2.5-14B-Instruct` + `intfloat/e5-base-v2`
- Hardware target: RTX 4090 (24GB VRAM)
- Results go to: `results/experiments/final_hybridqa/`

Run all enabled: `python scripts/14_run_experiment_matrix.py --run-enabled`

---

## 12. Caching Strategy

| Cache type | Location | Format | Key |
|---|---|---|---|
| Generation | `cache/generations/gen_{sha256[:16]}.json` | `{backend, model, text}` | `backend::model::temp::max_tokens::prompt` |
| Embedding | `cache/generations/emb_{sha256[:16]}.json` | `{backend, model, vec}` | `backend::model::emb::text` |
| Vector index | `cache/indexes/{name}/embeddings.npy` + `metadata.jsonl` | numpy + JSONL | IDs in metadata (skip already-indexed) |
| GraphRAG index | `graphrag_workspace/{name}/output/lancedb/` | LanceDB | Existence check via `has_graphrag_index()` |

**Invalidation:** No automatic TTL. Caches persist across runs. Use `--force-query` or `--force-reindex` flags to bypass.

---

## 13. Key Data Flows (End-to-End)

### Baseline RAG Flow
```
raw JSON/ZIP
  → 02_parse_hybridqa.py → data/hybridqa/original/dev.jsonl
  → corpus_builder.build_corpus() → list of retrieval units
  → LocalVectorIndex.add() → cache/indexes/{name}/
  → hybrid_retrieve(question, index, passage_lookup) → evidence units
  → generate_text(UNIFIED_QA_PROMPT) → predicted_answer
  → Evaluator.evaluate() → results/experiments/{id}/
```

### GraphRAG Flow
```
data/hybridqa/original/dev.jsonl
  → corpus_prep.hybridqa_to_graphrag_docs() → graphrag_workspace/{name}/input/*.txt
  → graphrag index -r graphrag_workspace/{name}/ → output/lancedb/ (knowledge graph)
  → get_local_search_engine() / get_basic_search_engine()
  → engine.search(question) → SearchResult
  → Evaluator.evaluate() → results/experiments/{id}/
```

### Evaluation Flow
```
list[EvaluationExample] + list[SystemPrediction]
  → per-question: answer_metrics + retrieval_metrics + hallucination_metrics [+ judge_metrics]
  → aggregate_rows() → summarize_group() per question_type
  → bootstrap_confidence_interval() per metric
  → build_experiment_report() → report.md
  → write_json/write_jsonl/write_csv → output_dir/
```

---

## 14. Environment Variables & `.env`

The `.env` file at project root is loaded by `settings.py`. All variables are optional with defaults.

```dotenv
# Model backend
MODEL_BACKEND=local_openai         # or gemini

# Gemini credentials (when MODEL_BACKEND=gemini)
GOOGLE_API_KEY=AIza...

# Local server (when MODEL_BACKEND=local_openai)
LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1
LOCAL_LLM_API_KEY=local-dev-key
LOCAL_GENERATION_MODEL=Qwen/Qwen2.5-14B-Instruct
LOCAL_EMBEDDING_MODEL=intfloat/e5-base-v2
LOCAL_GRAPHRAG_INDEX_MODEL=Qwen/Qwen2.5-14B-Instruct

# Hardware
LOCAL_MODEL_DEVICE=auto            # cuda / cpu / auto
LOCAL_MODEL_DEVICE_MAP=none        # or "auto" for multi-GPU
LOCAL_MODEL_TRUST_REMOTE_CODE=false
LOCAL_MODEL_VRAM_GB=24
LOCAL_MODEL_RAM_GB=32
LOCAL_MODEL_STORAGE_BUDGET_GB=150
LOCAL_SERVER_HOST=127.0.0.1
LOCAL_SERVER_PORT=8001
LOCAL_MAX_NEW_TOKENS=2048

# Rate limits (Gemini)
MAX_RPM=10
MAX_RPD=500
EMBEDDING_RPM=100
SLEEP_BETWEEN_REQUESTS=4.0
SLEEP_BETWEEN_EMBEDDINGS=0.5

# Sample sizes
DEV_SAMPLE_SIZE=5
EVAL_SAMPLE_SIZE=10
MAX_LINKED_PASSAGES=3
TOP_K_RETRIEVAL=5

# Fair comparison controls
FAIR_BASELINE_MAX_CONTEXT_CHARS=16000
FAIR_BASELINE_MAX_ANSWER_TOKENS=128

# GraphRAG tuning
GRAPHRAG_LOCAL_REPORT_COMMUNITY_LEVEL=2
GRAPHRAG_LOCAL_COMMUNITY_PROP=0.0
GRAPHRAG_LOCAL_TEXT_UNIT_PROP=0.75
GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES=20
GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS=20
GRAPHRAG_LOCAL_INCLUDE_RELATIONSHIP_WEIGHT=true

# Conda env for runtime.py
THESIS_RAG_CONDA_ENV=thesis_rag_gpu
```

**Runtime deps not in requirements.txt** (installed in conda env):
- `torch`, `transformers`, `fastapi`, `uvicorn` — for local server
- `json_repair` — for judge metrics JSON parsing
- `lancedb` — for GraphRAG vector store
- `graphrag` >= 1.0.0 (already in requirements.txt)
- `openai` — Python SDK for local server client

---

## 15. Outputs & Results Structure

```
results/
  experiments/
    {experiment_id}/          ← one dir per experiment run
      config.json             ← evaluator config + git SHA
      predictions.jsonl       ← raw SystemPrediction dicts
      metrics.jsonl           ← per-question MetricResult dicts
      summary.json            ← aggregated metrics (overall + by_type)
      by_type.json            ← per-question-type breakdown
      report.md               ← markdown report
      comparison.csv          ← flat CSV for spreadsheet import
    index.jsonl               ← running log of all experiment summaries
    final_hybridqa/           ← matrix run outputs
      reports/
        matrix_compare_{stamp}.md
        matrix_compare_{stamp}.csv
  logs/
    fair_compare_local_server.{out,err}.log
  metrics/                    ← (additional standalone metric files)
  outputs/                    ← (additional output files)

graphrag_workspace/
  {workspace_name}/
    input/                    ← text files fed to GraphRAG
    output/
      lancedb/                ← GraphRAG knowledge graph + embeddings
    settings.yaml             ← auto-generated (overwritten each run)
    prompts/                  ← custom system prompts injected by runner.py
    logs/                     ← GraphRAG indexing logs
    cache/                    ← GraphRAG internal cache
```

---

## 16. Known Issues & TODOs

### Bugs
- **`src/baseline/retriever.py` line ~75**: Unreachable dead code after `return combined[:max_total]` (two extra lines). Harmless.

### Evaluation Gaps (for thesis robustness)
- No structured-only vs hybrid comparison stratification within HybridQA
- No explicit difficulty labels in HybridQA records (would require annotation)
- No train/dev/test split strategy for compliance dataset
- Minimal multi-hop chains in compliance questions (depth ≤ 2)
- No evidence reference validation (gold_evidence IDs not verified against corpus IDs)
- No hallucination type classification (temporal / numeric / entity confusion)
- Judge metrics disabled by default (requires LLM call per question, expensive)

### Architectural Notes
- `answer_generator.py`'s `ANSWER_PROMPT` is NOT used in the main comparison script. The script uses its own `UNIFIED_QA_PROMPT` inline. These should be reconciled.
- `gemini_api.py` is purely a backward-compat shim — all logic is in `model_client.py`.
- GraphRAG `concurrent_requests: 1` is hardcoded in generated `settings.yaml` — necessary for local single-GPU server but suboptimal for Gemini API.

---

## 17. Changelog

| Date | Who | What changed |
|---|---|---|
| 2026-05-21 | AI (initial) | Created this CODEBASE_MAP.md from full codebase scan |
| 2026-05-24 | AI | Added Mistral-7B-Instruct-v0.3 to the local model registry and experiment matrix as the primary non-Qwen fallback for HybridQA runs |
| 2026-05-24 | AI | Added `GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS` and wired it into GraphRAG local search plus matrix reruns that reuse the existing index |

> **Instructions for updating:** When you add a file, change a schema, rename a function, change a default, or complete a TODO — add a row to this table and update the relevant section above. Keep entries brief (one line per logical change).
