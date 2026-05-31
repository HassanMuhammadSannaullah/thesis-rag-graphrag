# Thesis RAG vs GraphRAG Experiment Suite

This repository runs a fair comparison between:

- `baseline` RAG: vector retrieval over the same text units used by GraphRAG
- `graphrag`: Microsoft GraphRAG indexing plus GraphRAG query-time retrieval

The experiment matrix lives in `configs/experiment_matrix.json`. That file is the single place where experiments are defined, enabled, and configured.

## What The Shareable Runner Does

The one-command runner:

1. Downloads HybridQA automatically if it is missing
2. Parses the dataset split(s) needed by the selected experiments
3. Runs all enabled experiments from the matrix, or specific `--run-id` entries
4. Builds aggregate metrics and markdown/CSV comparison reports
5. Writes one full text log under `results/logs/` that can be sent back for debugging

Main entrypoint:

```bash
python scripts/15_run_full_experiment_suite.py
```

By default it runs all experiments marked `"enabled": true` in `configs/experiment_matrix.json`.

## Quick Start

Tested with Python `3.12`.

```bash
git clone <your-repo-url>
cd thesis-rag-graphrag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then edit the values you need.

For the current local-model experiment matrix, the important defaults are:

- `MODEL_BACKEND=local_openai`
- `EVALUATION_FRAMEWORK=classic`
- `LOCAL_GENERATION_MODEL=mistralai/Mistral-7B-Instruct-v0.3`
- `LOCAL_GRAPHRAG_INDEX_MODEL=mistralai/Mistral-7B-Instruct-v0.3`
- `LOCAL_EMBEDDING_MODEL=intfloat/e5-base-v2`

Then run:

```bash
python scripts/15_run_full_experiment_suite.py
```

The run will print the path of a single text log file like:

```bash
results/logs/full_experiment_suite_YYYYMMDD_HHMMSS.log
```

## Running Specific Experiments

List the matrix:

```bash
python scripts/14_run_experiment_matrix.py --list
```

Run one experiment from the matrix:

```bash
python scripts/15_run_full_experiment_suite.py --run-id hybridqa_20_questions_mistral7b_e5base_tuned
```

Run all enabled experiments without the final comparison report:

```bash
python scripts/15_run_full_experiment_suite.py --skip-compare
```

Preview commands only:

```bash
python scripts/15_run_full_experiment_suite.py --dry-run
```

## Metrics Reported

The classic evaluation pipeline now reports the same comparison metrics for both `baseline` and `graphrag`.

Answer quality:

- `strict_exact_match`
- `normalized_exact_match`
- `token_f1`
- `answer_contains_gold`
- `numeric_value_match`
- `numeric_value_match_tolerant`
- `unit_match`
- `scale_match`
- `answer_type_match`

Grounding and hallucination:

- `prediction_in_context`
- `gold_in_context`
- `unsupported_prediction`
- `likely_hallucination`
- `hallucination_score`
- `abstention`

Retrieval:

- `answer_support_coverage`
- `answer_support_mrr`
- `answer_support_hit_at_1`
- `answer_support_hit_at_3`
- `answer_support_hit_at_k`
- `evidence_coverage`
- `mrr`
- `evidence_precision_at_k`
- `evidence_recall_at_k`
- `evidence_f1_at_k`
- `ndcg_at_k`

Efficiency:

- `latency_seconds`
- `prompt_tokens`
- `output_tokens`
- `total_tokens`
- `retrieved_context_count`

Note: the `evidence_*` metrics require gold evidence annotations. If a dataset split does not provide them, those fields are written as `null` instead of being silently mixed with other metrics.

## Output Files

Per experiment, the evaluator writes:

- `results/experiments/<experiment_id>/config.json`
- `results/experiments/<experiment_id>/predictions.jsonl`
- `results/experiments/<experiment_id>/per_question_metrics.jsonl`
- `results/experiments/<experiment_id>/aggregate_metrics.json`
- `results/experiments/<experiment_id>/aggregate_metrics.csv`
- `results/experiments/<experiment_id>/statistical_report.json`
- `results/experiments/<experiment_id>/report.md`

Matrix-level runs also write:

- `results/experiments/.../matrix_run_<timestamp>.json`
- comparison markdown and CSV reports under `results/.../reports/`

## Notes For Another Machine

- HybridQA raw files are downloaded automatically by the runner.
- Parsed dataset files, caches, downloaded models, GraphRAG workspaces, and results are all ignored by `.gitignore`.
- Hugging Face models are downloaded on first use into `local_models/`.
- The currently enabled local-model experiment is heavy; a GPU machine is strongly recommended.
