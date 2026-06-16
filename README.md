# RAG vs GraphRAG Benchmark Framework

## How To Run The Full Experiment

### Step 1: Start The LLM Server

Open one terminal and start the local OpenAI-compatible server:

```bash
python scripts/local_openai_server.py
```

Leave this terminal running. This server file is:

```text
scripts/local_openai_server.py
```

If using another person's LLM server instead, do not run this local server.
Update `base_url` in `configs/hybridqa_full_3_models.json` to point to their
server.

### Step 2: Run The Experiment Matrix

Open a second terminal and run:

```bash
python scripts/run_experiment_matrix.py --matrix configs/hybridqa_full_3_models.json
```

The matrix runner file is:

```text
scripts/run_experiment_matrix.py
```

The experiment config is:

```text
configs/hybridqa_full_3_models.json
```

### What This Runs

- full HybridQA dev/validation split, not the 3-question smoke test
- standard hybrid RAG baseline
- Microsoft GraphRAG
- three local open-source 7B-class models:
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `Qwen/Qwen2.5-7B-Instruct`
  - `HuggingFaceH4/zephyr-7b-beta`

On first run, the matrix runner downloads missing HybridQA raw files into
`data/raw/hybridqa/` and downloads/caches missing Hugging Face models into
`local_models/`.

The final model/system comparison report is written to:

```text
results/experiments/hybridqa_full_3_models/model_comparison_report.md
```

### Configuration

All main experiment settings are in:

```text
configs/hybridqa_full_3_models.json
```

Set the LLM server URL here:

```json
"models": {
  "base_url": "http://YOUR_LLM_SERVER:PORT/v1",
  "api_key": "local-dev-key"
}
```

For your local single-GPU machine, keep parallelism at `1`:

```json
"parallelism": {
  "embedding_concurrent_requests": 1,
  "retrieval_concurrent_requests": 1,
  "evaluation_concurrent_requests": 1,
  "graphrag_concurrent_requests": 1
}
```

For an 8-GPU server where the LLM server distributes requests across model
instances, change the parallelism block to:

```json
"parallelism": {
  "llm_concurrent_requests": 8
}
```

To download/cache first without starting the long benchmark:

```bash
python scripts/run_experiment_matrix.py --matrix configs/hybridqa_full_3_models.json --download-only
```

You can list the three configured runs with:

```bash
python scripts/run_experiment_matrix.py --matrix configs/hybridqa_full_3_models.json --list
```

This repository is now organized around one configurable benchmark pipeline:

- normalize a dataset into canonical documents and questions
- run a standard hybrid baseline RAG pipeline
- run Microsoft GraphRAG on the same benchmark input
- evaluate both systems with the same reporting stack
- write per-system and comparison reports under `results/experiments/`

The benchmark is driven by JSON config files, so future experiments can change
dataset, model backend, generation model, embedding model, retrieval settings,
GraphRAG settings, and output location without changing runner code.

## Parallel LLM Requests

The `parallelism` config section controls client-side request concurrency. It
does not choose GPUs directly; it decides how many requests this code sends to
the configured LLM server at the same time. Keep these values at `1` on a
single-GPU local server. On an external server that distributes requests across
8 GPUs, set them to `8`.

```json
"parallelism": {
  "embedding_concurrent_requests": 1,
  "retrieval_concurrent_requests": 1,
  "evaluation_concurrent_requests": 1,
  "graphrag_concurrent_requests": 1
}
```

`embedding_concurrent_requests` fans out baseline document embedding batches.
`retrieval_concurrent_requests` runs benchmark questions concurrently for
baseline and GraphRAG. `evaluation_concurrent_requests` is used by RAGAS
evaluation. `graphrag_concurrent_requests` is written into GraphRAG indexing
settings for GraphRAG's own LLM calls. You can also set
`llm_concurrent_requests` as a shorthand when all four should use the same
number.

For the usual 8-GPU server setup, keep one OpenAI-compatible URL in
`models.base_url` and let that server distribute the simultaneous requests:

```json
"models": {
  "backend": "local_openai",
  "base_url": "http://their-llm-server:8001/v1"
},
"parallelism": {
  "llm_concurrent_requests": 8
}
```

If the model instances are exposed as separate OpenAI-compatible endpoints
instead of one load-balanced URL, direct baseline generation and embedding calls
can also use `models.base_urls`:

```json
"models": {
  "backend": "local_openai",
  "base_urls": [
    "http://gpu-worker-1:8001/v1",
    "http://gpu-worker-2:8001/v1"
  ]
}
```

GraphRAG indexing/search should use a single `base_url`, so put a load balancer
or router URL there when using GraphRAG with multiple server-side instances.

## Main Entry Points

## GPU Setup

For NVIDIA GPU runs, install CUDA-enabled PyTorch before the main requirements:

```bash
pip install -r requirements-gpu-cu128.txt
pip install -r requirements.txt
```

Then verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Do not rely on plain `pip install torch` for GPU; it may install a CPU build.

Check the active benchmark environment:

```bash
python scripts/check_environment.py
```

Run one configured benchmark:

```bash
python scripts/run_benchmark.py --config configs/benchmark.example.json
```

Run a matrix of benchmark configs:

```bash
python scripts/run_experiment_matrix.py --matrix configs/benchmark_matrix.example.json
```

Run the full HybridQA dev/validation model comparison:

```bash
python scripts/run_experiment_matrix.py --matrix configs/hybridqa_full_3_models.json
```

On first run this matrix downloads the HybridQA `dev.json` split and
`WikiTables-WithLinks.zip` into `data/raw/hybridqa/` if they are missing, then
downloads/caches the configured Hugging Face models into `local_models/`. It
runs both `standard_hybrid_rag` and `standard_graphrag` for each configured
model and writes the cross-model comparison to:

```text
results/experiments/hybridqa_full_3_models/model_comparison_report.md
```

The full matrix currently compares:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`
- `HuggingFaceH4/zephyr-7b-beta`

The config defaults to one request at a time for a single-GPU machine. On a
server whose LLM endpoint distributes requests across 8 GPUs/model instances,
set the matrix `parallelism` block to:

```json
"parallelism": {
  "llm_concurrent_requests": 8
}
```

To only download/cache the dataset and models before a long run:

```bash
python scripts/run_experiment_matrix.py --matrix configs/hybridqa_full_3_models.json --download-only
```

Prepare/inspect canonical dataset files without running any model:

```bash
python scripts/prepare_dataset.py --config configs/benchmark.example.json
```

The local OpenAI-compatible model server helper is still available:

```bash
python scripts/local_openai_server.py
```

## Pipeline

```text
dataset adapter
  -> BenchmarkDocument + BenchmarkQuestion
  -> shared chunking/text-unit construction
  -> standard_hybrid_rag
  -> standard_graphrag
  -> common SystemPrediction schema
  -> Evaluator reports
  -> comparison_report.md + comparison_summary.json
```

## Supported Dataset Shapes

The adapter layer currently supports:

- HybridQA raw files (`dev.json`/`train.json` plus `WikiTables-WithLinks.zip`)
  with parsed JSONL fallback
- generic JSON/JSONL/CSV/TSV/Parquet records containing question, answer, and text fields
- structured table corpus plus a separate QA file
- unstructured text/Markdown/HTML directory corpus plus a separate QA file

New datasets should be added by writing a small adapter that returns canonical
`BenchmarkDocument` and `BenchmarkQuestion` objects. Everything after that point
is shared by baseline RAG and GraphRAG.

## Standard Baseline RAG

The baseline is intentionally library-backed:

- dense retrieval uses FAISS
- lexical retrieval uses `rank-bm25`
- dense and lexical rankings are combined with reciprocal rank fusion
- reranking uses `sentence-transformers` CrossEncoder
- answer generation uses the configured model provider through `src/utils/model_client.py`

The pipeline fails fast if required retrieval libraries are missing. It does not
silently replace FAISS or BM25 with handmade fallback retrieval.

## GraphRAG

The GraphRAG path writes canonical benchmark documents into a Microsoft GraphRAG
workspace, creates GraphRAG config, runs indexing, and queries GraphRAG through
GraphRAG's Python search engine APIs so returned context text is available to
the evaluator.

## Outputs

Each benchmark run writes:

- `benchmark_config.json`
- `canonical_documents.jsonl`
- `index_units.jsonl`
- `questions.jsonl`
- `predictions.jsonl`
- one evaluation folder per system
- `benchmark_summary.json`
- `comparison_summary.json`
- `comparison_report.md`

## Documentation

See [docs/standard_benchmark_framework.md](docs/standard_benchmark_framework.md)
for adapter examples and config guidance.
