# Standard RAG vs GraphRAG Benchmark Framework

This project now has a dataset-agnostic benchmark layer under `src/benchmark/`.
The goal is to keep custom code focused on experiment orchestration and dataset
adapters, while relying on standard libraries for ingestion, chunking, retrieval,
reranking, GraphRAG indexing, and evaluation wherever possible.

## Architecture

```
dataset adapter
  -> BenchmarkDocument + BenchmarkQuestion
  -> shared chunking/text-unit construction
  -> standard_hybrid_rag
  -> standard_graphrag
  -> common SystemPrediction schema
  -> existing Evaluator reports
```

## Dataset Adapters

Adapters convert a dataset into canonical objects:

- `HybridQAAdapter`: reads official HybridQA raw question files plus
  `WikiTables-WithLinks.zip` by default, preserving table rows, links, passages,
  source paths, and stable IDs. Parsed JSONL is only a fallback.
- `RecordsQAAdapter`: JSONL/JSON/CSV/TSV/Parquet rows containing question,
  answer, and text fields.
- `DirectoryCorpusQAAdapter`: unstructured files plus a separate QA file. It
  uses `LlamaIndex` `SimpleDirectoryReader` when available.
- `TableCorpusQAAdapter`: structured CSV/TSV/JSON/Parquet tables where each row
  becomes an indexable document, with questions loaded from a separate QA file.

New datasets should usually only need a small adapter that returns:

- `BenchmarkDocument(id, text, source_type, metadata)`
- `BenchmarkQuestion(question_id, question, gold_answers, gold_evidence_ids, metadata)`

## Baseline RAG

`StandardRagPipeline` is a stronger industry-style baseline:

- Dense retrieval uses FAISS.
- Lexical retrieval uses `rank-bm25`.
- Dense and lexical results are fused with reciprocal rank fusion.
- Cross-encoder reranking uses `sentence-transformers`.

These are required libraries for the standard pipeline. Missing retrieval
libraries, or disabling dense/BM25/reranking, raises an error instead of
falling back to simplified handmade code.

## GraphRAG

`StandardGraphRagPipeline` writes canonical documents into a GraphRAG workspace
and runs Microsoft GraphRAG indexing. Querying uses GraphRAG's Python search
engine APIs (`get_local_search_engine` / `get_basic_search_engine`) so the
pipeline captures GraphRAG's returned `context_text` and passes it into the
common evaluator as retrieved context.

## Run Example

```bash
python scripts/run_benchmark.py --config configs/benchmark.example.json
```

The example runs baseline RAG and GraphRAG on HybridQA dev questions and writes:

- `benchmark_config.json`
- `canonical_documents.jsonl`
- `index_units.jsonl`
- `questions.jsonl`
- `predictions.jsonl`
- per-system evaluation folders
- `benchmark_summary.json`
- `comparison_summary.json`
- `comparison_report.md`

To run multiple experiments, use `scripts/run_experiment_matrix.py` with
`configs/benchmark_matrix.example.json`.

To parse and inspect a dataset without running RAG/GraphRAG:

```bash
python scripts/prepare_dataset.py --config configs/benchmark.example.json
```

This writes `canonical_documents.jsonl`, `index_units.jsonl`, `questions.jsonl`,
and `evaluation_availability.json`.

## Fairness Modes

Use `chunking.strategy = "none"` when the adapter already emits meaningful text
units and you want both systems to see the same units.

Use `chunking.strategy = "sentence"` or `"token"` for an industry-default mode
where LlamaIndex splits long documents before indexing.

## Evaluation Caveat

Answer-quality metrics and context-text support metrics are comparable across
baseline RAG and GraphRAG. Exact ID-based retrieval metrics are strongest when
both systems return the same canonical document IDs. Baseline RAG does this
directly; GraphRAG returns real query context text, but GraphRAG context sections
may not always expose the original canonical document ID. In those cases, use
answer support, gold-in-context, faithfulness/Ragas, and answer quality as the
primary cross-system comparison metrics.

For HybridQA specifically, the available raw `dev.json`/`train.json` files
provide `question_id`, `question`, `table_id`, `answer-text`, and POS tags. The
linked table/passage zip provides the evidence corpus. These files provide gold
answers, but not official per-question supporting row/passage IDs. The adapter
therefore infers proxy evidence IDs by finding canonical documents containing
the normalized gold answer.

## Config Interface

Core config sections:

- `models`: backend, base URL, API key, generation model, embedding model, and
  optional GraphRAG indexing model.
- `dataset`: adapter type and adapter-specific fields.
- `systems`: `["baseline", "graphrag"]` for the full comparison.
- `chunking`: shared text-unit construction.
- `baseline`: dense/BM25/fusion/reranker settings.
- `graphrag`: query method, response type, indexing method, and workspace flags.
- `parallelism`: optional client-side LLM request concurrency. Defaults should
  remain `1` for a single-GPU local server; use `8` when the configured LLM
  server can distribute simultaneous requests across 8 GPUs. Use
  `llm_concurrent_requests` as a shorthand when embedding, retrieval,
  evaluation, and GraphRAG indexing should all use the same number.

For multi-GPU local-OpenAI deployments, prefer one `models.base_url` that points
to the server/router/load balancer responsible for distributing requests across
GPU-backed model instances. If direct baseline generation and embedding calls
need to talk to separate endpoints, use `models.base_urls`; those calls will
round-robin across the list. GraphRAG should still receive a single routed
`base_url`.
