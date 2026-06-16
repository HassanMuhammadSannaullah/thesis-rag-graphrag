# RAG vs GraphRAG Benchmark Implementation Plan

This repository is now centered on a single configurable benchmark pipeline for
comparing industry-style baseline RAG with Microsoft GraphRAG.

## Current Architecture

```text
config JSON
  -> dataset adapter
  -> BenchmarkDocument / BenchmarkQuestion
  -> shared chunking
  -> standard_hybrid_rag
  -> standard_graphrag
  -> common evaluation schema
  -> reports and comparison summary
```

## Main Files

- `scripts/run_benchmark.py`: run one benchmark config.
- `scripts/run_experiment_matrix.py`: run multiple benchmark configs.
- `configs/benchmark.example.json`: single-experiment example.
- `configs/benchmark_matrix.example.json`: matrix example.
- `src/benchmark/`: canonical adapters, chunking, RAG, GraphRAG, and runner.
- `src/evaluation/`: shared evaluator and report generation.
- `src/graphrag_system/runner.py`: Microsoft GraphRAG CLI integration.
- `src/utils/model_client.py`: model generation and embedding provider facade.

## Required Behavior

Every experiment should define:

- dataset adapter and dataset parameters
- generation model and embedding model
- baseline RAG settings
- GraphRAG settings
- output directory

Each run should write:

- canonical input artifacts
- predictions for both systems
- per-system evaluation reports
- `comparison_summary.json`
- `comparison_report.md`

## Dataset Extension Rule

New datasets should be added by creating a small adapter that emits:

- `BenchmarkDocument(id, text, source_type, metadata)`
- `BenchmarkQuestion(question_id, question, gold_answers, gold_evidence_ids, metadata)`

After that, the same baseline RAG, GraphRAG, and evaluation code should run
without dataset-specific changes.

## Remaining Improvement

The current standard GraphRAG wrapper uses the GraphRAG CLI for compatibility.
For strict retrieval-context metrics, the next improvement is to expose real
GraphRAG query contexts from the GraphRAG Python APIs and map them into
`RetrievedContext`.
