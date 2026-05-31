# Experiment Logs

## 2026-05-24 - HybridQA matrix rerun after Qwen failures

### Observed Qwen 3B matrix failure

- Command:
  - `python scripts/14_run_experiment_matrix.py --run-enabled --auto-compare`
- Active matrix row:
  - `hybridqa_50_questions_qwen3b_e5base`
- Environment:
  - `MODEL_BACKEND=local_openai`
  - `LOCAL_GENERATION_MODEL=Qwen/Qwen2.5-3B-Instruct`
  - `LOCAL_GRAPHRAG_INDEX_MODEL=Qwen/Qwen2.5-3B-Instruct`
  - `LOCAL_EMBEDDING_MODEL=intfloat/e5-base-v2`
- Failure point:
  - GraphRAG indexing failed during `extract_graph`
  - Workspace log: `graphrag_workspace/hybridqa_dev_fair_compare_full/logs/indexing-engine.log`
  - Stack terminates with `KeyError: 'source'`
- Interpretation:
  - This is not just low answer quality on HybridQA.
  - The local GraphRAG extraction path receives malformed relationship output, so the pipeline fails before full QA evaluation completes.
  - Local server traces show malformed extraction patterns from Qwen-family runs, including missing relationship fields and stray HTML-style fragments such as `<br>` and extra delimiter noise.

### Qwen 14B note

- Qwen 14B is not the current fallback on this machine.
- Prior user observation for this machine: `Qwen/Qwen2.5-14B-Instruct` runs out of memory, so it is not the preferred recovery path for this experiment matrix.

### Next action prepared in repo

- Disabled the Qwen 3B row in `configs/experiment_matrix.json`.
- Added a non-Qwen replacement row:
  - `hybridqa_50_questions_mistral7b_e5base`
  - Generation/index model: `mistralai/Mistral-7B-Instruct-v0.3`
  - Embedding model: `intfloat/e5-base-v2`

### Pending run

- The Mistral 7B experiment will be launched next and this file will be updated with the terminal result after the run exits.

### Live Mistral 7B observations

- Status:
  - `mistralai/Mistral-7B-Instruct-v0.3` successfully starts, probes, and completes GraphRAG indexing on the 50-question HybridQA run.
  - Shared-unit baseline RAG also completes all 50 questions.
- New failure point:
  - The run becomes unreliable during GraphRAG question answering, not during indexing.
  - Local server error log shows repeated `500 Internal Server Error` responses caused by `torch.OutOfMemoryError`.
- Concrete server-side error:
  - `Tried to allocate 17.35 GiB. GPU 0 has a total capacity of 15.99 GiB`
- Interpretation:
  - This machine is effectively operating with a 16 GB GPU budget for this run, so Mistral 7B plus long GraphRAG local-search prompts do not fit reliably.
  - The process may continue with retries and per-question errors, but the current run is not a clean or trustworthy experiment result.
