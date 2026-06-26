# RAG vs GraphRAG Benchmark Framework

This project runs a thesis benchmark comparing:

- a standard hybrid RAG baseline
- Microsoft GraphRAG
- three local open-source instruction models
- the HybridQA dev dataset

The main full experiment config is:

```text
configs/hybridqa_full_3_models.json
```

The safest way to start is: install environment -> start the local model API -> run a tiny smoke test -> run the full experiment.

## 1. Install The Environment

Open PowerShell in the project root. For example, change into the folder that contains this `README.md` file:

```powershell
cd "C:\path\to\thesis-rag-graphrag"
```

Create or activate the local virtual environment. If `.venv` already exists, activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by PowerShell policy, use the venv Python directly in commands:

```powershell
.\.venv\Scripts\python.exe --version
```

Install GPU PyTorch first, then the project requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify that the environment sees CUDA:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

Look for:

```text
"torch_cuda_available": true
```

or:

```text
"cuda_available": true
```

If CUDA is false, the local server will run on CPU and the experiment will be extremely slow.

## 2. Start The Local Model API

Open **Terminal 1** and run:

```powershell
.\.venv\Scripts\python.exe scripts\local_openai_server.py
```

Leave this terminal open. It is the local OpenAI-compatible model server.

The experiment code talks to it at:

```text
http://127.0.0.1:8001/v1
```

In another terminal, you can check the server:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health | ConvertTo-Json
```

You want to see:

```text
"device": "cuda"
```

If the device is `cpu`, stop the server, check the venv and CUDA PyTorch installation, then start the server again.

## 3. Run The Local Smoke Test First

Open **Terminal 2** in the project root.

List the local smoke runs:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_full_3_models_smoke_local.json --list
```

Run only Qwen first:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_full_3_models_smoke_local.json --run-id qwen_2_5_7b_smoke1_local
```

If that works, run the full 3-model smoke:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_full_3_models_smoke_local.json
```

The smoke config is intentionally tiny:

```text
dataset.limit = 1
dataset.max_passages = 2
baseline.top_k = 2
```

It proves the pipeline runs. It does not prove final answer quality.

During local runs, the code now performs a preflight check. If the local server is not running, it stops before indexing and prints a message telling you to run:

```powershell
.\.venv\Scripts\python.exe scripts\local_openai_server.py
```

If the server reports CUDA, you will see a message like:

```text
Local model API preflight OK: server is running on GPU (cuda)
```

## 4. Run The Full Local Experiment

Only run the full experiment after the local smoke test succeeds.

List the full runs:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_full_3_models.json --list
```

Run the full experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_full_3_models.json
```

This runs:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`
- `HuggingFaceH4/zephyr-7b-beta`
- baseline hybrid RAG
- GraphRAG indexing and querying
- full HybridQA dev with `limit: null`

The full output report is:

```text
results/experiments/hybridqa_full_3_models/model_comparison_report.md
```

You can also run only one full model:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_full_3_models.json --run-id qwen_2_5_7b_full_hybridqa_dev
```

## 5. What To Watch During The Full Run

Keep Terminal 1 visible. It shows the local model server activity.

Important signs:

- `Loading generation model: ... on cuda` means the model is loading on GPU.
- `Loading embedding model: ... on cuda` means embeddings are on GPU.
- GraphRAG `extract_graph` can be very slow. This is expected because it calls the LLM many times to extract entities and relationships.
- If the server says CPU, stop and fix CUDA before running the full experiment.
- If the benchmark stops with a local API preflight error, the server was not reachable. Start Terminal 1 again and rerun.

## 6. Download Data And Models Only

To prepare files without running the benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_full_3_models.json --download-only
```

This downloads or checks:

- HybridQA raw files under `data/raw/hybridqa/`
- Hugging Face models under `local_models/`

## 7. OpenRouter Smoke Test

OpenRouter is optional. Use it when you want to test the API path without running local models.

The OpenRouter smoke config is:

```text
configs/hybridqa_openrouter_smoke3_models.json
```

It uses:

- `https://openrouter.ai/api/v1`
- `OPENROUTER_API_KEY` from your environment
- `openai/text-embedding-3-small` for embeddings

Set the key in PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_key_here"
```

Run one OpenRouter smoke model:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_openrouter_smoke3_models.json --run-id qwen_2_5_7b_smoke1_openrouter
```

Run all OpenRouter smoke rows:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\hybridqa_openrouter_smoke3_models.json
```

Do not put the OpenRouter key directly in JSON. The config uses:

```json
"api_key_env": "OPENROUTER_API_KEY"
```

GraphRAG may write the resolved key into a generated workspace `.env` file under `results/`, but `results/` is ignored by git.

## 8. Running The Full Experiment With OpenRouter

The full local config can be adapted to OpenRouter by changing the `models` block in a copy of:

```text
configs/hybridqa_full_3_models.json
```

Example:

```json
"models": {
  "backend": "local_openai",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_env": "OPENROUTER_API_KEY",
  "auto_download": false,
  "generation_model": "qwen/qwen-2.5-7b-instruct",
  "embedding_model": "openai/text-embedding-3-small",
  "embedding_dimension": 1536,
  "graphrag_index_model": "qwen/qwen-2.5-7b-instruct"
}
```

Then change each experiment row to OpenRouter model names. For example:

```json
"models": {
  "generation_model": "mistralai/mistral-small-3.2-24b-instruct",
  "graphrag_index_model": "mistralai/mistral-small-3.2-24b-instruct"
}
```

Important: full GraphRAG on OpenRouter can cost real money because `extract_graph` sends many LLM calls. Use the OpenRouter smoke first.

## 9. Important Config Knobs

These are the most common settings to change.

### Dataset Size

```json
"dataset": {
  "limit": 1,
  "max_passages": 2
}
```

- `limit`: number of HybridQA questions.
- `null`: full dev split.
- `1` or `3`: smoke test.
- `max_passages`: maximum linked passages per question/table.
- Smaller values run faster but can remove evidence needed to answer.

For a more meaningful local smoke:

```json
"limit": 3,
"max_passages": 10
```

### Retrieval Size

```json
"baseline": {
  "top_k": 8,
  "dense_top_k": 30,
  "lexical_top_k": 30
}
```

- `top_k`: final contexts passed to the answer model.
- `dense_top_k`: dense retrieval candidates.
- `lexical_top_k`: BM25 candidates.
- Larger values improve recall but cost more time.

For tiny smoke tests, `top_k: 2` is okay. For quality checks, use at least `top_k: 5`.

### GraphRAG Rebuild

```json
"graphrag": {
  "force_rebuild": true
}
```

- `true`: rebuild the GraphRAG index from scratch.
- `false`: reuse an existing index when possible.

Use `true` when changing input data, model, embeddings, or GraphRAG settings. Use `false` when rerunning the same workspace to save time.

### Parallelism

```json
"parallelism": {
  "embedding_concurrent_requests": 1,
  "retrieval_concurrent_requests": 1,
  "evaluation_concurrent_requests": 1,
  "graphrag_concurrent_requests": 1
}
```

Keep all values at `1` on a single local GPU.

If using a multi-GPU server or hosted API and you know it can handle concurrency, you can use:

```json
"parallelism": {
  "llm_concurrent_requests": 4
}
```

Use higher values carefully. GraphRAG can send many requests.

## 10. Output Files

Each run writes:

- `benchmark_config.json`
- `canonical_documents.jsonl`
- `index_units.jsonl`
- `questions.jsonl`
- `predictions.jsonl`
- one folder per system
- `benchmark_summary.json`
- `comparison_summary.json`
- `comparison_report.md`

For the local smoke, check:

```text
results/experiments/hybridqa_full_3_models_smoke_local/
```

For the full local run, check:

```text
results/experiments/hybridqa_full_3_models/
```

## 11. Common Problems

### The run stops before indexing

The local API preflight failed. Start the local server in Terminal 1:

```powershell
.\.venv\Scripts\python.exe scripts\local_openai_server.py
```

Then rerun the experiment in Terminal 2.

### The server reports CPU

The local environment is not using CUDA PyTorch. Reinstall:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu128.txt
```

Then check:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

### Smoke metrics are zero

This can be normal. The smallest smoke uses only 1 question and 2 passages, so it may exclude the evidence passage needed to answer. The smoke proves that the pipeline works; it does not measure final quality.

### GraphRAG `extract_graph` is slow

This is expected. GraphRAG indexing calls the LLM repeatedly to extract entities and relationships from text chunks. Full runs can take a long time on a single laptop GPU.

## 12. Project Entry Points

Run one config:

```powershell
.\.venv\Scripts\python.exe scripts\run_benchmark.py --config configs\benchmark.example.json
```

Run a matrix:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment_matrix.py --matrix configs\benchmark_matrix.example.json
```

Start the local model API:

```powershell
.\.venv\Scripts\python.exe scripts\local_openai_server.py
```

Check the environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

More framework details are in:

```text
docs/standard_benchmark_framework.md
```
