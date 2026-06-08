# Phase 5: Split Experiment Types Clearly

**Status**: ✅ Complete
**Files**: `scripts/17_run_phase5_experiments.py`, `src/graphrag_system/query_runner.py`, `scripts/test_phase5_smoke.py`

## Overview

Phase 5 addresses a critical methodological issue: **mixing two different research questions into one experiment design**. The solution is to create two distinct experiment families with clear separation and proper naming.

## The Problem

Previous experiments conflated two fundamentally different questions:
1. **Retrieval method comparison**: Does graph-aware retrieval help over pure vector search when text units are controlled?
2. **System comparison**: Which full-stack QA system (baseline RAG vs GraphRAG) performs better in practice?

Mixing these leads to:
- Unclear interpretation of results
- Difficulty defending methodology in thesis
- Inability to isolate what causes performance differences
- Confusion about which findings generalize

## The Solution: Two Experiment Families

### Family A: Controlled Retrieval Comparison

**Research Question**: Does graph-aware retrieval help over vector retrieval when evidence units are controlled?

**Systems**:
- `baseline_shared_text_unit_rag`: Simple vector retrieval (no lexical, no reranking)
- `pure_graphrag`: GraphRAG retrieval with graph-aware community ranking

**Key Properties**: - Same text source documents for both systems- Minimal processing differences- Isolates impact of graph structure on retrieval
- Fair comparison of retrieval *methods*

**When to use**: Thesis claims about retrieval algorithms, graph-aware indexing benefits

### Family B: Realistic End-to-End System Comparison

**Research Question**: Which system is better as an actual QA pipeline on HybridQA?

**Systems**:
- `baseline_strong_rag`: Full-featured baseline with hybrid scoring + reranking (Phase 3)
- `graphrag_practical`: GraphRAG with improved corpus construction (Phase 4)

**Key Properties**:
- Best-practice implementations of each paradigm
- Different preprocessing (row-aware vs graph-indexed)
- Represents real deployment scenarios
- Fair comparison of *systems*

**When to use**: Practical guidance, industry recommendations, system performance claims

## Architecture

### Unified Experiment Runner

`scripts/17_run_phase5_experiments.py` orchestrates both families with:
- Clear family separation (A vs B)
- Shared infrastructure (data loading, model variants, evaluation)
- Distinct output directories
- Comprehensive metadata tracking

### GraphRAG Query Runner

`src/graphrag_system/query_runner.py` provides:
- Batch query execution against indexed GraphRAG workspace
- Automatic result formatting for evaluation
- Timing statistics
- Error handling

### Integration with Previous Phases

- **Phase 3**: Strong baseline RAG with simple/strong variants
- **Phase 3.5**: Model variant system for ablation studies
- **Phase 4**: Improved GraphRAG corpus construction strategies

## Usage

### Quick Start

```bash
# Test infrastructure
python scripts/test_phase5_smoke.py

# Run Family A (controlled comparison) - baseline only (fast)
python scripts/17_run_phase5_experiments.py --family A --split dev --limit 10 --skip-graphrag

# Run Family B (realistic comparison) - baseline only (fast)
python scripts/17_run_phase5_experiments.py --family B --split dev --limit 10 --skip-graphrag

# Run both families with GraphRAG (requires indexing)
python scripts/17_run_phase5_experiments.py --family both --split dev --limit 20
```

### Command-Line Arguments

`--family {A, B, both}` (required)
  - Which experiment family to run

`--split {dev, train}`
  - Dataset split (default: dev)

`--limit N`
  - Number of questions (default: all)

`--corpus-strategy {row_centric, table_centric, entity_centric, hybrid}`
  - GraphRAG corpus construction strategy (default: hybrid)

`--variant NAME`
  - Model variant from Phase 3.5 registry (e.g., qwen_14b_e5_base)
  - If not specified, uses environment variables

`--output-root PATH`
  - Root directory for outputs (default: results/experiments/phase5)

`--force-rebuild-corpus`
  - Force rebuild GraphRAG corpus even if exists

`--force-rebuild-index`
  - Force rebuild GraphRAG index even if exists

`--skip-graphrag`
  - Skip GraphRAG experiments (only run baseline)

### Example Workflows

#### 1. Quick Test (Baseline Only)

```bash
# Test both families without GraphRAG (fast)
python scripts/17_run_phase5_experiments.py \
  --family both \
  --split dev \
  --limit 5 \
  --skip-graphrag
```

**Time**: ~1-2 minutes  
**Output**: Baseline results for both families

#### 2. Small-Scale Comparison

```bash
# Run full comparison on 20 questions
python scripts/17_run_phase5_experiments.py \
  --family both \
  --split dev \
  --limit 20 \
  --corpus-strategy hybrid
```

**Time**: ~10-20 minutes (includes GraphRAG indexing)  
**Output**: Complete results for baseline vs GraphRAG in both families

#### 3. Production Run

```bash
# Full dev split, best strategy, specific model
python scripts/17_run_phase5_experiments.py \
  --family both \
  --split dev \
  --corpus-strategy hybrid \
  --variant qwen_14b_e5_base
```

**Time**: 2-6 hours (depends on hardware and dev split size)  
**Output**: Thesis-quality experiment results

#### 4. Model Ablation Study

```bash
# Run multiple variants for model ablation (Phase 6.5)
for variant in qwen_3b_e5_base mistral_7b_e5_base qwen_14b_e5_base; do
  python scripts/17_run_phase5_experiments.py \
    --family B \
    --split dev \
    --limit 50 \
    --variant $variant
done
```

## Output Structure

Each experiment run creates a timestamped directory:

```
results/experiments/phase5/
  ├── B_dev_20250607_143022/           # Experiment run
  │   ├── experiment_summary.json      # Master summary
  │   ├── family_a/                    # Family A results
  │   │   ├── family_a_baseline_shared_text_unit_rag/
  │   │   │   ├── predictions.json
  │   │   │   ├── metrics.json
  │   │   │   └── metadata.json
  │   │   └── family_a_pure_graphrag/
  │   │       ├── predictions.json
  │   │       ├── metrics.json
  │   │       └── metadata.json
  │   └── family_b/                    # Family B results
  │       ├── family_b_baseline_strong_rag/
  │       │   ├── predictions.json
  │       │   ├── metrics.json
  │       │   └── metadata.json
  │       └── family_b_graphrag_practical/
  │           ├── predictions.json
  │           ├── metrics.json
  │           └── metadata.json
```

### Metadata Tracking

Each experiment saves comprehensive metadata:

```json
{
  "experiment_family": "B",
  "family_purpose": "realistic_endtoend_comparison",
  "system_name": "baseline_strong_rag",
  "baseline_variant": "strong",
  "corpus_strategy": "hybrid",
  "model_variant": {
    "name": "qwen_14b_e5_base",
    "generation_model": "Qwen/Qwen2.5-14B-Instruct",
    "embedding_model": "intfloat/e5-base-v2",
    "model_family": "qwen",
    "model_size_b": 14.0
  },
  "num_questions": 50,
  "total_time_seconds": 1847.3
}
```

## Key Design Decisions

### 1. Separate Text Units vs Shared Text Units

**Family A** uses shared text units to control variables:
- Both systems see identical documents
- Differences come purely from retrieval/ranking method
- Enables clean ablation of graph benefit

**Family B** uses different preprocessing:
- Baseline: Row-aware passage expansion
- GraphRAG: Graph-indexed communities
- Reflects real-world system differences

### 2. Simple vs Strong Baseline

**Family A** uses "simple" baseline:
- Pure dense vector retrieval
- No lexical scoring
- No cross-encoder reranking
- Minimum viable baseline for controlled comparison

**Family B** uses "strong" baseline:
- Hybrid lexical + semantic scoring
- Cross-encoder reranking
- Row-aware context expansion
- Best-practice baseline for realistic comparison

### 3. Output Naming Convention

All outputs are prefixed with family:
- `family_a_baseline_shared_text_unit_rag`
- `family_a_pure_graphrag`
- `family_b_baseline_strong_rag`
- `family_b_graphrag_practical`

This prevents confusion in thesis reports and analysis.

### 4. Corpus Strategy Selection

GraphRAG experiments support all Phase 4 strategies:
- `row_centric`: One doc per row (fine-grained)
- `table_centric`: One doc per table (holistic)
- `entity_centric`: One doc per entity (entity-focused)
- `hybrid`: Table overviews + row details (recommended)

Default is `hybrid` for balanced performance.

## Integration with Phases 3-4

### Phase 3: Strong Baseline RAG

```python
# Family A: Simple variant (ablation baseline)
pipeline_simple = StrongBaselinePipeline(
    variant="simple",
    use_lexical=False,
    use_reranking=False,
)

# Family B: Strong variant (best-practice)
pipeline_strong = StrongBaselinePipeline(
    variant="strong",
    use_lexical=True,
    use_reranking=True,
)
```

### Phase 3.5: Model Variants

```python
# Use specific variant
variant = model_variants.get_variant("qwen_14b_e5_base")

# Set environment variables
env_vars = variant.to_env_dict()
for key, value in env_vars.items():
    os.environ[key] = value
```

### Phase 4: Improved GraphRAG Corpus

```python
# Build corpus with selected strategy
hybridqa_to_improved_graphrag_docs(
    records=records,
    output_dir=graphrag_input_dir,
    strategy="hybrid",  # or row_centric, table_centric, entity_centric
    format="txt",
)
```

## Evaluation

Both families use the same evaluation metrics:
- **Answer correctness**: F1, Exact Match
- **Retrieval quality**: Recall@K, Precision@K (if evidence available)
- **Efficiency**: Query time, total time

Results are saved with full provenance for reproducibility.

## Testing

### Smoke Test

```bash
python scripts/test_phase5_smoke.py
```

Tests:
1. ✅ Phase 5 structure
2. ✅ GraphRAG query runner
3. ✅ Data loading
4. ✅ Model variant loading
5. ✅ Baseline pipeline
6. ✅ Improved corpus preparation

**Expected time**: <30 seconds

### Integration Test

```bash
# Fast test (baseline only)
python scripts/17_run_phase5_experiments.py --family both --limit 3 --skip-graphrag
```

**Expected time**: ~30-60 seconds

### Full System Test

```bash
# With GraphRAG (requires indexing)
python scripts/17_run_phase5_experiments.py --family A --limit 5
```

**Expected time**: ~5-10 minutes (first run with indexing)

## Expected Results

Based on Phase 3-4 improvements:

### Family A (Controlled Comparison)

**Baseline (simple vector)**: Answer F1 ~0.40-0.50  
**GraphRAG**: Answer F1 ~0.45-0.55 (10-20% improvement expected)

**Interpretation**: Graph-aware retrieval provides moderate benefit over pure vector search when text units are controlled.

### Family B (Realistic Comparison)

**Baseline (strong)**: Answer F1 ~0.50-0.65  
**GraphRAG (hybrid corpus)**: Answer F1 ~0.55-0.70 (5-15% improvement expected)

**Interpretation**: Both systems perform well when optimized; GraphRAG may excel on multi-hop questions.

## Next Steps (Phase 6+)

After Phase 5:

1. **Phase 6**: Strengthen evaluation (larger samples, statistical tests)
2. **Phase 6.5**: Model ablation study (test across 3+ LLMs, 2+ embeddings)
3. **Phase 6.6**: Statistical significance testing
4. **Phase 6.7**: Error analysis (categorize failure modes)

Phase 5 provides the infrastructure for all of these.

## Acceptance Criteria

- [x] Two experiment families clearly separated
- [x] Family A uses simple baseline + GraphRAG
- [x] Family B uses strong baseline + improved GraphRAG
- [x] Clear output naming prevents confusion
- [x] Comprehensive metadata tracking
- [x] Integration with Phases 3-4
- [x] GraphRAG query runner implemented
- [x] Smoke test suite passes
- [x] Documentation with usage examples

**Phase 5 Complete!** Ready for Phase 6 (strengthen evaluation).

## Troubleshooting

### GraphRAG Indexing Fails

**Symptoms**: `run_graphrag_index` returns error  
**Solutions**:
- Check local LLM server is running
- Verify `GRAPHRAG_API_KEY` environment variable
- Check GraphRAG CLI installed: `pip install graphrag`
- Try `--force-rebuild-index` flag

### Out of Memory

**Symptoms**: Process killed during indexing  
**Solutions**:
- Use smaller `--limit` (start with 10-20 questions)
- Use smaller model variant (qwen_3b instead of qwen_14b)
- Close other applications
- Increase system swap space

### Slow Performance

**Symptoms**: Experiments take too long  
**Solutions**:
- Use `--skip-graphrag` to test baseline only first
- Reduce `--limit` for development/testing
- Ensure GPU is being used (check CUDA availability)
- Use smaller model variants for faster inference

### Import Errors

**Symptoms**: Module not found errors  
**Solutions**:
- Verify all Phase 3-4 files are created
- Check Python path includes project root
- Reinstall dependencies: `pip install -r requirements.txt`

## Files Created

```
scripts/17_run_phase5_experiments.py  (~800 lines)
├── Argument parsing and configuration
├── Model variant loading
├── Family A: Controlled retrieval comparison
│   ├── Baseline shared text unit RAG
│   └── Pure GraphRAG
├── Family B: Realistic system comparison
│   ├── Baseline strong RAG
│   └── GraphRAG practical
└── Results summary and saving

src/graphrag_system/query_runner.py  (~170 lines)
├── run_graphrag_batch_queries (batch execution)
└── run_graphrag_experiment (with timing stats)

scripts/test_phase5_smoke.py  (~280 lines)
├── Test 1: Phase 5 structure
├── Test 2: GraphRAG query runner
├── Test 3: Data loading
├── Test 4: Model variant loading
├── Test 5: Baseline pipeline
└── Test 6: Improved corpus prep

docs/phase5_split_experiments.md  (this file)
```

## Summary

Phase 5 transforms the thesis methodology from "mixed comparisons" to "clear, defendable experiments" by:

1. **Separating research questions** into two distinct families
2. **Controlling variables** appropriately for each question
3. **Using correct baselines** (simple for controlled, strong for realistic)
4. **Clear naming** that prevents methodological confusion
5. **Comprehensive tracking** for reproducibility and defense

This enables defendable thesis claims about both retrieval methods and practical systems.
