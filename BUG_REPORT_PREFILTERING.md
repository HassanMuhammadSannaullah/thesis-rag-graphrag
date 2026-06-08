# CRITICAL BUGS FOUND: Pre-filtering Destroying RAG

## Problem: The system pre-filters passages BEFORE semantic search
This completely defeats the purpose of RAG. You can't find relevant information if you throw it away before searching!

## All Instances Found:

### 1. src/baseline/corpus_builder.py
- `build_linked_passages(record, max_passages=30)` ← Cuts at 30
- `build_corpus_for_record(record, max_passages=30)` ← Cuts at 30
- `build_corpus(records, max_passages=30)` ← Cuts at 30

### 2. src/graphrag_system/corpus_prep.py
- `hybridqa_record_to_text(record, max_passages_per_record=10)` ← Cuts at 10!
- `hybridqa_to_graphrag_docs(records, max_passages_per_record=10)` ← Cuts at 10!

### 3. src/graphrag_system/improved_corpus_prep.py
- `build_row_centric_corpus(records, max_passages=30)` ← Cuts at 30
- `build_table_centric_document(record, max_passages=30)` ← Cuts at 30
- `build_table_centric_corpus(records, max_passages=30)` ← Cuts at 30
- `build_hybrid_documents(record, max_passages_in_overview=10)` ← Cuts at 10!
- `build_hybrid_corpus(records, max_passages_in_overview=10)` ← Cuts at 10!

### 4. src/config/settings.py
- `MAX_LINKED_PASSAGES = 3` ← ONLY 3?! Insane!

### 5. scripts/17_run_phase5_experiments.py
- Family A: `pipeline_shared.prepare(records, max_passages=30)` ← Cuts at 30
- Family B: `pipeline_strong.prepare(records, max_passages=30)` ← Cuts at 30
- GraphRAG calls don't override max_passages (uses defaults)

## Impact:
- HybridQA has ~50 passages per question
- Walter Payton answer is at position 44
- System cuts off at positions 3, 10, or 30
- Answer is NEVER indexed
- Model can't find what was never indexed
- Result: 0.000 F1 (but model is actually correct!)

## Fix: 
REMOVE ALL PRE-FILTERING. Let semantic search do its job!
- Index ALL passages
- Use top_k at QUERY TIME (retrieval), not corpus-building time
- This is how RAG is supposed to work!
