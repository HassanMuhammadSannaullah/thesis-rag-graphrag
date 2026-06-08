# Phase 4: GraphRAG Source Construction Improvements

**Status**: ✅ Complete  
**Files**: `src/graphrag_system/improved_corpus_prep.py`, `scripts/test_graphrag_strategies.py`

## Overview

Phase 4 replaces the simplistic GraphRAG corpus preparation with four sophisticated document construction strategies. The goal is to better preserve table structure, row-level granularity, and entity context while maintaining stable provenance IDs for retrieval evaluation.

## Problem Statement

The original `corpus_prep.py` had several limitations:

1. **Overly simplistic**: Combined all table rows and passages into a single document
2. **Loss of structure**: Couldn't retrieve specific rows or entities
3. **Truncation issues**: 800-character limit lost important context
4. **Unstable IDs**: Made evaluation and debugging difficult
5. **Poor retrieval**: GraphRAG couldn't outperform baseline due to poor source quality

## Four Document Construction Strategies

### 1. Row-Centric (Default)

**Philosophy**: Each row is a document with its linked passages.

**Structure**:
```
[DOCUMENT_ID: table_123::row_2]
[TABLE_ID: table_123]
[ROW_ID: row_2]

# Table Title - Row 2

Row Data:
Column1: Value1
Column2: Value2
...

Related Entities:
Entity1 - Description (1500 chars max)
Entity2 - Description (1500 chars max)
...
```

**When to use**:
- Questions focus on specific rows
- Need fine-grained retrieval
- Want to minimize false positives from irrelevant rows

**Pros**: 
- High precision for row-specific questions
- Natural granularity matches HybridQA structure
- Easy to trace provenance back to exact rows

**Cons**:
- May miss table-level context
- Higher document count (could slow indexing)

### 2. Table-Centric

**Philosophy**: One comprehensive document per table with all rows and entities.

**Structure**:
```
[DOCUMENT_ID: table_123]
[TABLE_ID: table_123]

# Table Title

## Introduction
Table intro text...

## Section: Section Title

Row 0:
Column1: Value1, Column2: Value2, ...

Row 1:
Column1: Value1, Column2: Value2, ...

## Related Entities
Entity1: Description (1500 chars)
Entity2: Description (1500 chars)
...
```

**When to use**:
- Questions require multiple rows
- Need holistic table understanding
- Want fewer, richer documents

**Pros**:
- All table context in one place
- Easier for LLM to see relationships
- Lower document count

**Cons**:
- Can be very long (may exceed context limits)
- Less precise retrieval
- Harder to identify exact source row

### 3. Entity-Centric

**Philosophy**: One document per entity (passage) with table context.

**Structure**:
```
[DOCUMENT_ID: entity_Eiffel_Tower]
[PASSAGE_ID: passage_456]
[TABLE_ID: table_123]

# Eiffel Tower

Entity Content:
The Eiffel Tower is... (full passage, 2000 chars max)

Context from Table: table_123
Title: Famous Buildings
This entity appears in row 3:
Column1: Eiffel Tower, Column2: Paris, Column3: 1889
```

**When to use**:
- Questions focus on entities/people/places
- Need deep entity information
- Want to leverage Wikipedia-style passages

**Pros**:
- Rich entity descriptions
- Good for entity-focused questions
- Preserves passage structure

**Cons**:
- Table context may be sparse
- Doesn't capture row relationships well
- Many documents if many entities

### 4. Hybrid (Recommended)

**Philosophy**: Best of both worlds - table overview + detailed rows.

**Structure**:

**Overview Document:**
```
[DOCUMENT_ID: table_123_overview]
[TABLE_ID: table_123]

# Table Title - Overview

## Introduction
Intro text...

## Structure
Columns: Column1, Column2, ...
Number of rows: 10
Related entities: 15

## Sample Related Entities
- Entity1
- Entity2
...
```

**Row Documents** (same as row-centric):
```
[DOCUMENT_ID: table_123::row_2]
...
```

**When to use**:
- Want flexibility in retrieval
- Both overview and details matter
- Can afford higher document count

**Pros**:
- Flexible: Can retrieve overview OR specific rows
- Preserves both macro and micro context
- Good for diverse question types

**Cons**:
- Highest document count
- May retrieve redundant context (overview + rows)

## API Usage

### Basic Usage

```python
from src.graphrag_system.improved_corpus_prep import build_graphrag_corpus
from src.data_pipeline.hybridqa_parser import load_hybridqa_data

# Load data
records = load_hybridqa_data("dev")

# Build corpus with default (row-centric) strategy
docs = build_graphrag_corpus(records, strategy="row_centric")

# Try other strategies
docs_table = build_graphrag_corpus(records, strategy="table_centric")
docs_entity = build_graphrag_corpus(records, strategy="entity_centric")
docs_hybrid = build_graphrag_corpus(records, strategy="hybrid")
```

### Save to Disk

```python
from src.graphrag_system.improved_corpus_prep import save_graphrag_corpus
from pathlib import Path

output_dir = Path("graphrag_workspace/my_experiment")

# Save as individual .txt files (for GraphRAG ingestion)
save_graphrag_corpus(docs, output_dir, format="txt")

# Or save as .jsonl (for manual inspection/processing)
save_graphrag_corpus(docs, output_dir, format="jsonl")
```

### Compare Strategies

```python
from src.graphrag_system.improved_corpus_prep import compare_strategies

# Compare on first 10 records
comparison = compare_strategies(records, sample_size=10)

for strategy, stats in comparison.items():
    print(f"{strategy}: {stats['num_documents']} docs, "
          f"avg {stats['avg_doc_length_chars']:.0f} chars")
```

### One-Stop Convenience Function

```python
from src.graphrag_system.improved_corpus_prep import hybridqa_to_improved_graphrag_docs

# Build and save in one call
docs = hybridqa_to_improved_graphrag_docs(
    records=records,
    output_dir=Path("graphrag_workspace/hybrid_corpus"),
    strategy="hybrid",
    format="txt",
)
```

## Integration with Existing Pipeline

### Before (Phase 0-2):

```python
# Old approach in graphrag_runner.py
from src.graphrag_system.corpus_prep import hybridqa_to_graphrag_docs

docs = hybridqa_to_graphrag_docs(records, output_dir)  # Simplistic
```

### After (Phase 4):

```python
# New approach
from src.graphrag_system.improved_corpus_prep import hybridqa_to_improved_graphrag_docs

docs = hybridqa_to_improved_graphrag_docs(
    records,
    output_dir,
    strategy="hybrid",  # Choose your strategy
    format="txt",
)
```

Then run GraphRAG indexing as usual:
```bash
graphrag index --root ./graphrag_workspace/hybrid_corpus
```

## Improvements Over Original

| Aspect | Original (corpus_prep.py) | Improved (Phase 4) |
|--------|---------------------------|-------------------|
| **Document Granularity** | One doc per table | 4 strategies: row/table/entity/hybrid |
| **Max Length** | 800 chars | 1500-2000 chars per section |
| **Structure Preservation** | Lost | Maintained with metadata headers |
| **Provenance IDs** | Unstable | Stable, predictable (table_id::row_idx) |
| **Metadata** | Minimal | Rich (table_id, row_id, passage_id, doc_type, strategy) |
| **Flexibility** | One size fits all | Choose strategy per experiment |
| **Traceability** | Difficult | Easy (IDs directly map to source) |

## Testing

Run the test suite:
```bash
python scripts/test_graphrag_strategies.py
```

Tests cover:
1. ✅ Basic corpus building (all strategies work)
2. ✅ Document ID stability (same input → same IDs)
3. ✅ Metadata preservation (all fields present)
4. ✅ Saving and loading (txt and jsonl formats)
5. ✅ Strategy comparison (side-by-side stats)
6. ✅ Convenience function (end-to-end)

## Strategy Selection Guide

**Choose based on your research questions:**

| Question Type | Recommended Strategy | Reason |
|--------------|---------------------|---------|
| "What is [entity]'s [attribute] in [table]?" | Row-centric or Entity-centric | Need specific row or entity |
| "Compare [entity1] and [entity2]" | Table-centric or Hybrid | Need multiple rows together |
| "Who/what is [entity]?" | Entity-centric | Pure entity lookup |
| "How many [category] in [table]?" | Table-centric | Need full table view |
| **Mixed questions (HybridQA)** | **Hybrid** | Handles diverse question types |

**For thesis experiments**: Use **Hybrid** as primary, then ablate to row-centric and table-centric to measure impact.

## Expected Impact

Based on GraphRAG design principles, improved source construction should yield:

- **Better community detection**: Clear row/entity boundaries help LLM identify clusters
- **Richer summaries**: More context → better community summaries
- **More precise retrieval**: Row-level documents reduce false positives
- **Improved answer quality**: Better sources → better final answers

**Hypothesis for Phase 5**: Hybrid strategy will outperform original by 10-20% on answer F1 due to better source quality.

## Next Steps (Phase 5)

1. Rebuild GraphRAG indexes with hybrid strategy
2. Run full retrieval comparison:
   - Baseline RAG (Phase 3 strong retriever)
   - GraphRAG (original corpus_prep)
   - GraphRAG (hybrid strategy)
   - GraphRAG (row-centric strategy)
3. Measure retrieval metrics (Recall@k, Precision@k, MRR)
4. Analyze which strategy works best for different question types

## Files Created

```
src/graphrag_system/improved_corpus_prep.py  (~550 lines)
├── Helper functions (_get_row_id, _format_row_content, etc.)
├── Strategy 1: Row-centric (build_row_centric_document/corpus)
├── Strategy 2: Table-centric (build_table_centric_document/corpus)
├── Strategy 3: Entity-centric (build_entity_centric_documents/corpus)
├── Strategy 4: Hybrid (build_hybrid_documents/corpus)
├── Main builder: build_graphrag_corpus (strategy dispatcher)
├── Saver: save_graphrag_corpus (txt/jsonl output)
├── Comparison: compare_strategies
└── Convenience: hybridqa_to_improved_graphrag_docs

scripts/test_graphrag_strategies.py  (~380 lines)
├── Test 1: Basic corpus building
├── Test 2: Document ID stability
├── Test 3: Metadata preservation
├── Test 4: Corpus saving and loading
├── Test 5: Strategy comparison
└── Test 6: Convenience function

docs/phase4_graphrag_improvements.md  (this file)
```

## Acceptance Criteria

- [x] Four document construction strategies implemented
- [x] Stable, predictable document IDs
- [x] Rich metadata preservation
- [x] Support both txt and jsonl output formats
- [x] Strategy comparison utility
- [x] Comprehensive test suite (6 tests)
- [x] Documentation with usage examples
- [x] Integration guide for existing pipeline

**Phase 4 Complete!** Ready for Phase 5 retrieval comparison.
