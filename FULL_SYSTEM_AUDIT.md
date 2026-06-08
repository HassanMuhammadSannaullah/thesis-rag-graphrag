# COMPREHENSIVE RAG/GraphRAG SYSTEM AUDIT

## Critical Issues Found

### 1. ✅ FIXED: Pre-filtering passages before indexing
**Status**: Already fixed (changed to max_passages=None)
- Baseline: All passages now indexed
- GraphRAG: All passages now included in corpus

### 2. ⚠️ HARDCODED TRUNCATION in corpus building
**Location**: `src/graphrag_system/corpus_prep.py`
```python
if len(text) > 800:
    text = text[:800] + "..."  # ← CUTS OFF PASSAGE TEXT!
```
**Problem**: Walter Payton passage is 1795 chars, gets cut to 800 chars, might lose "Jerry"!
**Impact**: GraphRAG corpus has incomplete passages

### 3. ⚠️ HARDCODED TRUNCATION in answer generation
**Location**: `src/baseline/answer_generator.py`
```python
if len(evidence_text) > 6000:
    evidence_text = evidence_text[:6000] + "\n... (truncated)"
```
**Problem**: Arbitrary 6000 char limit on context
**Should**: Use model's actual context limit, or let context packing handle it

### 4. ⚠️ HARDCODED TRUNCATION in passage retrieval
**Location**: `src/baseline/corpus_builder.py`
```python
if len(passage_text) > 1000:
    passage_text = passage_text[:1000] + "..."
```
**Problem**: Cuts passages to 1000 chars during corpus building
**Impact**: Loses information even though passage was retrieved

### 5. ⚠️ RETRIEVAL CUTOFF LIMITS
**Location**: `src/baseline/strong_retriever.py`
```python
def retrieve_strong(...):
    result = retrieve_rows_and_passages_separately(
        top_k_rows=max(3, top_k // 2),      # ← Hardcoded formula
        top_k_passages=max(2, top_k // 3),  # ← Hardcoded formula
        max_expansion=max(2, top_k // 4),   # ← Hardcoded formula
    )
    return result["merged"][:top_k]  # ← Final cutoff
```
**Problem**: Complex hardcoded retrieval formula, not configurable
**Should**: Simple top-k by relevance score

### 6. ⚠️ NO PASSAGE CHUNKING FOR LONG TEXT
**Problem**: No intelligent chunking strategy for long passages
- 800/1000/6000 char hard cuts
- No sliding window
- No semantic chunking
- Loses information at arbitrary boundaries

### 7. ❓ CONTEXT PACKING LOGIC
**Location**: `src/baseline/context_utils.py`
- Has proper deduplication ✓
- Has proper packing logic ✓
- BUT: Gets fed already-truncated passages ✗

### 8. ⚠️ MISSING: Passage relevance re-ranking for long texts
**Problem**: If passage is >1000 chars, it gets truncated before even being scored
**Should**: 
1. Index full passage (or chunks)
2. Retrieve based on full content
3. Then intelligently select which parts to include in context

### 9. ⚠️ GRAPHRAG: No verification passages are in knowledge graph
**Problem**: GraphRAG builds graph from corpus, but corpus might have truncated passages
**Should**: Verify entity extraction works on full passages

### 10. ❓ ANSWER GENERATION: Prompt quality
**Location**: Multiple answer prompts
- Simple baseline: Basic prompt ✓
- Strong baseline: Better prompt ✓
- No explicit instruction to combine table + passage info

## Severity Assessment

### CRITICAL (Breaks RAG fundamentally):
1. ✅ Pre-filtering before indexing → FIXED

### HIGH (Significant information loss):
2. Passage truncation at 800 chars (GraphRAG corpus)
3. Passage truncation at 1000 chars (baseline corpus)
4. Context truncation at 6000 chars (answer generation)

### MEDIUM (Suboptimal but works):
5. Complex hardcoded retrieval formulas6. No semantic chunking strategy
7. Context packing receives truncated input

### LOW (Minor improvements):
9. GraphRAG entity extraction from truncated text
10. Answer generation prompt could be more explicit

## Recommended Fixes (Priority Order)

### 1. Remove ALL hardcoded truncation limits
```python
# BEFORE:
if len(text) > 800:
    text = text[:800] + "..."

# AFTER:
# No truncation - respect full passage content
# Let model context limits handle truncation intelligently
```

### 2. Implement proper passage chunking
- For passages >context limit: Use semantic chunking
- For retrieval: Index chunks separately with parent passage ID
- For answer generation: Include most relevant chunks

### 3. Simplify retrieval logic
- Remove complex hardcoded formulas (top_k//2, etc.)
- Simple approach: Retrieve top-k by score, done.
- Let semantic search do the ranking

### 4. Verify GraphRAG entity extraction
- Test that entities/relationships are extracted from full passages
- Not from truncated 800-char versions

### 5. Improve answer prompts
- Make explicit: "Combine information from tables and passages"
- Add examples of multi-hop reasoning
