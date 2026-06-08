"""
Check if GraphRAG has the same max_passages problem.
"""
import json

# Load test questions
with open('data/hybridqa/original/dev.jsonl', encoding='utf-8') as f:
    records = [json.loads(line) for line in f][:5]

print("="*80)
print("CHECKING GRAPHRAG CORPUS BUILDING")
print("="*80)

# Test different strategies
from src.graphrag_system.improved_corpus_prep import (
    build_graphrag_corpus,
    build_table_centric_corpus,
    build_hybrid_corpus,
)

print("\n--- Strategy: table_centric (default max_passages=30) ---")
docs_table = build_table_centric_corpus(records, max_passages=30)
print(f"Total documents: {len(docs_table)}")

# Check if Walter Payton is in any doc
walter_docs = [d for d in docs_table if 'Walter Jerry' in d['text']]
print(f"Documents with 'Walter Jerry': {len(walter_docs)}")

print("\n--- Strategy: hybrid (default max_passages_in_overview=10) ---")
docs_hybrid = build_hybrid_corpus(records, max_passages_in_overview=10)
print(f"Total documents: {len(docs_hybrid)}")

walter_hybrid = [d for d in docs_hybrid if 'Walter Jerry' in d['text']]
print(f"Documents with 'Walter Jerry': {len(walter_hybrid)}")

print("\n--- Testing with max_passages=50 (all passages) ---")
docs_table_full = build_table_centric_corpus(records, max_passages=50)
print(f"Total documents: {len(docs_table_full)}")

walter_full = [d for d in docs_table_full if 'Walter Jerry' in d['text']]
print(f"Documents with 'Walter Jerry': {len(walter_full)} ✓")

if walter_full:
    print(f"\nFOUND in document: {walter_full[0]['id']}")
    idx = walter_full[0]['text'].find('Walter Jerry')
    context = walter_full[0]['text'][max(0, idx-50):idx+100]
    print(f"Context: ...{context}...")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"GraphRAG with max_passages=30:  Walter Jerry found: {len(walter_docs) > 0}")
print(f"GraphRAG with max_passages=10:  Walter Jerry found: {len(walter_hybrid) > 0}")  
print(f"GraphRAG with max_passages=50:  Walter Jerry found: {len(walter_full) > 0} ✓")
print("\n⚠️  GraphRAG HAS THE SAME PROBLEM!")
print("="*80)
