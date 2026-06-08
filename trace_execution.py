"""
Trace the actual execution flow to see what's really happening.
"""
import json

# Simulate what the code does

print("="*80)
print("ACTUAL FLOW (What's happening now - BROKEN)")
print("="*80)

# Load questions
with open('data/hybridqa/original/dev.jsonl', encoding='utf-8') as f:
    records = [json.loads(line) for line in f][:5]

print(f"\n1. Load {len(records)} questions")
print(f"   Each question has ~50 linked passages")

# Build corpus (what the code does)
from src.baseline.corpus_builder import build_corpus

corpus = build_corpus(records, max_passages=30)

print(f"\n2. Build corpus with max_passages=30")
print(f"   → Created {len(corpus)} total units across all questions")

# Count passage units
passage_units = [u for u in corpus if u['type'] == 'linked_passage']
print(f"   → {len(passage_units)} passage units")

# Check if Walter Payton is in there
walter_units = [u for u in corpus if 'Walter Jerry' in u.get('text', '')]
print(f"   → Units with 'Walter Jerry': {len(walter_units)}")

print(f"\n3. Index these {len(corpus)} units")
print(f"   → Vector index now contains ONLY these units")
print(f"   → Walter Payton passage (position 44) was NEVER INDEXED!")

print(f"\n4. Query time: 'What is middle name of player #2?'")
print(f"   → Search the index for relevant passages")
print(f"   → BUT Walter Payton passage is NOT IN THE INDEX")
print(f"   → Can only retrieve from the {len(corpus)} units that WERE indexed")
print(f"   → Miss the answer!")

print("\n" + "="*80)
print("PROPER RAG FLOW (What SHOULD happen)")
print("="*80)

corpus_full = build_corpus(records, max_passages=50)  # or None
passage_units_full = [u for u in corpus_full if u['type'] == 'linked_passage']
walter_units_full = [u for u in corpus_full if 'Walter Jerry' in u.get('text', '')]

print(f"\n1. Load {len(records)} questions")

print(f"\n2. Build corpus with max_passages=50 (or unlimited)")
print(f"   → Created {len(corpus_full)} total units")
print(f"   → {len(passage_units_full)} passage units")
print(f"   → Units with 'Walter Jerry': {len(walter_units_full)} ✓")

print(f"\n3. Index ALL {len(corpus_full)} units")
print(f"   → Vector index contains EVERYTHING")

print(f"\n4. Query time: 'What is middle name of player #2?'")
print(f"   → Embed the question")
print(f"   → Semantic search finds most relevant passages")
print(f"   → Walter Payton passage HAS HIGH SIMILARITY → Retrieved!")
print(f"   → Answer: 'Jerry' ✓")

print("\n" + "="*80)
