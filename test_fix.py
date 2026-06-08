"""
Test that the fix works - verify all passages are now indexed.
"""
from src.baseline.corpus_builder import build_corpus
from src.data_pipeline.hybridqa_parser import load_hybridqa_data

print("Testing corpus building with fixed code...")
print("="*80)

# Load one question
records = load_hybridqa_data('dev', limit=1)
record = records[0]

print(f"\n1. Question: {record['question'][:80]}...")
print(f"   Total linked passages in data: {len(record['linked_passages'])}")

# Build corpus with new fixed code (no limit)
corpus = build_corpus(records, max_passages=None)

passage_units = [u for u in corpus if u['type'] == 'linked_passage']
print(f"\n2. Corpus building (max_passages=None):")
print(f"   Total corpus units: {len(corpus)}")
print(f"   Passage units created: {len(passage_units)}")

# Verify
if len(passage_units) == len(record['linked_passages']):
    print(f"\n✓ SUCCESS! ALL {len(passage_units)} PASSAGES INDEXED!")
    print("   No pre-filtering - semantic search can now find everything!")
else:
    print(f"\n✗ PROBLEM: Only {len(passage_units)}/{len(record['linked_passages'])} passages indexed")
    print("   Still filtering somewhere!")

# Check if Walter Jerry is accessible
walter_units = [u for u in corpus if 'Walter Jerry' in u['text']]
print(f"\n3. Critical test: Walter Jerry Payton")
if walter_units:
    print(f"   ✓ FOUND in {len(walter_units)} unit(s)")
    print(f"   ID: {walter_units[0]['id']}")
else:
    print(f"   ✗ NOT FOUND (might be in a different question)")

print("\n" + "="*80)
