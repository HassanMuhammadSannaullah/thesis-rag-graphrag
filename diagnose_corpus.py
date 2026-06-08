"""
Quick diagnostic script to check the corpus building process.
"""
import json
from pathlib import Path

# Load a record from parsed data
with open('data/hybridqa/original/dev.jsonl', encoding='utf-8') as f:
    for line in f:
        if '00153f694413a536' in line:
            record = json.loads(line)
            break

print("="*80)
print("DIAGNOSTIC: CORPUS BUILDING PROCESS")
print("="*80)

print(f"\n1. Original record (from data/hybridqa/original/dev.jsonl):")
print(f"   Question: {record['question']}")
print(f"   Answer: {record['answer']}")
print(f"   linked_passages count: {len(record.get('linked_passages', []))}")

# Check first few passages
if record.get('linked_passages'):
    print(f"\n   First 3 passage links:")
    for i, p in enumerate(record['linked_passages'][:3]):
        print(f"     {i+1}. {p['link']} ({len(p.get('text', ''))} chars)")

# Simulate corpus builder
from src.baseline.corpus_builder import build_corpus_for_record

print(f"\n2. Building corpus units from this record...")
corpus_units = build_corpus_for_record(record, max_passages=30)

print(f"   Total units created: {len(corpus_units)}")

# Count by type
by_type = {}
for unit in corpus_units:
    t = unit.get('type', 'unknown')
    by_type[t] = by_type.get(t, 0) + 1

print(f"   Breakdown:")
for t, count in by_type.items():
    print(f"     - {t}: {count}")

# Check if any passage units were created
passage_units = [u for u in corpus_units if u['type'] == 'linked_passage']
print(f"\n3. Linked passage units created: {len(passage_units)}")

if passage_units:
    print(f"   First passage unit:")
    print(f"     ID: {passage_units[0]['id']}")
    print(f"     Text (first 200 chars): {passage_units[0]['text'][:200]}...")
else:
    print(f"   ⚠️  NO PASSAGE UNITS CREATED!")
    print(f"\n4. Debugging why no passages...")
    print(f"   Input record has 'linked_passages' key: {'linked_passages' in record}")
    print(f"   Type: {type(record.get('linked_passages'))}")
    print(f"   Length: {len(record.get('linked_passages', []))}")
    
    if record.get('linked_passages'):
        print(f"\n   First passage structure:")
        first = record['linked_passages'][0]
        print(f"     Keys: {list(first.keys())}")
        print(f"     Link: {first.get('link')}")
        print(f"     Has text: {'text' in first}")
        print(f"     Text length: {len(first.get('text', ''))}")

print("\n" + "="*80)
