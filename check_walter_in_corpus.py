"""
Check if Walter Payton passage is in the corpus.
"""
import json

# Load the record
with open('data/hybridqa/original/dev.jsonl', encoding='utf-8') as f:
    for line in f:
        if '00153f694413a536' in line:
            record = json.loads(line)
            break

# Build corpus
from src.baseline.corpus_builder import build_corpus_for_record

corpus_units = build_corpus_for_record(record, max_passages=30)

# Find Walter Payton passage
print("Looking for Walter Payton in corpus...")
print(f"Total units: {len(corpus_units)}")

walter_units = [u for u in corpus_units if 'Walter' in u['text']]
print(f"\nUnits mentioning 'Walter': {len(walter_units)}")

for unit in walter_units:
    print(f"\n{'='*80}")
    print(f"Type: {unit['type']}")
    print(f"ID: {unit['id']}")
    has_jerry = 'Jerry' in unit['text']
    print(f"Contains 'Jerry': {has_jerry}")
    
    if has_jerry:
        idx = unit['text'].find('Jerry')
        start = max(0, idx - 100)
        end = min(len(unit['text']), idx + 150)
        print(f"\nContext around 'Jerry':")
        print(f"...{unit['text'][start:end]}...")

# Check which passages were included (first 30)
print(f"\n{'='*80}")
print("First 30 passage links (what gets included):")
for i, p in enumerate(record['linked_passages'][:30]):
    link = p['link']
    has_walter = 'Walter' in str(p)
    marker = " ← WALTER" if has_walter else ""
    print(f"  {i+1}. {link}{marker}")
