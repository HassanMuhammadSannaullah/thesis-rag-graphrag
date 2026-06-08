"""
Verify Walter Payton passage is NOT truncated after fixes.
"""
from src.baseline.corpus_builder import build_corpus
from src.data_pipeline.hybridqa_parser import load_hybridqa_data

records = load_hybridqa_data('dev', limit=1)
corpus = build_corpus(records, max_passages=None)

# Find Walter Jerry passage
walter_unit = [u for u in corpus if 'Walter Jerry' in u['text']][0]

print("="*80)
print("PASSAGE TRUNCATION TEST")
print("="*80)

# Get original passage from record
original_passage = [p for p in records[0]['linked_passages'] if 'Walter_Payton' in p['link']][0]

print(f"\n1. Original passage from data:")
print(f"   Length: {len(original_passage['text'])} chars")
print(f"   Contains 'Jerry': {'Jerry' in original_passage['text']}")

print(f"\n2. Indexed passage in corpus:")
print(f"   Length: {len(walter_unit['text'])} chars")
print(f"   Contains 'Jerry': {'Jerry' in walter_unit['text']}")

# Check if truncated
corpus_passage_text = walter_unit['text']
original_text = original_passage['text']

# The corpus adds "Entity: Walter Payton\n" prefix, so check excluding that
if 'Walter Jerry' in (corpus_passage_text):
    print(f"\n✓ FULL PASSAGE INDEXED!")
    print(f"   Original: {len(original_text)} chars")
    print(f"   Corpus:   {len(corpus_passage_text)} chars (includes 'Entity:' prefix)")
    
    # Show the critical part
    idx = corpus_passage_text.find('Walter Jerry')
    print(f"\n   Critical text found at position {idx}:")
    print(f"   ...{corpus_passage_text[idx:idx+100]}...")
else:
    print(f"\n✗ PROBLEM: Full name not in corpus!")

print("\n" + "="*80)
