"""
Check if Walter Payton passage is in the index but just ranked low.
"""
from src.baseline.corpus_builder import build_corpus
from src.data_pipeline.hybridqa_parser import load_hybridqa_data
from src.baseline.vector_index import LocalVectorIndex
from pathlib import Path

# Load data
records = load_hybridqa_data('dev', limit=1)

# Build corpus
corpus = build_corpus(records, max_passages=None)

# Check if Walter Payton passage is in corpus
walter_passage = [u for u in corpus if 'Walter Jerry' in u['text']]

print("="*80)
print("VERIFICATION: Is Walter Payton passage indexed?")
print("="*80)

if walter_passage:
    print(f"\n✓ YES, Walter Payton passage IS in corpus!")
    print(f"  ID: {walter_passage[0]['id']}")
    print(f"  Length: {len(walter_passage[0]['text'])} chars")
    print(f"  Contains 'Jerry': {'Jerry' in walter_passage[0]['text']}")
else:
    print(f"\n✗ NO, Walter Payton passage NOT in corpus!")
    print(f"  This would be a corpus building problem!")

# Load the index  
index = LocalVectorIndex(Path("cache/embeddings/family_a_env_based/vector_index"))

print(f"\n\nVector Index Stats:")
print(f"  Total vectors: {index.size}")

# Do a search
question = "What is the middle name of the player with the second most National Football League career rushing yards?"

results = index.search(question, top_k=20)  # Get top 20

print(f"\n\nSearch Results for: '{question[:60]}...'")
print(f"Retrieved {len(results)} results")

# Check if Walter Payton passage is in top-20
if walter_passage:
    walter_id = walter_passage[0]['id']
    walter_rank = None
    for i, result in enumerate(results, 1):
        if result.get('id') == walter_id:
            walter_rank = i
            break
    
    if walter_rank:
        print(f"\n✓ Walter Payton passage found at rank #{walter_rank}/20")
        print(f"  Score: {results[walter_rank-1].get('score', 'N/A')}")
        if walter_rank > 8:
            print(f"\n⚠️  PROBLEM: Ranked #{walter_rank}, but only top-8 retrieved!")
            print(f"     The EMBEDDING MODEL ranked it too low!")
    else:
        print(f"\n✗ Walter Payton passage NOT in top-20!")
        print(f"   The EMBEDDING MODEL ranked it very poorly!")
        
print("\n" + "="*80)
