"""Verify corpus has row_links."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jsonlines
from src.config import settings as cfg
from src.baseline.corpus_builder import build_corpus, save_corpus, load_corpus

records = list(jsonlines.open(str(cfg.SAMPLES_DIR / "dev_sample.jsonl")))
corpus = build_corpus(records, max_passages=cfg.MAX_LINKED_PASSAGES)
save_corpus(corpus, cfg.PARSED_DIR / "baseline_corpus.jsonl")

# Check row_links
rows = [u for u in corpus if u["type"] == "table_row"]
rows_with_links = [u for u in rows if u.get("row_links")]
print(f"Rows with links: {len(rows_with_links)}/{len(rows)}")

# Check Q1 row (Walter Payton = row index 1)
q1_rows = [u for u in rows if "List_of_National_Football_League_rushing_yards_leaders" in u["id"]]
for r in q1_rows[:3]:
    print(f"\n{r['id']}: {r['text'][:100]}")
    print(f"  row_links: {r.get('row_links', [])[:3]}")

# Check if Walter Payton passage is in the corpus
passages = [u for u in corpus if u["type"] == "linked_passage"]
print(f"\nTotal passages: {len(passages)}")
for p in passages:
    if "walter" in p.get("link", "").lower() or "payton" in p["text"].lower():
        print(f"  FOUND: {p['link']} -> {p['text'][:150]}")
