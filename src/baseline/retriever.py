"""
Baseline retrieval: two-stage hybrid approach.
Stage 1: Vector search over table summaries and rows
Stage 2: For retrieved table rows, follow entity links to fetch passages from a lookup

This avoids needing to pre-select which passages to embed.
The passage lookup is built from the full parsed records.
"""
from src.baseline.vector_index import LocalVectorIndex


def build_passage_lookup(records: list[dict]) -> dict[str, dict]:
    """Build a lookup dict: wiki_link -> {link, text, table_id}."""
    lookup = {}
    for rec in records:
        for lp in rec.get("linked_passages", []):
            link = lp["link"]
            if link not in lookup and lp["text"].strip():
                text = lp["text"]
                if len(text) > 1000:
                    text = text[:1000] + "..."
                entity_name = link.split("/")[-1].replace("_", " ")
                lookup[link] = {
                    "id": f"passage_lookup_{link}",
                    "type": "linked_passage",
                    "text": f"Entity: {entity_name}\n{text}",
                    "table_id": rec["table_id"],
                    "link": link,
                    "score": -0.1,
                }
    return lookup


def hybrid_retrieve(
    question: str,
    index: LocalVectorIndex,
    passage_lookup: dict[str, dict],
    top_k: int = 5,
    max_expansion: int = 5,
    max_total: int = 10,
) -> list[dict]:
    """
    Two-stage retrieval:
      1) Vector search for top-k matches (summaries + rows)
      2) For retrieved rows, follow entity links to add passages from lookup
    """
    # Stage 1: vector search
    candidates = index.search(question, top_k=top_k)

    # Stage 2: for retrieved rows, follow their entity links
    seen_ids = {c.get("id") for c in candidates}
    expansion = []

    for cand in candidates:
        if cand.get("type") != "table_row":
            continue
        row_links = cand.get("row_links", [])
        for link in row_links:
            if link in passage_lookup:
                p = passage_lookup[link]
                if p["id"] not in seen_ids:
                    expansion.append(dict(p))
                    seen_ids.add(p["id"])
        if len(expansion) >= max_expansion:
            break

    # Combine: original candidates + expansion passages
    combined = candidates + expansion[:max_expansion]
    return combined[:max_total]
    combined = candidates + expansion
    return combined[:max_total]
