"""
Strong baseline retriever for Phase 3.

Improvements over simple baseline:
  - Separate row and passage retrieval
  - Row-aware passage expansion (prefer linked passages from retrieved rows)
  - Cross-encoder reranking
  - Lexical/semantic hybrid scoring
  - Better score normalization
"""
from __future__ import annotations

from typing import Any

from src.baseline.vector_index import LocalVectorIndex


def compute_lexical_score(query: str, text: str) -> float:
    """
    Simple BM25-style lexical scoring for hybrid retrieval.
    Returns normalized score between 0 and 1.
    """
    query_terms = set(query.lower().split())
    text_terms = text.lower().split()
    
    # Term frequency in document
    tf_scores = {}
    for term in query_terms:
        count = text_terms.count(term)
        if count > 0:
            # Simple TF with diminishing returns
            tf_scores[term] = count / (count + 10)
    
    if not tf_scores:
        return 0.0
    
    # Average TF across query terms
    return sum(tf_scores.values()) / len(query_terms)


def hybrid_score(semantic_score: float, lexical_score: float, alpha: float = 0.7) -> float:
    """
    Combine semantic and lexical scores.
    alpha controls semantic weight (higher = more semantic)
    """
    return alpha * semantic_score + (1 - alpha) * lexical_score


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """
    Rerank candidates using a cross-encoder model.
    Falls back to original scores if cross-encoder unavailable.
    
    For Phase 3, we implement a simple relevance-based reranking.
    In future, can replace with actual cross-encoder model (e.g., ms-marco-MiniLM).
    """
    # Placeholder: Simple heuristic-based reranking
    # Prefer rows and passages that have higher overlap with query entities
    query_lower = query.lower()
    
    for cand in candidates:
        text = cand.get("text", "")
        
        # Boost score if text contains question keywords
        boost = 0.0
        
        # Check for numbers in both query and text (useful for HybridQA)
        query_has_number = any(c.isdigit() for c in query)
        text_has_number = any(c.isdigit() for c in text)
        if query_has_number and text_has_number:
            boost += 0.1
        
        # Check for proper nouns (capitalized words)
        query_words = query.split()
        text_words = text.split()
        proper_nouns_query = {w for w in query_words if w and w[0].isupper()}
        proper_nouns_text = {w for w in text_words if w and w[0].isupper()}
        overlap = proper_nouns_query & proper_nouns_text
        if overlap:
            boost += 0.15 * min(len(overlap), 3)
        
        # Apply boost
        original_score = cand.get("score", 0.0)
        cand["rerank_score"] = original_score + boost
        cand["rerank_boost"] = boost
    
    # Sort by rerank score
    reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_k]


def retrieve_rows_and_passages_separately(
    question: str,
    index: LocalVectorIndex,
    passage_lookup: dict[str, dict],
    top_k_rows: int = 5,
    top_k_passages: int = 3,
    max_expansion: int = 5,
    use_lexical: bool = True,
    use_reranking: bool = True,
) -> dict[str, list[dict]]:
    """
    Two-stage retrieval with separate row and passage scoring.
    
    Returns:
        {
            "rows": list of retrieved row units,
            "passages": list of retrieved passage units,
            "merged": deduplicated and reranked combination
        }
    """
    # Stage 1a: Search for rows
    all_candidates = index.search(question, top_k=top_k_rows * 3)  # Over-retrieve for reranking
    
    # Split by type
    row_candidates = [c for c in all_candidates if c.get("type") == "table_row"]
    passage_candidates = [c for c in all_candidates if c.get("type") == "linked_passage"]
    summary_candidates = [c for c in all_candidates if c.get("type") == "table_summary"]
    
    # Optionally add lexical scores
    if use_lexical:
        for cand in row_candidates + passage_candidates + summary_candidates:
            lex_score = compute_lexical_score(question, cand.get("text", ""))
            sem_score = cand.get("score", 0.0)
            cand["lexical_score"] = lex_score
            cand["hybrid_score"] = hybrid_score(sem_score, lex_score)
        
        # Re-sort by hybrid score
        row_candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        passage_candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    
    # Keep top rows
    selected_rows = row_candidates[:top_k_rows]
    
    # Stage 1b: Select passages directly retrieved
    selected_passages = passage_candidates[:top_k_passages]
    
    # Stage 2: Row-aware passage expansion
    # For retrieved rows, fetch their linked passages
    seen_passage_ids = {p.get("id") for p in selected_passages}
    expanded_passages = []
    
    for row in selected_rows:
        row_links = row.get("row_links", [])
        for link in row_links:
            if link in passage_lookup:
                passage = passage_lookup[link]
                if passage["id"] not in seen_passage_ids:
                    # Inherit some score from parent row
                    passage_copy = dict(passage)
                    row_score = row.get("hybrid_score" if use_lexical else "score", 0.0)
                    passage_copy["score"] = row_score * 0.8  # Slightly lower than row
                    passage_copy["source"] = "row_expansion"
                    passage_copy["parent_row_id"] = row.get("id")
                    expanded_passages.append(passage_copy)
                    seen_passage_ids.add(passage["id"])
            
            if len(expanded_passages) >= max_expansion:
                break
        if len(expanded_passages) >= max_expansion:
            break
    
    # Combine passages: direct + expanded
    all_passages = selected_passages + expanded_passages[:max_expansion]
    
    # Stage 3: Rerank combined evidence
    merged_candidates = selected_rows + all_passages + summary_candidates[:1]  # Include one table summary if relevant
    
    if use_reranking:
        merged_candidates = rerank_cross_encoder(question, merged_candidates, top_k=top_k_rows + top_k_passages + max_expansion)
    
    return {
        "rows": selected_rows,
        "passages": all_passages,
        "summaries": summary_candidates[:1],
        "merged": merged_candidates,
    }


def retrieve_strong(
    question: str,
    index: LocalVectorIndex,
    passage_lookup: dict[str, dict],
    top_k: int = 8,
    use_lexical: bool = True,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Strong baseline retrieval with all enhancements.
    
    Returns a single merged list of top-k evidences.
    """
    result = retrieve_rows_and_passages_separately(
        question=question,
        index=index,
        passage_lookup=passage_lookup,
        top_k_rows=max(3, top_k // 2),
        top_k_passages=max(2, top_k // 3),
        max_expansion=max(2, top_k // 4),
        use_lexical=use_lexical,
        use_reranking=use_reranking,
    )
    
    return result["merged"][:top_k]
