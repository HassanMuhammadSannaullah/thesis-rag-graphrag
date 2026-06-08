"""
Context processing utilities for strong baseline.

Functions for:
  - Deduplicating retrieved evidence
  - Merging overlapping contexts
  - Smart context packing within token limits
  - Provenance tracking
"""
from __future__ import annotations

from typing import Any


def deduplicate_contexts(contexts: list[dict]) -> list[dict]:
    """
    Remove duplicate contexts based on ID and text similarity.
    """
    seen_ids = set()
    seen_texts = set()
    deduplicated = []
    
    for ctx in contexts:
        ctx_id = ctx.get("id")
        ctx_text = ctx.get("text", "").strip()
        
        # Skip if we've seen this ID
        if ctx_id and ctx_id in seen_ids:
            continue
        
        # Skip if we've seen very similar text (exact match for now, could use fuzzy)
        text_signature = ctx_text[:200]  # Use first 200 chars as signature
        if text_signature in seen_texts:
            continue
        
        # Accept this context
        deduplicated.append(ctx)
        if ctx_id:
            seen_ids.add(ctx_id)
        if ctx_text:
            seen_texts.add(text_signature)
    
    return deduplicated


def group_by_type(contexts: list[dict]) -> dict[str, list[dict]]:
    """
    Group contexts by type for structured presentation.
    """
    groups = {}
    for ctx in contexts:
        ctx_type = ctx.get("type", "unknown")
        if ctx_type not in groups:
            groups[ctx_type] = []
        groups[ctx_type].append(ctx)
    return groups


def format_context_for_prompt(
    contexts: list[dict],
    max_chars: int = 12000,
    include_metadata: bool = True,
) -> str:
    """
    Format retrieved contexts into a structured prompt string.
    Respects character limits and maintains provenance.
    """
    # Deduplicate first
    contexts = deduplicate_contexts(contexts)
    
    # Group by type for better organization
    groups = group_by_type(contexts)
    
    # Build formatted sections
    sections = []
    total_chars = 0
    
    # Order: summaries, rows, passages
    type_order = ["table_summary", "table_row", "linked_passage"]
    
    for ctx_type in type_order:
        if ctx_type not in groups:
            continue
        
        type_label = {
            "table_summary": "Table Overview",
            "table_row": "Table Rows",
            "linked_passage": "Linked Entity Information",
        }.get(ctx_type, ctx_type)
        
        section_lines = [f"=== {type_label} ==="]
        
        for i, ctx in enumerate(groups[ctx_type], 1):
            text = ctx.get("text", "")
            
            # Add metadata header if requested
            if include_metadata:
                meta_parts = []
                if ctx.get("id"):
                    meta_parts.append(f"ID: {ctx['id']}")
                if ctx.get("score") is not None:
                    meta_parts.append(f"Score: {ctx['score']:.3f}")
                if ctx.get("table_id"):
                    meta_parts.append(f"Table: {ctx['table_id']}")
                
                if meta_parts:
                    header = f"[{i}] ({', '.join(meta_parts)})"
                    section_lines.append(header)
            
            section_lines.append(text)
            section_lines.append("")  # blank line between items
        
        section_text = "\n".join(section_lines)
        
        # Check if adding this section exceeds limit
        if total_chars + len(section_text) > max_chars:
            # Try to fit at least some of this section
            remaining = max_chars - total_chars
            if remaining > 500:  # Only add if we have reasonable space
                truncated = section_text[:remaining] + "\n... (truncated)"
                sections.append(truncated)
            break
        
        sections.append(section_text)
        total_chars += len(section_text)
    
    return "\n\n".join(sections)


def pack_contexts_within_limit(
    contexts: list[dict],
    max_tokens: int = 4000,
    chars_per_token: float = 3.5,
) -> list[dict]:
    """
    Pack as many contexts as possible within token limit.
    Prioritizes by score.
    """
    max_chars = int(max_tokens * chars_per_token)
    
    # Sort by score (highest first)
    sorted_contexts = sorted(
        contexts,
        key=lambda x: x.get("rerank_score", x.get("hybrid_score", x.get("score", 0.0))),
        reverse=True,
    )
    
    packed = []
    total_chars = 0
    
    for ctx in sorted_contexts:
        ctx_chars = len(ctx.get("text", ""))
        if total_chars + ctx_chars <= max_chars:
            packed.append(ctx)
            total_chars += ctx_chars
        elif len(packed) == 0:
            # Ensure we include at least one context, even if truncated
            truncated_ctx = dict(ctx)
            truncated_ctx["text"] = ctx["text"][:max_chars] + "... (truncated)"
            packed.append(truncated_ctx)
            break
    
    return packed


def extract_provenance(contexts: list[dict]) -> dict[str, Any]:
    """
    Extract provenance information from retrieved contexts.
    """
    provenance = {
        "context_ids": [c.get("id") for c in contexts if c.get("id")],
        "table_ids": list({c.get("table_id") for c in contexts if c.get("table_id")}),
        "context_types": [c.get("type") for c in contexts],
        "num_rows": sum(1 for c in contexts if c.get("type") == "table_row"),
        "num_passages": sum(1 for c in contexts if c.get("type") == "linked_passage"),
        "num_summaries": sum(1 for c in contexts if c.get("type") == "table_summary"),
        "scores": [c.get("score", 0.0) for c in contexts],
        "avg_score": sum(c.get("score", 0.0) for c in contexts) / len(contexts) if contexts else 0.0,
    }
    
    return provenance


def create_citation_map(contexts: list[dict]) -> dict[str, str]:
    """
    Create a map from context IDs to short citation labels.
    Useful for answer generation with citations.
    """
    citation_map = {}
    
    # Group by type
    groups = group_by_type(contexts)
    
    # Assign labels
    for ctx_type, items in groups.items():
        prefix = {
            "table_summary": "TS",
            "table_row": "R",
            "linked_passage": "P",
        }.get(ctx_type, "C")
        
        for i, ctx in enumerate(items, 1):
            ctx_id = ctx.get("id")
            if ctx_id:
                citation_map[ctx_id] = f"{prefix}{i}"
    
    return citation_map
