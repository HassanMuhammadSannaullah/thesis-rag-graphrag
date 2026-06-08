"""
Strong baseline answer generator with provenance tracking.

Improvements over simple answer generator:
  - Better prompt structure distinguishing table vs passage evidence
  - Provenance tracking (which specific evidence led to answer)
  - Citation support (optional)
  - Confidence scoring
"""
from __future__ import annotations

from typing import Any

from src.baseline.context_utils import (
    create_citation_map,
    deduplicate_contexts,
    extract_provenance,
    format_context_for_prompt,
    pack_contexts_within_limit,
)
from src.utils.model_client import generate_text


STRONG_RAG_PROMPT = """You are answering questions using provided evidence from tables and linked entity information (Wikipedia passages).

=== Instructions ===

1. Use ONLY facts present in the evidence below.
2. HybridQA questions often require combining:
   - Information from a TABLE ROW (e.g., a person's name, nationality)
   - Information from a LINKED ENTITY PASSAGE (e.g., background about that person)
3. Return the SHORTEST EXACT ANSWER possible.
4. Do NOT add explanations, citations, qualifications, or extra text.
5. If the answer is a single word, name, number, date, or nationality → return ONLY that.
6. If you cannot find a confident answer in the evidence → return: "INSUFFICIENT EVIDENCE"

=== Evidence ===

{context}

=== Question ===

{question}

=== Answer ===
"""


STRONG_RAG_WITH_CITATIONS_PROMPT = """You are answering questions using provided evidence from tables and linked entity information (Wikipedia passages).

=== Instructions ===

1. Use ONLY facts present in the evidence below.
2. HybridQA questions often require combining:
   - Information from a TABLE ROW (e.g., a person's name, nationality)
   - Information from a LINKED ENTITY PASSAGE (e.g., background about that person)
3. Provide:
   - Final answer (shortest exact answer possible)
   - Supporting evidence IDs from the context used to derive the answer
4. Format your response as:
   ANSWER: <your answer>
   EVIDENCE: <comma-separated evidence IDs, e.g., R1, P2>

=== Evidence ===

{context}

=== Question ===

{question}

=== Response ===
"""


def parse_answer_with_citations(response: str) -> dict[str, Any]:
    """
    Parse a response that includes both answer and evidence citations.
    
    Expected format:
        ANSWER: Jerry
        EVIDENCE: R2, P1
    """
    lines = response.strip().split("\n")
    answer = ""
    evidence_ids = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("ANSWER:"):
            answer = line.replace("ANSWER:", "").strip()
        elif line.startswith("EVIDENCE:"):
            evidence_str = line.replace("EVIDENCE:", "").strip()
            evidence_ids = [e.strip() for e in evidence_str.split(",") if e.strip()]
    
    # Fallback: if parsing failed, treat entire response as answer
    if not answer:
        answer = response.strip()
    
    return {
        "answer": answer,
        "cited_evidence": evidence_ids,
    }


def generate_answer_strong(
    question: str,
    evidence_units: list[dict],
    max_context_chars: int = 12000,
    max_answer_tokens: int = 256,
    include_citations: bool = False,
) -> dict[str, Any]:
    """
    Generate answer using strong baseline approach.
    
    Returns:
        {
            "answer": str,
            "provenance": dict with evidence tracking,
            "cited_evidence": list of IDs (if include_citations=True),
            "num_evidence_units": int,
            "context_summary": dict,
        }
    """
    # Deduplicate and pack contexts
    deduplicated = deduplicate_contexts(evidence_units)
    packed = pack_contexts_within_limit(
        deduplicated,
        max_tokens=max_context_chars // 4,  # Rough token estimate
        chars_per_token=4.0,
    )
    
    # Format context with metadata
    context_text = format_context_for_prompt(
        packed,
        max_chars=max_context_chars,
        include_metadata=include_citations,  # Add IDs if we want citations
    )
    
    # Extract provenance
    provenance = extract_provenance(packed)
    
    # Create citation map if needed
    citation_map = create_citation_map(packed) if include_citations else {}
    
    # Select prompt
    if include_citations:
        prompt = STRONG_RAG_WITH_CITATIONS_PROMPT.format(
            context=context_text,
            question=question,
        )
    else:
        prompt = STRONG_RAG_PROMPT.format(
            context=context_text,
            question=question,
        )
    
    # Generate answer
    response = generate_text(prompt, max_tokens=max_answer_tokens)
    
    # Parse response
    if include_citations:
        parsed = parse_answer_with_citations(response)
        answer = parsed["answer"]
        cited_evidence = parsed["cited_evidence"]
    else:
        answer = response.strip()
        cited_evidence = []
    
    # Build result
    result = {
        "question": question,
        "generated_answer": answer,
        "provenance": provenance,
        "num_evidence_units": len(packed),
        "num_original_units": len(evidence_units),
        "context_summary": {
            "num_rows": provenance["num_rows"],
            "num_passages": provenance["num_passages"],
            "num_summaries": provenance["num_summaries"],
            "avg_score": provenance["avg_score"],
        },
    }
    
    if include_citations:
        result["cited_evidence"] = cited_evidence
        result["citation_map"] = citation_map
    
    return result


def generate_answer_simple_baseline(
    question: str,
    evidence_units: list[dict],
    max_answer_tokens: int = 256,
) -> dict[str, Any]:
    """
    Simple baseline for ablation comparison.
    No deduplication, no fancy packing, minimal prompt.
    """
    # Just concatenate all evidence
    evidence_texts = []
    for i, unit in enumerate(evidence_units):
        prefix = f"[{unit.get('type', 'unknown')}]"
        evidence_texts.append(f"{prefix} {unit['text']}")
    
    evidence_str = "\n\n".join(evidence_texts)
    
    # Truncate if too long
    if len(evidence_str) > 6000:
        evidence_str = evidence_str[:6000] + "\n... (truncated)"
    
    prompt = f"""Answer the question using only the provided evidence. Return the shortest exact answer.

Evidence:
{evidence_str}

Question: {question}

Answer:"""
    
    answer = generate_text(prompt, max_tokens=max_answer_tokens)
    
    return {
        "question": question,
        "generated_answer": answer.strip(),
        "num_evidence_units": len(evidence_units),
        "baseline_variant": "simple",
    }
