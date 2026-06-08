"""
Improved GraphRAG source construction for Phase 4.

Multiple document construction strategies with better provenance and structure:
  1. row_centric: One document per table row with its linked passages
  2. table_centric: One document per table with all rows and passages
  3. entity_centric: Separate documents for entities/passages with table context
  4. hybrid: Mix of table overview + detailed row documents

All strategies preserve:
  - Stable IDs for traceability
  - Row-to-passage linkage
  - Minimal truncation
  - Structured metadata
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _get_row_id(record: dict, row_index: int) -> str:
    """Get stable row ID from record metadata."""
    table = record["table"]
    row_metadata = table.get("row_metadata", [])
    for rm in row_metadata:
        if rm.get("row_index") == row_index:
            return rm.get("row_id", f"row::{record['table_id']}::{row_index}")
    return f"row::{record['table_id']}::{row_index}"


def _get_passage_id(passage: dict, fallback_id: str) -> str:
    """Get stable passage ID."""
    return passage.get("passage_id", fallback_id)


def _format_row_content(row: dict, row_index: int, table_title: str) -> str:
    """Format a table row into natural language."""
    row_data = {k: v for k, v in row.items() if k != "_links"}
    if not row_data:
        return f"Row {row_index + 1} of {table_title}: (empty)"
    
    # Create natural language description
    items = [f"{k}: {v}" for k, v in row_data.items()]
    return f"Row {row_index + 1} of {table_title}: " + ", ".join(items)


def _format_passage_content(passage: dict, max_chars: int = 1500) -> str:
    """Format a linked passage with entity context."""
    link = passage.get("link", "")
    entity_name = link.split("/")[-1].replace("_", " ") if link else "Unknown Entity"
    text = passage.get("text", "")
    
    # Less aggressive truncation
    if len(text) > max_chars:
        text = text[:max_chars] + " [...]"
    
    return f"Entity: {entity_name}\n{text}"


# ==============================================================================
# Strategy 1: Row-Centric Documents
# ==============================================================================

def build_row_centric_document(
    record: dict,
    row_index: int,
    include_table_context: bool = True,
) -> dict[str, Any]:
    """
    Build one document per table row with its linked passages.
    
    Best for: Preserving row-level provenance and row-passage relationships.
    
    Returns:
        {
            "id": str,
            "title": str,
            "text": str,
            "metadata": dict,
        }
    """
    table = record["table"]
    row = table["rows"][row_index]
    row_id = _get_row_id(record, row_index)
    
    parts = []
    
    # Metadata header (parseable by downstream tools)
    parts.append(f"[DOCUMENT_ID: {row_id}]")
    parts.append(f"[TABLE_ID: {record['table_id']}]")
    parts.append(f"[ROW_INDEX: {row_index}]")
    parts.append("")
    
    # Document title
    parts.append(f"# {table['title']} - Row {row_index + 1}")
    
    # Optional: Brief table context
    if include_table_context and table.get("intro"):
        intro = table["intro"][:300] if len(table["intro"]) > 300 else table["intro"]
        parts.append(f"\n## Table Context")
        parts.append(intro)
    
    # Row content
    parts.append(f"\n## Row Data")
    row_text = _format_row_content(row, row_index, table['title'])
    parts.append(row_text)
    
    # Linked passages for this row
    row_links = row.get("_links", [])
    if row_links:
        parts.append(f"\n## Related Entities (from this row)")
        
        # Find passages matching this row's links
        linked_passages = record.get("linked_passages", [])
        for passage in linked_passages:
            if passage.get("link") in row_links:
                passage_id = _get_passage_id(passage, f"passage::{record['table_id']}::unknown")
                parts.append(f"\n[PASSAGE_ID: {passage_id}]")
                parts.append(_format_passage_content(passage, max_chars=1500))
    
    doc_text = "\n".join(parts)
    
    return {
        "id": row_id,
        "title": f"{table['title']} - Row {row_index + 1}",
        "text": doc_text,
        "metadata": {
            "table_id": record["table_id"],
            "question_id": record.get("question_id"),
            "row_index": row_index,
            "num_linked_passages": len(row_links),
            "strategy": "row_centric",
        },
    }


def build_row_centric_corpus(
    records: list[dict],
    include_table_context: bool = True,
) -> list[dict[str, Any]]:
    """Build row-centric documents for all records."""
    documents = []
    for record in records:
        table = record["table"]
        for row_idx in range(len(table["rows"])):
            doc = build_row_centric_document(
                record,
                row_idx,
                include_table_context=include_table_context,
            )
            documents.append(doc)
    return documents


# ==============================================================================
# Strategy 2: Table-Centric Documents
# ==============================================================================

def build_table_centric_document(
    record: dict,
    max_passages: int = 30,
) -> dict[str, Any]:
    """
    Build one comprehensive document per table with all rows and passages.
    
    Best for: Maintaining table-level relationships and full context.
    
    Returns:
        {
            "id": str,
            "title": str,
            "text": str,
            "metadata": dict,
        }
    """
    table = record["table"]
    table_id = record["table_id"]
    
    parts = []
    
    # Metadata header
    parts.append(f"[DOCUMENT_ID: {table_id}]")
    parts.append(f"[TABLE_ID: {table_id}]")
    parts.append("")
    
    # Table header
    parts.append(f"# {table['title']}")
    if table.get("section_title"):
        parts.append(f"Section: {table['section_title']}")
    
    if table.get("intro"):
        parts.append(f"\n## Introduction")
        parts.append(table["intro"])
    
    # Table structure info
    parts.append(f"\n## Table Structure")
    parts.append(f"Columns: {', '.join(table['headers'])}")
    parts.append(f"Number of rows: {table['num_rows']}")
    
    # All rows
    parts.append(f"\n## Table Rows")
    for row_idx, row in enumerate(table["rows"]):
        row_id = _get_row_id(record, row_idx)
        parts.append(f"\n[ROW_ID: {row_id}]")
        parts.append(_format_row_content(row, row_idx, table['title']))
    
    # All linked passages
    all_passages = record.get("linked_passages", [])
    linked_passages = all_passages if max_passages is None else all_passages[:max_passages]
    if linked_passages:
        parts.append(f"\n## Related Entities and Context")
        for passage in linked_passages:
            passage_id = _get_passage_id(passage, f"passage::{table_id}::unknown")
            parts.append(f"\n[PASSAGE_ID: {passage_id}]")
            parts.append(_format_passage_content(passage, max_chars=2000))
    
    doc_text = "\n".join(parts)
    
    return {
        "id": table_id,
        "title": table['title'],
        "text": doc_text,
        "metadata": {
            "table_id": table_id,
            "question_id": record.get("question_id"),
            "num_rows": table['num_rows'],
            "num_passages": len(linked_passages),
            "strategy": "table_centric",
        },
    }


def build_table_centric_corpus(
    records: list[dict],
    max_passages: int = None,
) -> list[dict[str, Any]]:
    """Build table-centric documents for all records."""
    return [build_table_centric_document(rec, max_passages) for rec in records]


# ==============================================================================
# Strategy 3: Entity-Centric Documents
# ==============================================================================

def build_entity_centric_documents(
    record: dict,
) -> list[dict[str, Any]]:
    """
    Build separate documents for each entity/passage, with table context.
    
    Best for: Entity-focused graph construction and entity relationship extraction.
    
    Returns:
        List of documents, one per entity/passage.
    """
    documents = []
    table = record["table"]
    table_id = record["table_id"]
    
    linked_passages = record.get("linked_passages", [])
    for passage in linked_passages:
        passage_id = _get_passage_id(passage, f"passage::{table_id}::unknown")
        link = passage.get("link", "")
        entity_name = link.split("/")[-1].replace("_", " ") if link else "Unknown Entity"
        
        parts = []
        
        # Metadata header
        parts.append(f"[DOCUMENT_ID: {passage_id}]")
        parts.append(f"[PASSAGE_ID: {passage_id}]")
        parts.append(f"[TABLE_ID: {table_id}]")
        parts.append(f"[ENTITY_LINK: {link}]")
        parts.append("")
        
        # Entity header
        parts.append(f"# Entity: {entity_name}")
        
        # Entity content
        parts.append(f"\n## Wikipedia Article")
        parts.append(passage.get("text", ""))
        
        # Table context: Which rows reference this entity?
        parts.append(f"\n## Appears in Table: {table['title']}")
        parts.append(f"This entity is referenced in the following rows:")
        
        for row_idx, row in enumerate(table["rows"]):
            if link in row.get("_links", []):
                row_id = _get_row_id(record, row_idx)
                parts.append(f"\n[ROW_ID: {row_id}]")
                parts.append(_format_row_content(row, row_idx, table['title']))
        
        doc_text = "\n".join(parts)
        
        documents.append({
            "id": passage_id,
            "title": f"Entity: {entity_name}",
            "text": doc_text,
            "metadata": {
                "table_id": table_id,
                "question_id": record.get("question_id"),
                "passage_id": passage_id,
                "entity_link": link,
                "strategy": "entity_centric",
            },
        })
    
    return documents


def build_entity_centric_corpus(
    records: list[dict],
) -> list[dict[str, Any]]:
    """Build entity-centric documents for all records."""
    documents = []
    for record in records:
        documents.extend(build_entity_centric_documents(record))
    return documents


# ==============================================================================
# Strategy 4: Hybrid (Table Overview + Detailed Rows)
# ==============================================================================

def build_hybrid_documents(
    record: dict,
    max_passages_in_overview: int = None,
) -> list[dict[str, Any]]:
    """
    Build hybrid corpus: one table overview + one doc per row.
    
    Best for: Balancing detail with overview, flexible retrieval.
    
    Returns:
        List of documents (1 overview + N rows).
    """
    documents = []
    table = record["table"]
    table_id = record["table_id"]
    
    # 1. Table overview document (no full row details)
    overview_parts = []
    overview_parts.append(f"[DOCUMENT_ID: {table_id}_overview]")
    overview_parts.append(f"[TABLE_ID: {table_id}]")
    overview_parts.append("")
    overview_parts.append(f"# {table['title']} - Overview")
    
    if table.get("section_title"):
        overview_parts.append(f"Section: {table['section_title']}")
    
    if table.get("intro"):
        overview_parts.append(f"\n## Introduction")
        overview_parts.append(table["intro"])
    
    overview_parts.append(f"\n## Structure")
    overview_parts.append(f"Columns: {', '.join(table['headers'])}")
    overview_parts.append(f"Number of rows: {table['num_rows']}")
    overview_parts.append(f"Related entities: {len(record.get('linked_passages', []))}")
    
    # Sample passages in overview
    linked_passages = record.get("linked_passages", [])[:max_passages_in_overview]
    if linked_passages:
        overview_parts.append(f"\n## Sample Related Entities")
        for passage in linked_passages:
            link = passage.get("link", "")
            entity_name = link.split("/")[-1].replace("_", " ") if link else "Unknown"
            overview_parts.append(f"- {entity_name}")
    
    overview_text = "\n".join(overview_parts)
    
    documents.append({
        "id": f"{table_id}_overview",
        "title": f"{table['title']} - Overview",
        "text": overview_text,
        "metadata": {
            "table_id": table_id,
            "question_id": record.get("question_id"),
            "doc_type": "overview",
            "strategy": "hybrid",
        },
    })
    
    # 2. Detailed row documents (same as row-centric)
    for row_idx in range(len(table["rows"])):
        row_doc = build_row_centric_document(
            record,
            row_idx,
            include_table_context=False,  # Already in overview
        )
        row_doc["metadata"]["doc_type"] = "row_detail"
        row_doc["metadata"]["strategy"] = "hybrid"
        documents.append(row_doc)
    
    return documents


def build_hybrid_corpus(
    records: list[dict],
    max_passages_in_overview: int = None,
) -> list[dict[str, Any]]:
    """Build hybrid corpus for all records."""
    documents = []
    for record in records:
        documents.extend(build_hybrid_documents(record, max_passages_in_overview))
    return documents


# ==============================================================================
# Corpus Building and Saving
# ==============================================================================

def build_graphrag_corpus(
    records: list[dict],
    strategy: str = "row_centric",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Build GraphRAG corpus with specified strategy.
    
    Args:
        records: Parsed HybridQA records
        strategy: One of ["row_centric", "table_centric", "entity_centric", "hybrid"]
        **kwargs: Strategy-specific parameters
    
    Returns:
        List of document dicts with id, title, text, metadata
    """
    if strategy == "row_centric":
        return build_row_centric_corpus(records, **kwargs)
    elif strategy == "table_centric":
        return build_table_centric_corpus(records, **kwargs)
    elif strategy == "entity_centric":
        return build_entity_centric_corpus(records, **kwargs)
    elif strategy == "hybrid":
        return build_hybrid_corpus(records, **kwargs)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: row_centric, table_centric, entity_centric, hybrid"
        )


def save_graphrag_corpus(
    documents: list[dict[str, Any]],
    output_dir: Path,
    format: str = "txt",
) -> None:
    """
    Save GraphRAG corpus to disk.
    
    Args:
        documents: List of document dicts
        output_dir: Output directory
        format: "txt" (one file per doc) or "jsonl" (all docs in one file)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "txt":
        # One .txt file per document
        for doc in documents:
            filename = f"{doc['id'].replace('/', '_').replace('::', '_')}.txt"
            filepath = output_dir / filename
            filepath.write_text(doc["text"], encoding="utf-8")
        
        print(f"Saved {len(documents)} documents as .txt files to {output_dir}")
    
    elif format == "jsonl":
        # All documents in one .jsonl file
        filepath = output_dir / "corpus.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(documents)} documents to {filepath}")
    
    else:
        raise ValueError(f"Unknown format: {format}. Choose from: txt, jsonl")
    
    # Also save a metadata summary
    summary = {
        "num_documents": len(documents),
        "strategies": list({doc["metadata"].get("strategy") for doc in documents}),
        "document_types": list({doc["metadata"].get("doc_type", "default") for doc in documents}),
    }
    
    summary_path = output_dir / "corpus_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Saved corpus summary to {summary_path}")


def compare_strategies(
    records: list[dict],
    sample_size: int = 3,
) -> dict[str, dict]:
    """
    Compare all strategies on a sample of records.
    
    Returns stats for each strategy:
        {
            "row_centric": {"num_docs": int, "avg_length": float, ...},
            ...
        }
    """
    sample_records = records[:sample_size]
    strategies = ["row_centric", "table_centric", "entity_centric", "hybrid"]
    
    comparison = {}
    
    for strategy in strategies:
        docs = build_graphrag_corpus(sample_records, strategy=strategy)
        
        comparison[strategy] = {
            "num_documents": len(docs),
            "avg_doc_length_chars": sum(len(d["text"]) for d in docs) / len(docs) if docs else 0,
            "total_chars": sum(len(d["text"]) for d in docs),
            "sample_doc_ids": [d["id"] for d in docs[:5]],
        }
    
    return comparison


# ==============================================================================
# Convenience Functions
# ==============================================================================

def hybridqa_to_improved_graphrag_docs(
    records: list[dict],
    output_dir: Path,
    strategy: str = "row_centric",
    format: str = "txt",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    One-stop function: build and save GraphRAG corpus.
    
    Args:
        records: Parsed HybridQA records
        output_dir: Where to save documents
        strategy: Document construction strategy
        format: Output format ("txt" or "jsonl")
        **kwargs: Strategy-specific parameters
    
    Returns:
        List of document dicts
    """
    print(f"\nBuilding GraphRAG corpus with strategy: {strategy}")
    print(f"Input: {len(records)} HybridQA records")
    
    documents = build_graphrag_corpus(records, strategy=strategy, **kwargs)
    
    print(f"Generated: {len(documents)} documents")
    print(f"Avg length: {sum(len(d['text']) for d in documents) / len(documents):.0f} chars")
    
    save_graphrag_corpus(documents, output_dir, format=format)
    
    return documents
