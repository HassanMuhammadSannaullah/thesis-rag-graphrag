"""
Baseline corpus builder.

Takes parsed HybridQA records and creates retrieval units:
  - table_summary: one per table
  - table_row: one per row
  - linked_passage: one per linked passage

Each unit is a dict with: id, type, text, table_id, metadata
"""
import json
from pathlib import Path


def build_table_summary(record: dict) -> dict:
    """Create a table summary retrieval unit."""
    table = record["table"]
    headers = ", ".join(table["headers"])
    # Include first few rows as preview
    row_previews = []
    for row in table["rows"][:3]:
        vals = " | ".join(str(v) for v in row.values())
        row_previews.append(vals)
    preview = "\n".join(row_previews)

    text = (
        f"Table: {table['title']}\n"
        f"Section: {table.get('section_title', '')}\n"
        f"Columns: {headers}\n"
        f"Rows: {table['num_rows']}\n"
        f"Preview:\n{preview}"
    )
    if table.get("intro"):
        text += f"\nIntro: {table['intro'][:300]}"

    return {
        "id": f"summary_{record['table_id']}",
        "type": "table_summary",
        "text": text,
        "table_id": record["table_id"],
        "question_id": record["question_id"],
    }


def build_table_rows(record: dict) -> list[dict]:
    """Create one retrieval unit per table row."""
    table = record["table"]
    units = []
    for i, row in enumerate(table["rows"]):
        # Format: "Column: value" pairs (skip internal _links field)
        pairs = [f"{k}: {v}" for k, v in row.items() if k != "_links"]
        text = (
            f"Table: {table['title']} | Row {i+1}/{table['num_rows']}\n"
            + " | ".join(pairs)
        )
        row_links = row.get("_links", [])
        units.append({
            "id": f"row_{record['table_id']}_{i}",
            "type": "table_row",
            "text": text,
            "table_id": record["table_id"],
            "question_id": record["question_id"],
            "row_index": i,
            "row_links": row_links,
        })
    return units


def build_linked_passages(record: dict, max_passages: int = 30) -> list[dict]:
    """Create one retrieval unit per linked passage."""
    units = []
    for j, lp in enumerate(record["linked_passages"][:max_passages]):
        link = lp["link"]
        passage_text = lp["text"]
        if not passage_text.strip():
            continue
        # Truncate very long passages
        if len(passage_text) > 1000:
            passage_text = passage_text[:1000] + "..."

        entity_name = link.split("/")[-1].replace("_", " ")
        text = f"Entity: {entity_name}\n{passage_text}"

        units.append({
            "id": f"passage_{record['table_id']}_{j}",
            "type": "linked_passage",
            "text": text,
            "table_id": record["table_id"],
            "question_id": record["question_id"],
            "link": link,
        })
    return units


def build_corpus_for_record(record: dict, max_passages: int = 30) -> list[dict]:
    """Build all retrieval units for a single HybridQA record."""
    units = []
    units.append(build_table_summary(record))
    units.extend(build_table_rows(record))
    units.extend(build_linked_passages(record, max_passages))
    return units


def build_corpus(records: list[dict], max_passages: int = 30) -> list[dict]:
    """Build the full corpus from a list of parsed records."""
    corpus = []
    for rec in records:
        corpus.extend(build_corpus_for_record(rec, max_passages))
    return corpus


def save_corpus(corpus: list[dict], path: Path):
    """Save corpus to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for unit in corpus:
            f.write(json.dumps(unit, ensure_ascii=False) + "\n")
    print(f"Saved {len(corpus)} corpus units to {path}")


def load_corpus(path: Path) -> list[dict]:
    """Load corpus from JSONL."""
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    return corpus
