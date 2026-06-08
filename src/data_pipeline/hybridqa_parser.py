"""HybridQA parsing helpers with stable provenance metadata."""
from __future__ import annotations

import json
import zipfile
from typing import Any

from src.data_pipeline.hybridqa_evidence import attach_proxy_evidence

ZIP_PREFIX = "WikiTables-WithLinks-master"
PARSER_SCHEMA_VERSION = "hybridqa_v2"


def load_json_from_zip(zf: zipfile.ZipFile, path_in_zip: str):
    try:
        with zf.open(path_in_zip) as handle:
            return json.load(handle)
    except KeyError:
        return None


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _row_id(table_id: str, row_index: int) -> str:
    return f"row::{table_id}::{row_index}"


def _cell_id(table_id: str, row_index: int, column_index: int) -> str:
    return f"cell::{table_id}::{row_index}::{column_index}"


def _passage_id(table_id: str, link_rank: int) -> str:
    return f"passage::{table_id}::{link_rank}"


def parse_table(table_json: dict[str, Any], table_id: str) -> dict[str, Any]:
    """Parse a HybridQA table while preserving row and cell provenance."""
    headers = [cell[0] for cell in table_json.get("header", [])]
    rows: list[dict[str, Any]] = []
    row_metadata: list[dict[str, Any]] = []
    cell_metadata: list[dict[str, Any]] = []
    all_links: list[str] = []

    for row_index, row_data in enumerate(table_json.get("data", [])):
        row: dict[str, Any] = {}
        row_links: list[str] = []
        row_id = _row_id(table_id, row_index)
        row_cell_ids: list[str] = []

        for column_index, cell in enumerate(row_data):
            column_name = headers[column_index] if column_index < len(headers) else f"col_{column_index}"
            text = cell[0] if isinstance(cell, list) else str(cell)
            links = cell[1] if isinstance(cell, list) and len(cell) > 1 else []
            ordered_links = _dedupe_preserve_order(list(links))

            row[column_name] = text
            row_links.extend(ordered_links)
            all_links.extend(ordered_links)

            cell_id = _cell_id(table_id, row_index, column_index)
            row_cell_ids.append(cell_id)
            cell_metadata.append(
                {
                    "cell_id": cell_id,
                    "row_id": row_id,
                    "row_index": row_index,
                    "column_index": column_index,
                    "column_name": column_name,
                    "text": text,
                    "links": ordered_links,
                }
            )

        ordered_row_links = _dedupe_preserve_order(row_links)
        row["_links"] = ordered_row_links
        rows.append(row)
        row_metadata.append(
            {
                "row_id": row_id,
                "row_index": row_index,
                "cell_ids": row_cell_ids,
                "row_links": ordered_row_links,
                "row_link_count": len(ordered_row_links),
            }
        )

    ordered_all_links = _dedupe_preserve_order(all_links)
    return {
        "table_id": table_id,
        "title": table_json.get("title", ""),
        "section_title": table_json.get("section_title", ""),
        "section_text": table_json.get("section_text", ""),
        "intro": table_json.get("intro", ""),
        "headers": headers,
        "rows": rows,
        "num_rows": len(rows),
        "all_links": ordered_all_links,
        "row_metadata": row_metadata,
        "cell_metadata": cell_metadata,
    }


def build_linked_passages(table_id: str, all_links: list[str], passages_json: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return all available linked passages plus a full link inventory."""
    if not passages_json:
        return [], []

    entity_links: list[str] = []
    generic_links: list[str] = []
    for link in all_links:
        name = link.split("/")[-1]
        if any(name.startswith(str(year)) for year in range(1800, 2100)):
            generic_links.append(link)
        else:
            entity_links.append(link)
    ordered_links = entity_links + generic_links

    linked_passages: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    valid_rank = 0

    for original_rank, link in enumerate(ordered_links):
        text = (passages_json.get(link) or "").strip()
        link_group = "generic" if link in generic_links else "entity"
        entity_name = link.split("/")[-1].replace("_", " ")
        passage_identifier = _passage_id(table_id, valid_rank) if text else None
        inventory.append(
            {
                "link": link,
                "entity_name": entity_name,
                "link_group": link_group,
                "original_rank": original_rank,
                "has_text": bool(text),
                "passage_id": passage_identifier,
            }
        )
        if not text:
            continue
        linked_passages.append(
            {
                "passage_id": passage_identifier,
                "link": link,
                "entity_name": entity_name,
                "text": text,
                "link_group": link_group,
                "link_rank": valid_rank,
                "original_rank": original_rank,
                "text_char_count": len(text),
                "source": "request_tok",
            }
        )
        valid_rank += 1

    return linked_passages, inventory


def attach_passage_provenance(table: dict[str, Any], linked_passages: list[dict[str, Any]]) -> None:
    """Attach passage-id mappings to row and cell metadata in-place."""
    link_to_passage_id = {
        passage["link"]: passage["passage_id"]
        for passage in linked_passages
        if passage.get("passage_id")
    }

    for row_meta in table.get("row_metadata", []):
        row_meta["row_link_passage_ids"] = [
            link_to_passage_id[link]
            for link in row_meta.get("row_links", [])
            if link in link_to_passage_id
        ]

    for cell_meta in table.get("cell_metadata", []):
        cell_meta["linked_passage_ids"] = [
            link_to_passage_id[link]
            for link in cell_meta.get("links", [])
            if link in link_to_passage_id
        ]


def build_hybridqa_record(
    *,
    question_payload: dict[str, Any],
    table_json: dict[str, Any],
    passages_json: dict[str, str] | None,
    split: str,
) -> dict[str, Any]:
    """Build one parsed HybridQA record with stable provenance metadata."""
    table_id = str(question_payload["table_id"])
    table = parse_table(table_json, table_id)
    linked_passages, link_inventory = build_linked_passages(table_id, table["all_links"], passages_json or {})
    attach_passage_provenance(table, linked_passages)

    record = {
        "question_id": question_payload["question_id"],
        "question": question_payload["question"],
        "answer": question_payload["answer-text"],
        "table_id": table_id,
        "table": table,
        "linked_passages": linked_passages,
        "num_linked_passages": len(linked_passages),
        "linked_passage_ids": [passage["passage_id"] for passage in linked_passages],
        "linked_passage_inventory": link_inventory,
        "split": split,
        "parser_metadata": {
            "schema_version": PARSER_SCHEMA_VERSION,
            "linked_passage_strategy": "all_table_links_with_text",
            "all_link_count": len(table["all_links"]),
            "linked_passage_count": len(linked_passages),
        },
        "source_metadata": {
            "table_path": f"{ZIP_PREFIX}/tables_tok/{table_id}.json",
            "request_path": f"{ZIP_PREFIX}/request_tok/{table_id}.json",
        },
    }
    return attach_proxy_evidence(record)


def load_hybridqa_data(split: str = "dev", limit: int = None) -> list[dict]:
    """
    Load parsed HybridQA records from disk.
    
    Args:
        split: Dataset split ("dev" or "train")
        limit: Optional limit on number of records to load
    
    Returns:
        List of parsed HybridQA record dicts
    
    Raises:
        FileNotFoundError: If parsed data doesn't exist
    """
    from pathlib import Path
    from src.config import settings as cfg
    
    parsed_path = Path(cfg.ORIGINAL_DIR) / f"{split}.jsonl"
    
    if not parsed_path.exists():
        raise FileNotFoundError(
            f"Parsed data not found: {parsed_path}\n"
            f"Run `python scripts/02_parse_hybridqa.py --split {split}` first."
        )
    
    records = []
    with open(parsed_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # Backfill proxy evidence if not present
                if "proxy_evidence" not in rec:
                    attach_proxy_evidence(rec)
                records.append(rec)
    
    if limit:
        records = records[:limit]
    
    return records
