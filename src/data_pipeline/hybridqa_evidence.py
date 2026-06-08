"""HybridQA evidence-alignment helpers.

HybridQA raw files in this repository provide final answers but do not expose
official evidence ids. These helpers build a documented proxy-evidence view
that stays separate from strict gold-evidence labels.
"""
from __future__ import annotations

from typing import Any

from src.evaluation.normalization import canonical_text_variants, normalize_answer_text, normalize_tokens


PROXY_EVIDENCE_LABEL_MODE = "proxy_answer_anchored_v1"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_ORDINAL_ROW_HINTS = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
}


def _answer_variants(answer: str) -> list[str]:
    return [variant for variant in canonical_text_variants([answer]) if variant]


def _row_id(table_id: str, row_index: int) -> str:
    return f"row::{table_id}::{row_index}"


def _cell_id(table_id: str, row_index: int, column_index: int) -> str:
    return f"cell::{table_id}::{row_index}::{column_index}"


def _contains_answer(text: str, answer_variants: list[str]) -> bool:
    normalized_text = normalize_answer_text(text or "")
    if not normalized_text:
        return False
    for variant in answer_variants:
        normalized_variant = normalize_answer_text(variant)
        if not normalized_variant:
            continue
        if normalized_variant == normalized_text or normalized_variant in normalized_text:
            return True
    return False


def _question_tokens(question: str) -> set[str]:
    return {
        token
        for token in normalize_tokens(question)
        if token and token not in _STOPWORDS and len(token) > 1
    }


def _ordinal_row_hint(question: str) -> int | None:
    question_norm = normalize_answer_text(question)
    for token, row_index in _ORDINAL_ROW_HINTS.items():
        if token in question_norm.split():
            return row_index
    if "last" in question_norm.split():
        return -1
    return None


def _ensure_table_provenance(table: dict[str, Any], table_id: str) -> dict[str, Any]:
    if table.get("row_metadata") and table.get("cell_metadata"):
        return table

    headers = list(table.get("headers") or [])
    row_metadata: list[dict[str, Any]] = []
    cell_metadata: list[dict[str, Any]] = []
    all_links: list[str] = []
    for row_index, row in enumerate(table.get("rows", [])):
        row_id = _row_id(table_id, row_index)
        row_links = list(dict.fromkeys(row.get("_links", [])))
        all_links.extend(row_links)
        cell_ids: list[str] = []
        for column_index, column_name in enumerate(headers):
            cell_id = _cell_id(table_id, row_index, column_index)
            cell_ids.append(cell_id)
            cell_metadata.append(
                {
                    "cell_id": cell_id,
                    "row_id": row_id,
                    "row_index": row_index,
                    "column_index": column_index,
                    "column_name": column_name,
                    "text": str(row.get(column_name, "")),
                    "links": [],
                    "linked_passage_ids": [],
                }
            )
        row_metadata.append(
            {
                "row_id": row_id,
                "row_index": row_index,
                "cell_ids": cell_ids,
                "row_links": row_links,
                "row_link_count": len(row_links),
                "row_link_passage_ids": [],
            }
        )

    table["row_metadata"] = row_metadata
    table["cell_metadata"] = cell_metadata
    table["all_links"] = list(dict.fromkeys(all_links or table.get("all_links", [])))
    table.setdefault("num_rows", len(table.get("rows", [])))
    return table


def build_proxy_evidence(
    *,
    question: str,
    answer: str,
    table: dict[str, Any],
    linked_passages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build parser-level proxy evidence aligned to row and passage ids.

    The output is intentionally marked as proxy evidence because the local
    HybridQA raw files do not include official gold evidence ids.
    """
    answer_variants = _answer_variants(answer)
    question_tokens = _question_tokens(question)
    ordinal_hint = _ordinal_row_hint(question)

    link_to_passage = {
        passage["link"]: passage["passage_id"]
        for passage in linked_passages
        if passage.get("passage_id")
    }
    passage_by_id = {
        passage["passage_id"]: passage
        for passage in linked_passages
        if passage.get("passage_id")
    }

    answer_cell_ids: list[str] = []
    answer_row_ids: list[str] = []
    row_id_to_cells: dict[str, list[str]] = {}
    for cell_meta in table.get("cell_metadata", []):
        row_id_to_cells.setdefault(cell_meta["row_id"], []).append(cell_meta["cell_id"])
        if _contains_answer(cell_meta.get("text", ""), answer_variants):
            answer_cell_ids.append(cell_meta["cell_id"])
            answer_row_ids.append(cell_meta["row_id"])

    answer_passage_ids: list[str] = []
    for passage in linked_passages:
        passage_id = passage.get("passage_id")
        if not passage_id:
            continue
        if _contains_answer(passage.get("text", ""), answer_variants) or _contains_answer(
            passage.get("entity_name", ""),
            answer_variants,
        ):
            answer_passage_ids.append(passage_id)

    answer_row_ids = list(dict.fromkeys(answer_row_ids))
    answer_cell_ids = list(dict.fromkeys(answer_cell_ids))
    answer_passage_ids = list(dict.fromkeys(answer_passage_ids))

    row_scores: dict[str, float] = {}
    selected_row_ids: list[str] = []
    for row_meta, row in zip(table.get("row_metadata", []), table.get("rows", [])):
        row_id = row_meta["row_id"]
        row_text_parts = [str(value) for key, value in row.items() if key != "_links"]
        row_text = " ".join(row_text_parts)
        row_tokens = set(normalize_tokens(row_text))
        overlap = len(question_tokens & row_tokens)
        score = float(overlap)
        if row_id in answer_row_ids:
            score += 5.0
        linked_passage_ids = [
            link_to_passage[link]
            for link in row_meta.get("row_links", [])
            if link in link_to_passage
        ]
        answer_link_hits = [pid for pid in linked_passage_ids if pid in answer_passage_ids]
        if answer_link_hits:
            score += 4.0
        if ordinal_hint is not None:
            if ordinal_hint >= 0 and row_meta["row_index"] == ordinal_hint:
                score += 3.0
            elif ordinal_hint == -1 and row_meta["row_index"] == table.get("num_rows", 0) - 1:
                score += 3.0
        row_scores[row_id] = score
        if row_id in answer_row_ids or answer_link_hits:
            selected_row_ids.append(row_id)

    if not selected_row_ids and row_scores:
        best_row_id = max(row_scores, key=row_scores.get)
        if row_scores[best_row_id] > 0:
            selected_row_ids.append(best_row_id)

    selected_row_ids = sorted(set(selected_row_ids), key=lambda row_id: row_scores.get(row_id, 0), reverse=True)

    if selected_row_ids:
        linked_row_passages = {
            link_to_passage[link]
            for row_meta in table.get("row_metadata", [])
            if row_meta["row_id"] in selected_row_ids
            for link in row_meta.get("row_links", [])
            if link in link_to_passage
        }
        selected_passage_ids = [
            passage_id for passage_id in answer_passage_ids if passage_id in linked_row_passages
        ]
    else:
        selected_passage_ids = list(answer_passage_ids)

    if not selected_passage_ids and answer_passage_ids:
        selected_passage_ids = list(answer_passage_ids)

    selected_passage_ids = sorted(
        set(selected_passage_ids),
        key=lambda passage_id: passage_by_id[passage_id]["link_rank"],
    )

    proxy_evidence_refs = [*selected_row_ids, *selected_passage_ids]
    if selected_row_ids and selected_passage_ids:
        proxy_kind = "table_and_passage"
    elif selected_row_ids:
        proxy_kind = "table_only"
    elif selected_passage_ids:
        proxy_kind = "passage_only"
    else:
        proxy_kind = "unresolved"

    return {
        "label_mode": PROXY_EVIDENCE_LABEL_MODE,
        "strict_gold_evidence_available": False,
        "proxy_kind": proxy_kind,
        "proxy_evidence_refs": proxy_evidence_refs,
        "answer_anchor_row_ids": answer_row_ids,
        "answer_anchor_cell_ids": answer_cell_ids,
        "answer_anchor_passage_ids": answer_passage_ids,
        "selected_row_ids": selected_row_ids,
        "selected_passage_ids": selected_passage_ids,
        "row_scores": row_scores,
        "question_tokens": sorted(question_tokens),
        "row_cell_map": row_id_to_cells,
    }


def attach_proxy_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Backfill proxy evidence fields onto an existing parsed HybridQA record."""
    if record.get("evidence_alignment") and "proxy_evidence" in record:
        return record

    record["table"] = _ensure_table_provenance(record["table"], str(record["table_id"]))
    alignment = build_proxy_evidence(
        question=record["question"],
        answer=record["answer"],
        table=record["table"],
        linked_passages=record.get("linked_passages", []),
    )
    record["gold_evidence"] = None
    record["proxy_evidence"] = alignment["proxy_evidence_refs"]
    record["evidence_alignment"] = alignment
    return record
