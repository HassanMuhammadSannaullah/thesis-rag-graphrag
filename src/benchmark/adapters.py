"""Dataset adapters that normalize structured and unstructured sources."""
from __future__ import annotations

import json
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from src.benchmark.schemas import BenchmarkDataset, BenchmarkDocument, BenchmarkQuestion
from src.config import settings as cfg
from src.evaluation.normalization import normalize_answer_text


class DatasetAdapter(ABC):
    """Base class for all benchmark dataset adapters."""

    @abstractmethod
    def load(self) -> BenchmarkDataset:
        raise NotImplementedError


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = cfg.PROJECT_ROOT / value
    return value.resolve()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                return _as_list(parsed)
            except json.JSONDecodeError:
                pass
        return [stripped]
    return [str(value)]


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("rows", "records", "data", "questions", "documents"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError(f"Cannot find a record list in {path}")
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep).fillna("").to_dict(orient="records")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).fillna("").to_dict(orient="records")
    raise ValueError(f"Unsupported structured dataset file type: {path.suffix}")


def _infer_proxy_evidence_ids(
    documents: list[BenchmarkDocument],
    gold_answers: list[str],
    *,
    max_ids: int = 20,
) -> list[str]:
    normalized_answers = [
        normalize_answer_text(answer)
        for answer in gold_answers
        if normalize_answer_text(answer)
    ]
    if not normalized_answers:
        return []
    matches: list[str] = []
    for document in documents:
        normalized_text = normalize_answer_text(document.text)
        if any(answer in normalized_text for answer in normalized_answers):
            matches.append(document.id)
            if len(matches) >= max_ids:
                break
    return matches


def _ensure_proxy_evidence(
    questions: list[BenchmarkQuestion],
    documents: list[BenchmarkDocument],
) -> None:
    for question in questions:
        if question.gold_evidence_ids or question.proxy_evidence_ids:
            continue
        question.proxy_evidence_ids = _infer_proxy_evidence_ids(documents, question.gold_answers)


def _load_hybridqa_records(split: str, limit: int | None) -> list[dict[str, Any]]:
    path = cfg.ORIGINAL_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Parsed HybridQA file not found: {path}. "
            "Use a generic adapter for new datasets, or provide pre-normalized HybridQA records."
        )
    records = _load_records(path)
    return records[:limit] if limit is not None else records


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _zip_json(zf: zipfile.ZipFile, path: str) -> Any:
    with zf.open(path) as handle:
        return json.load(handle)


def _hybridqa_zip_path() -> Path:
    return cfg.RAW_DIR / "WikiTables-WithLinks.zip"


def _hybridqa_raw_questions_path(split: str) -> Path:
    return cfg.RAW_DIR / f"{split}.json"


def _parse_hybridqa_table(table_json: dict[str, Any], table_id: str) -> dict[str, Any]:
    headers = [str(cell[0]) if isinstance(cell, list) and cell else str(cell) for cell in table_json.get("header", [])]
    rows: list[dict[str, Any]] = []
    row_metadata: list[dict[str, Any]] = []
    cell_metadata: list[dict[str, Any]] = []
    all_links: list[str] = []

    for row_index, row_data in enumerate(table_json.get("data", [])):
        row: dict[str, Any] = {}
        row_links: list[str] = []
        row_id = f"row::{table_id}::{row_index}"
        cell_ids: list[str] = []
        for column_index, cell in enumerate(row_data):
            column_name = headers[column_index] if column_index < len(headers) else f"col_{column_index}"
            text = str(cell[0]) if isinstance(cell, list) and cell else str(cell)
            links = cell[1] if isinstance(cell, list) and len(cell) > 1 and isinstance(cell[1], list) else []
            links = _dedupe_preserve_order([str(link) for link in links])
            row[column_name] = text
            row_links.extend(links)
            all_links.extend(links)
            cell_id = f"cell::{table_id}::{row_index}::{column_index}"
            cell_ids.append(cell_id)
            cell_metadata.append(
                {
                    "cell_id": cell_id,
                    "row_id": row_id,
                    "row_index": row_index,
                    "column_index": column_index,
                    "column_name": column_name,
                    "text": text,
                    "links": links,
                }
            )
        row["_links"] = _dedupe_preserve_order(row_links)
        rows.append(row)
        row_metadata.append(
            {
                "row_id": row_id,
                "row_index": row_index,
                "cell_ids": cell_ids,
                "row_links": row["_links"],
            }
        )

    return {
        "table_id": table_id,
        "url": table_json.get("url"),
        "title": table_json.get("title", ""),
        "section_title": table_json.get("section_title", ""),
        "section_text": table_json.get("section_text", ""),
        "intro": table_json.get("intro", ""),
        "headers": headers,
        "rows": rows,
        "num_rows": len(rows),
        "all_links": _dedupe_preserve_order(all_links),
        "row_metadata": row_metadata,
        "cell_metadata": cell_metadata,
    }


def _parse_hybridqa_raw_records(split: str, limit: int | None) -> list[dict[str, Any]]:
    questions_path = _hybridqa_raw_questions_path(split)
    zip_path = _hybridqa_zip_path()
    if not questions_path.exists() or not zip_path.exists():
        return []

    question_rows = json.loads(questions_path.read_text(encoding="utf-8"))
    if limit is not None:
        question_rows = question_rows[:limit]

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for question_row in question_rows:
            table_id = str(question_row["table_id"])
            table_path = f"WikiTables-WithLinks-master/tables_tok/{table_id}.json"
            request_path = f"WikiTables-WithLinks-master/request_tok/{table_id}.json"
            table_json = _zip_json(zf, table_path)
            passages_json = _zip_json(zf, request_path)
            table = _parse_hybridqa_table(table_json, table_id)

            link_to_passage_id: dict[str, str] = {}
            linked_passages: list[dict[str, Any]] = []
            for link_rank, link in enumerate(table["all_links"]):
                text = str(passages_json.get(link) or "").strip()
                if not text:
                    continue
                passage_id = f"passage::{table_id}::{len(linked_passages)}"
                link_to_passage_id[link] = passage_id
                linked_passages.append(
                    {
                        "passage_id": passage_id,
                        "link": link,
                        "entity_name": link.split("/")[-1].replace("_", " "),
                        "text": text,
                        "link_rank": len(linked_passages),
                        "original_link_rank": link_rank,
                        "text_char_count": len(text),
                    }
                )

            for row_meta in table["row_metadata"]:
                row_meta["row_link_passage_ids"] = [
                    link_to_passage_id[link]
                    for link in row_meta.get("row_links", [])
                    if link in link_to_passage_id
                ]
            for cell_meta in table["cell_metadata"]:
                cell_meta["linked_passage_ids"] = [
                    link_to_passage_id[link]
                    for link in cell_meta.get("links", [])
                    if link in link_to_passage_id
                ]

            records.append(
                {
                    "question_id": str(question_row["question_id"]),
                    "question": str(question_row["question"]),
                    "question_postag": question_row.get("question_postag"),
                    "answer": question_row.get("answer-text", ""),
                    "table_id": table_id,
                    "table": table,
                    "linked_passages": linked_passages,
                    "num_linked_passages": len(linked_passages),
                    "split": split,
                    "source_metadata": {
                        "question_path": str(questions_path),
                        "zip_path": str(zip_path),
                        "table_path": table_path,
                        "request_path": request_path,
                        "parser": "HybridQAAdapter.raw_zip",
                    },
                }
            )
    return records


def _hybridqa_table_summary(record: dict[str, Any]) -> BenchmarkDocument:
    table = record["table"]
    headers = ", ".join(str(header) for header in table.get("headers", []))
    row_previews = []
    for row in table.get("rows", [])[:3]:
        row_previews.append(" | ".join(f"{key}: {value}" for key, value in row.items() if key != "_links"))
    text = (
        f"Table: {table.get('title', '')}\n"
        f"Section: {table.get('section_title', '')}\n"
        f"Columns: {headers}\n"
        f"Rows: {table.get('num_rows', len(table.get('rows', [])))}\n"
        f"Preview:\n" + "\n".join(row_previews)
    )
    if table.get("intro"):
        text += f"\nIntro: {table['intro']}"
    return BenchmarkDocument(
        id=f"summary::{record['table_id']}",
        text=text,
        source_type="table_summary",
        metadata={"table_id": record["table_id"], "question_id": record.get("question_id")},
    )


def _hybridqa_row_documents(record: dict[str, Any]) -> list[BenchmarkDocument]:
    table = record["table"]
    row_id_by_index = {
        row_meta["row_index"]: row_meta["row_id"]
        for row_meta in table.get("row_metadata", [])
        if "row_index" in row_meta and "row_id" in row_meta
    }
    documents = []
    for index, row in enumerate(table.get("rows", [])):
        pairs = [f"{key}: {value}" for key, value in row.items() if key != "_links"]
        text = f"Table: {table.get('title', '')} | Row {index + 1}/{table.get('num_rows', len(table.get('rows', [])))}\n"
        text += " | ".join(pairs)
        documents.append(
            BenchmarkDocument(
                id=row_id_by_index.get(index, f"row::{record['table_id']}::{index}"),
                text=text,
                source_type="table_row",
                metadata={
                    "table_id": record["table_id"],
                    "question_id": record.get("question_id"),
                    "row_index": index,
                    "row_links": row.get("_links", []),
                },
            )
        )
    return documents


def _hybridqa_passage_documents(record: dict[str, Any], max_passages: int | None) -> list[BenchmarkDocument]:
    passages = record.get("linked_passages", [])
    if max_passages is not None:
        passages = passages[:max_passages]
    documents = []
    for index, passage in enumerate(passages):
        text = str(passage.get("text") or "").strip()
        if not text:
            continue
        entity_name = passage.get("entity_name") or str(passage.get("link", "")).split("/")[-1].replace("_", " ")
        documents.append(
            BenchmarkDocument(
                id=passage.get("passage_id") or f"passage::{record['table_id']}::{index}",
                text=f"Entity: {entity_name}\n{text}",
                source_type="linked_passage",
                metadata={
                    "table_id": record["table_id"],
                    "question_id": record.get("question_id"),
                    "link": passage.get("link"),
                    "entity_name": entity_name,
                },
            )
        )
    return documents


def _hybridqa_documents(record: dict[str, Any], max_passages: int | None) -> list[BenchmarkDocument]:
    return [
        _hybridqa_table_summary(record),
        *_hybridqa_row_documents(record),
        *_hybridqa_passage_documents(record, max_passages),
    ]


class HybridQAAdapter(DatasetAdapter):
    """Adapter for HybridQA raw files, with parsed JSONL fallback."""

    def __init__(
        self,
        *,
        split: str = "dev",
        limit: int | None = None,
        max_passages: int | None = None,
        prefer_raw: bool = True,
        auto_download: bool = False,
    ):
        self.split = split
        self.limit = limit
        self.max_passages = max_passages
        self.prefer_raw = prefer_raw
        self.auto_download = auto_download

    def load(self) -> BenchmarkDataset:
        records = _parse_hybridqa_raw_records(self.split, self.limit) if self.prefer_raw else []
        source = "raw_zip" if records else "parsed_jsonl"
        if not records:
            records = _load_hybridqa_records(self.split, self.limit)
        documents = []
        for record in records:
            documents.extend(_hybridqa_documents(record, self.max_passages))
        questions = []
        for record in records:
            proxy_evidence = record.get("proxy_evidence") or {}
            proxy_evidence_ids = (
                _as_list(proxy_evidence.get("evidence_refs"))
                if isinstance(proxy_evidence, dict)
                else _as_list(proxy_evidence)
            )
            questions.append(
                BenchmarkQuestion(
                    question_id=str(record["question_id"]),
                    question=str(record["question"]),
                    gold_answers=_as_list(record.get("answer")),
                    gold_evidence_ids=_as_list(record.get("gold_evidence")),
                    proxy_evidence_ids=proxy_evidence_ids,
                    question_type=record.get("question_type"),
                    operation_type=record.get("operation_type"),
                    difficulty=record.get("difficulty"),
                    metadata={
                        "table_id": record.get("table_id"),
                        "split": record.get("split", self.split),
                        "adapter": "hybridqa",
                    },
                )
            )
        _ensure_proxy_evidence(questions, documents)
        return BenchmarkDataset(
            name=f"hybridqa_{self.split}",
            documents=documents,
            questions=questions,
            metadata={
                "adapter": "hybridqa",
                "split": self.split,
                "limit": self.limit,
                "max_passages": self.max_passages,
                "auto_download": self.auto_download,
                "record_count": len(records),
                "source": source,
            },
        )


class RecordsQAAdapter(DatasetAdapter):
    """Generic adapter for JSONL/JSON/CSV/TSV/Parquet QA rows.

    It supports rows where each record contains both the question and the text
    to index. For separate corpora, use ``DirectoryCorpusQAAdapter``.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        name: str | None = None,
        question_field: str = "question",
        answer_field: str = "answer",
        text_field: str = "text",
        id_field: str = "id",
        evidence_field: str | None = None,
        limit: int | None = None,
    ):
        self.path = _resolve_path(path)
        self.name = name or self.path.stem
        self.question_field = question_field
        self.answer_field = answer_field
        self.text_field = text_field
        self.id_field = id_field
        self.evidence_field = evidence_field
        self.limit = limit

    def load(self) -> BenchmarkDataset:
        rows = _load_records(self.path)
        if self.limit is not None:
            rows = rows[: self.limit]

        documents: list[BenchmarkDocument] = []
        questions: list[BenchmarkQuestion] = []
        for index, row in enumerate(rows):
            row_id = str(row.get(self.id_field) or f"row_{index}")
            text = str(row.get(self.text_field) or "").strip()
            if text:
                documents.append(
                    BenchmarkDocument(
                        id=f"doc::{row_id}",
                        text=text,
                        source_type="record_text",
                        metadata={"row_id": row_id, "adapter": "records_qa"},
                    )
                )
            question = str(row.get(self.question_field) or "").strip()
            if question:
                questions.append(
                    BenchmarkQuestion(
                        question_id=f"q::{row_id}",
                        question=question,
                        gold_answers=_as_list(row.get(self.answer_field)),
                        gold_evidence_ids=_as_list(row.get(self.evidence_field)) if self.evidence_field else [],
                        metadata={"row_id": row_id, "adapter": "records_qa"},
                    )
                )

        _ensure_proxy_evidence(questions, documents)
        return BenchmarkDataset(
            name=self.name,
            documents=documents,
            questions=questions,
            metadata={"adapter": "records_qa", "path": str(self.path), "row_count": len(rows)},
        )


class DirectoryCorpusQAAdapter(DatasetAdapter):
    """Adapter for unstructured files plus a separate QA file."""

    TEXT_SUFFIXES = {".txt", ".md", ".rst", ".html", ".htm"}

    def __init__(
        self,
        *,
        corpus_dir: str | Path,
        questions_path: str | Path,
        name: str | None = None,
        question_field: str = "question",
        answer_field: str = "answer",
        id_field: str = "question_id",
        evidence_field: str | None = None,
        recursive: bool = True,
        limit: int | None = None,
    ):
        self.corpus_dir = _resolve_path(corpus_dir)
        self.questions_path = _resolve_path(questions_path)
        self.name = name or self.corpus_dir.name
        self.question_field = question_field
        self.answer_field = answer_field
        self.id_field = id_field
        self.evidence_field = evidence_field
        self.recursive = recursive
        self.limit = limit

    def _load_documents(self) -> list[BenchmarkDocument]:
        try:
            from llama_index.core import SimpleDirectoryReader
        except Exception as exc:
            raise RuntimeError("LlamaIndex is required for directory corpus ingestion.") from exc

        reader = SimpleDirectoryReader(
            input_dir=str(self.corpus_dir),
            recursive=self.recursive,
            required_exts=sorted(self.TEXT_SUFFIXES),
        )
        loaded = reader.load_data()
        documents = []
        for index, document in enumerate(loaded):
            metadata = dict(getattr(document, "metadata", {}) or {})
            path = metadata.get("file_path") or metadata.get("filename") or f"document_{index}"
            doc_id = f"file::{Path(str(path)).name}::{index}"
            documents.append(
                BenchmarkDocument(
                    id=doc_id,
                    text=str(getattr(document, "text", "") or ""),
                    source_type="file",
                    metadata={
                        **metadata,
                        "adapter": "directory_corpus_qa",
                        "loader": "llama_index.SimpleDirectoryReader",
                    },
                )
            )
        return documents

    def load(self) -> BenchmarkDataset:
        documents = self._load_documents()
        rows = _load_records(self.questions_path)
        if self.limit is not None:
            rows = rows[: self.limit]
        questions = [
            BenchmarkQuestion(
                question_id=str(row.get(self.id_field) or f"q_{index}"),
                question=str(row[self.question_field]),
                gold_answers=_as_list(row.get(self.answer_field)),
                gold_evidence_ids=_as_list(row.get(self.evidence_field)) if self.evidence_field else [],
                metadata={"adapter": "directory_corpus_qa"},
            )
            for index, row in enumerate(rows)
            if row.get(self.question_field)
        ]
        _ensure_proxy_evidence(questions, documents)
        return BenchmarkDataset(
            name=self.name,
            documents=documents,
            questions=questions,
            metadata={
                "adapter": "directory_corpus_qa",
                "corpus_dir": str(self.corpus_dir),
                "questions_path": str(self.questions_path),
                "document_count": len(documents),
                "question_count": len(questions),
            },
        )


class TableCorpusQAAdapter(DatasetAdapter):
    """Adapter for structured tables where each row becomes an indexable document."""

    def __init__(
        self,
        *,
        corpus_path: str | Path,
        questions_path: str | Path,
        name: str | None = None,
        row_id_field: str | None = None,
        text_fields: list[str] | None = None,
        question_field: str = "question",
        answer_field: str = "answer",
        question_id_field: str = "question_id",
        evidence_field: str | None = None,
        limit: int | None = None,
    ):
        self.corpus_path = _resolve_path(corpus_path)
        self.questions_path = _resolve_path(questions_path)
        self.name = name or self.corpus_path.stem
        self.row_id_field = row_id_field
        self.text_fields = text_fields
        self.question_field = question_field
        self.answer_field = answer_field
        self.question_id_field = question_id_field
        self.evidence_field = evidence_field
        self.limit = limit

    def load(self) -> BenchmarkDataset:
        corpus_rows = _load_records(self.corpus_path)
        question_rows = _load_records(self.questions_path)
        if self.limit is not None:
            question_rows = question_rows[: self.limit]

        documents: list[BenchmarkDocument] = []
        for index, row in enumerate(corpus_rows):
            row_id = str(row.get(self.row_id_field) if self.row_id_field else f"row_{index}")
            fields = self.text_fields or [key for key in row.keys() if key != self.row_id_field]
            lines = [f"{field}: {row.get(field, '')}" for field in fields]
            documents.append(
                BenchmarkDocument(
                    id=f"table_row::{row_id}",
                    text="\n".join(lines),
                    source_type="table_row",
                    metadata={
                        "row_id": row_id,
                        "corpus_path": str(self.corpus_path),
                        "adapter": "table_corpus_qa",
                        "raw_row": row,
                    },
                )
            )

        questions = [
            BenchmarkQuestion(
                question_id=str(row.get(self.question_id_field) or f"q_{index}"),
                question=str(row[self.question_field]),
                gold_answers=_as_list(row.get(self.answer_field)),
                gold_evidence_ids=_as_list(row.get(self.evidence_field)) if self.evidence_field else [],
                metadata={"adapter": "table_corpus_qa"},
            )
            for index, row in enumerate(question_rows)
            if row.get(self.question_field)
        ]
        _ensure_proxy_evidence(questions, documents)
        return BenchmarkDataset(
            name=self.name,
            documents=documents,
            questions=questions,
            metadata={
                "adapter": "table_corpus_qa",
                "corpus_path": str(self.corpus_path),
                "questions_path": str(self.questions_path),
                "document_count": len(documents),
                "question_count": len(questions),
            },
        )


def build_dataset_adapter(config: dict[str, Any]) -> DatasetAdapter:
    adapter_type = str(config.get("type", "hybridqa")).lower()
    options = {key: value for key, value in config.items() if key != "type"}
    if adapter_type == "hybridqa":
        return HybridQAAdapter(**options)
    if adapter_type in {"records_qa", "jsonl_qa", "table_qa"}:
        return RecordsQAAdapter(**options)
    if adapter_type in {"directory_corpus_qa", "file_corpus_qa"}:
        return DirectoryCorpusQAAdapter(**options)
    if adapter_type in {"table_corpus_qa", "structured_table_qa"}:
        return TableCorpusQAAdapter(**options)
    raise ValueError(f"Unknown dataset adapter type: {adapter_type}")
