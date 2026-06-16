"""Canonical schemas for dataset-agnostic RAG/GraphRAG benchmarking."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.evaluation.schemas import EvaluationExample


@dataclass
class BenchmarkDocument:
    """A document, row, passage, table, or chunk that can be indexed."""

    id: str
    text: str
    source_type: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkQuestion:
    """A QA item with optional gold evidence for retrieval evaluation."""

    question_id: str
    question: str
    gold_answers: list[str]
    gold_evidence_ids: list[str] = field(default_factory=list)
    proxy_evidence_ids: list[str] = field(default_factory=list)
    answer_type: str | None = None
    question_type: str | None = None
    operation_type: str | None = None
    difficulty: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_evaluation_example(self) -> EvaluationExample:
        return EvaluationExample(
            question_id=self.question_id,
            question=self.question,
            gold_answer=self.gold_answers,
            gold_evidence=self.gold_evidence_ids or None,
            proxy_evidence=self.proxy_evidence_ids or None,
            evidence_label_mode="canonical",
            answer_type=self.answer_type,
            question_type=self.question_type,
            operation_type=self.operation_type,
            difficulty=self.difficulty,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkDataset:
    """Canonical dataset payload consumed by both baseline RAG and GraphRAG."""

    name: str
    documents: list[BenchmarkDocument]
    questions: list[BenchmarkQuestion]
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluation_examples(self) -> list[EvaluationExample]:
        return [question.to_evaluation_example() for question in self.questions]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
