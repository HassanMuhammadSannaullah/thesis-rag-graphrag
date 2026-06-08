"""Core schemas for reusable experiment evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievedContext:
    id: str | None = None
    text: str | None = None
    score: float | None = None
    rank: int | None = None
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedContext":
        return cls(
            id=data.get("id"),
            text=data.get("text"),
            score=data.get("score"),
            rank=data.get("rank"),
            source_type=data.get("source_type"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationExample:
    question_id: str
    question: str
    gold_answer: str | list[str]
    gold_evidence: list[str] | None = None
    proxy_evidence: list[str] | None = None
    evidence_label_mode: str | None = None
    question_type: str | None = None
    operation_type: str | None = None
    difficulty: str | None = None
    answer_type: str | None = None
    aliases: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationExample":
        gold_answer = data.get("gold_answer", data.get("answer", ""))
        return cls(
            question_id=str(data["question_id"]),
            question=str(data["question"]),
            gold_answer=gold_answer,
            gold_evidence=data.get("gold_evidence") or data.get("evidence_refs"),
            proxy_evidence=data.get("proxy_evidence") or data.get("proxy_evidence_refs"),
            evidence_label_mode=(
                data.get("evidence_label_mode")
                or (data.get("evidence_alignment") or {}).get("label_mode")
            ),
            question_type=data.get("question_type"),
            operation_type=data.get("operation_type"),
            difficulty=data.get("difficulty"),
            answer_type=data.get("answer_type"),
            aliases=data.get("aliases"),
            metadata=dict(data.get("metadata") or {}),
        )

    def all_gold_answers(self) -> list[str]:
        values = self.gold_answer if isinstance(self.gold_answer, list) else [self.gold_answer]
        extras = self.aliases or []
        return [str(v) for v in values + extras if v is not None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemPrediction:
    question_id: str
    system_name: str
    predicted_answer: str
    retrieved_contexts: list[RetrievedContext] | None = None
    latency_seconds: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemPrediction":
        contexts = data.get("retrieved_contexts")
        return cls(
            question_id=str(data["question_id"]),
            system_name=str(data["system_name"]),
            predicted_answer=str(data.get("predicted_answer", "")),
            retrieved_contexts=(
                [RetrievedContext.from_dict(row) for row in contexts]
                if contexts is not None
                else None
            ),
            latency_seconds=data.get("latency_seconds", data.get("query_latency_seconds")),
            prompt_tokens=data.get("prompt_tokens"),
            output_tokens=data.get("output_tokens"),
            total_tokens=data.get("total_tokens"),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "retrieved_contexts": (
                [ctx.to_dict() for ctx in self.retrieved_contexts]
                if self.retrieved_contexts is not None
                else None
            ),
        }


@dataclass
class MetricResult:
    question_id: str
    system_name: str
    dataset_name: str
    metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
