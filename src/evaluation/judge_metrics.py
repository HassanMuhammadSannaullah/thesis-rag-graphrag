"""Optional LLM-as-judge metrics via the shared model client."""
from __future__ import annotations

import json
import re
from typing import Any

from src.utils.model_client import generate_text

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    repair_json = None


JUDGE_TEMPLATE = """You are a deterministic evaluation judge for retrieval-augmented QA.
Return JSON only with these keys:
- faithfulness: float from 0 to 1
- answer_relevance: float from 0 to 1
- context_relevance: float from 0 to 1
- context_precision_llm: float from 0 to 1
- context_recall_llm: float from 0 to 1
- rationale: short string

Scoring rules:
- faithfulness: whether the answer is supported by the retrieved context, penalizing unsupported claims
- answer_relevance: whether the answer directly answers the question
- context_relevance: whether the retrieved context is useful for answering the question
- context_precision_llm: whether the retrieved context is focused rather than noisy
- context_recall_llm: whether the retrieved context contains the key evidence needed

Question:
{question}

Gold answer:
{gold_answer}

Predicted answer:
{predicted_answer}

Retrieved context:
{retrieved_context}
"""


def _extract_json_blob(text: str) -> str:
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if fenced:
        return fenced.group(0)
    return text.strip()


def _parse_judge_json(text: str) -> dict[str, Any]:
    blob = _extract_json_blob(text)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        if repair_json is None:
            raise
        repaired = repair_json(blob)
        return json.loads(repaired)


def compute_judge_metrics(
    *,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    retrieved_context_text: str,
    model: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        retrieved_context=retrieved_context_text or "(no retrieved context)",
    )
    raw = generate_text(prompt, model=model, temperature=0.0, max_tokens=256, use_cache=False)
    parsed = _parse_judge_json(raw)
    metrics = {}
    for key in [
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "context_precision_llm",
        "context_recall_llm",
    ]:
        value = parsed.get(key)
        metrics[key] = float(value) if value is not None else None
    return metrics, parsed.get("rationale")
