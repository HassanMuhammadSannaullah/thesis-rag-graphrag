"""Helpers for running Ragas against the thesis experiment schema."""
from __future__ import annotations

import importlib
import math
import time
import sys
import types
import asyncio
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from src.config import settings as cfg


RAGAS_CANONICAL_METRICS = [
    "answer_correctness",
    "answer_relevancy",
    "semantic_similarity",
    "factual_correctness",
    "faithfulness",
    "context_precision",
    "context_recall",
]

RAGAS_METRIC_ALIASES = {
    "answer_correctness": "answer_correctness",
    "answer_relevancy": "answer_relevancy",
    "response_relevancy": "answer_relevancy",
    "semantic_similarity": "semantic_similarity",
    "factual_correctness": "factual_correctness",
    "faithfulness": "faithfulness",
    "context_precision": "context_precision",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


@dataclass
class RagasRuntime:
    evaluate: Any
    EvaluationDataset: Any
    SingleTurnSample: Any
    RunConfig: Any
    llm_factory: Any
    embedding_factory: Any
    metric_classes: dict[str, Any]
    version: str


def _require_attr(module_name: str, attr_names: list[str]) -> Any:
    module = importlib.import_module(module_name)
    for attr_name in attr_names:
        value = getattr(module, attr_name, None)
        if value is not None:
            return value
    raise AttributeError(f"Could not find any of {attr_names} in {module_name}")


def _install_langchain_vertexai_shim() -> None:
    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "langchain_community.chat_models.vertexai":
            raise

    shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - compatibility shim only
        pass

    shim.ChatVertexAI = ChatVertexAI
    shim.__package__ = "langchain_community.chat_models"
    sys.modules[shim.__name__] = shim


def load_ragas_runtime() -> RagasRuntime:
    _install_langchain_vertexai_shim()
    try:
        ragas_module = importlib.import_module("ragas")
    except ImportError as exc:  # pragma: no cover - exercised in smoke tests
        raise RuntimeError(
            "ragas is not installed. Install project dependencies again after "
            "adding `ragas` to requirements.txt."
        ) from exc

    try:
        evaluate = getattr(ragas_module, "evaluate")
        EvaluationDataset = getattr(ragas_module, "EvaluationDataset")
        SingleTurnSample = getattr(ragas_module, "SingleTurnSample")
    except AttributeError as exc:  # pragma: no cover
        raise RuntimeError("Installed ragas version is missing core evaluation APIs.") from exc

    RunConfig = _require_attr("ragas.run_config", ["RunConfig"])
    llm_factory = _require_attr("ragas.llms", ["llm_factory"])
    try:
        embedding_factory = _require_attr("ragas.embeddings.base", ["embedding_factory"])
    except Exception:
        embedding_factory = _require_attr("ragas.embeddings", ["embedding_factory"])

    metric_module_candidates = [
        "ragas.metrics",
        "ragas.metrics.collections",
    ]
    metric_class_names = {
        "answer_correctness": ["AnswerCorrectness"],
        "answer_relevancy": ["AnswerRelevancy", "ResponseRelevancy"],
        "semantic_similarity": ["SemanticSimilarity"],
        "factual_correctness": ["FactualCorrectness"],
        "faithfulness": ["Faithfulness"],
        "context_precision": ["ContextPrecision", "LLMContextPrecisionWithReference"],
        "context_recall": ["ContextRecall", "LLMContextRecall"],
    }

    metric_classes: dict[str, Any] = {}
    for canonical_name, candidates in metric_class_names.items():
        for module_name in metric_module_candidates:
            try:
                metric_classes[canonical_name] = _require_attr(module_name, candidates)
                break
            except Exception:
                continue
        if canonical_name not in metric_classes:
            raise RuntimeError(
                f"Installed ragas version is missing a metric implementation for `{canonical_name}`."
            )

    return RagasRuntime(
        evaluate=evaluate,
        EvaluationDataset=EvaluationDataset,
        SingleTurnSample=SingleTurnSample,
        RunConfig=RunConfig,
        llm_factory=llm_factory,
        embedding_factory=embedding_factory,
        metric_classes=metric_classes,
        version=getattr(ragas_module, "__version__", "unknown"),
    )


def build_ragas_llm(runtime: RagasRuntime, *, generation_model: str) -> Any:
    if cfg.RAGAS_EVAL_BACKEND == "local_openai":
        client = AsyncOpenAI(
            api_key=cfg.LOCAL_LLM_API_KEY,
            base_url=cfg.LOCAL_LLM_BASE_URL,
        )
        return runtime.llm_factory(generation_model, client=client)

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-genai is required to use ragas with Gemini.") from exc

    client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
    llm = runtime.llm_factory(generation_model, provider="google", client=client)
    if hasattr(llm, "generate"):
        original_generate = llm.generate

        def throttled_generate(*args: Any, **kwargs: Any) -> Any:
            time.sleep(cfg.SLEEP_BETWEEN_REQ)
            return original_generate(*args, **kwargs)

        llm.generate = throttled_generate

    if hasattr(llm, "agenerate"):
        original_agenerate = llm.agenerate

        async def throttled_agenerate(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(cfg.SLEEP_BETWEEN_REQ)
            return await original_agenerate(*args, **kwargs)

        llm.agenerate = throttled_agenerate
    return llm


def build_ragas_embeddings(runtime: RagasRuntime, *, embedding_model: str) -> Any:
    if cfg.RAGAS_EVAL_BACKEND == "local_openai":
        client = AsyncOpenAI(
            api_key=cfg.LOCAL_LLM_API_KEY,
            base_url=cfg.LOCAL_LLM_BASE_URL,
        )
        embeddings = runtime.embedding_factory("openai", model=embedding_model, client=client)
    else:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai is required to use ragas with Gemini.") from exc

        client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
        embeddings = runtime.embedding_factory("google", model=embedding_model, client=client)

    # Some ragas metrics still expect LangChain-style embedding names.
    if not hasattr(embeddings, "embed_query"):
        embeddings.embed_query = embeddings.embed_text  # type: ignore[attr-defined]
    if not hasattr(embeddings, "aembed_query"):
        embeddings.aembed_query = embeddings.aembed_text  # type: ignore[attr-defined]
    if not hasattr(embeddings, "embed_documents"):
        embeddings.embed_documents = embeddings.embed_texts  # type: ignore[attr-defined]
    if not hasattr(embeddings, "aembed_documents"):
        embeddings.aembed_documents = embeddings.aembed_texts  # type: ignore[attr-defined]
    return embeddings


def build_default_ragas_metrics(
    runtime: RagasRuntime,
    *,
    llm: Any,
    embeddings: Any,
) -> list[Any]:
    metric_classes = runtime.metric_classes
    return [
        metric_classes["answer_correctness"](llm=llm, embeddings=embeddings),
        metric_classes["answer_relevancy"](llm=llm, embeddings=embeddings, strictness=1),
        metric_classes["semantic_similarity"](embeddings=embeddings),
        metric_classes["factual_correctness"](llm=llm),
        metric_classes["faithfulness"](llm=llm),
        metric_classes["context_precision"](llm=llm),
        metric_classes["context_recall"](llm=llm),
    ]


def canonicalize_ragas_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {name: None for name in RAGAS_CANONICAL_METRICS}
    for raw_name, canonical_name in RAGAS_METRIC_ALIASES.items():
        value = row.get(raw_name)
        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                metrics[canonical_name] = None
            else:
                metrics[canonical_name] = float(value)
        elif value is None:
            metrics.setdefault(canonical_name, None)
    return metrics
