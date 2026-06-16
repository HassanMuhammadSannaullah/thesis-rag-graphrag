"""Library-backed baseline RAG for standard benchmark runs."""
from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from src.benchmark.schemas import BenchmarkDocument, BenchmarkQuestion
from src.config import settings as cfg
from src.evaluation.schemas import RetrievedContext, SystemPrediction
from src.utils.model_client import embed_texts, generate_text


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class StandardRagConfig:
    top_k: int = 8
    dense_top_k: int = 30
    lexical_top_k: int = 30
    fusion_k: int = 60
    use_dense: bool = True
    use_bm25: bool = True
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = 16
    max_context_chars: int = 16000
    max_answer_tokens: int = 256
    vector_backend: str = "faiss"


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_index in enumerate(ranking, start=1):
            scores[doc_index] = scores.get(doc_index, 0.0) + (1.0 / (k + rank))
    return scores


class DenseVectorBackend:
    """Dense retrieval backed by FAISS."""

    def __init__(self, vectors: np.ndarray, backend: str = "faiss"):
        if backend != "faiss":
            raise ValueError("StandardRagPipeline requires vector_backend='faiss'.")
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError(
                "FAISS is required for standard dense retrieval, but `import faiss` failed "
                f"under Python: {sys.executable}. Install it into this exact environment with: "
                f"`{sys.executable} -m pip install faiss-cpu==1.12.0`, or use a conda FAISS package."
            ) from exc
        self.vectors = vectors.astype("float32")
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.normalized = self.vectors / np.where(norms == 0, 1, norms)
        self.index = faiss.IndexFlatIP(self.normalized.shape[1])
        self.index.add(self.normalized)
        self.backend = "faiss"

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        query = np.array(query_vector, dtype="float32")
        query = query / (np.linalg.norm(query) + 1e-10)
        scores, indices = self.index.search(query.reshape(1, -1), min(top_k, len(self.normalized)))
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if int(index) >= 0
        ]


class LexicalBackend:
    """BM25 retrieval using rank-bm25."""

    def __init__(self, documents: list[BenchmarkDocument]):
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:
            raise RuntimeError("rank-bm25 is required for standard lexical retrieval.") from exc
        self.documents = documents
        self.tokenized = [_tokens(document.text) for document in documents]
        self.bm25 = BM25Okapi(self.tokenized)
        self.backend = "rank_bm25"

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(index), float(scores[index])) for index in indices if float(scores[index]) > 0]


class OptionalCrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._lock = Lock()
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required when reranking is enabled.") from exc
        self.model = CrossEncoder(model_name)

    @property
    def available(self) -> bool:
        return self.model is not None

    def rerank(self, question: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not candidates:
            return candidates[:top_k]
        pairs = [(question, candidate["document"].text) for candidate in candidates]
        with self._lock:
            scores = self.model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        return sorted(candidates, key=lambda row: row.get("rerank_score", 0.0), reverse=True)[:top_k]


class StandardRagPipeline:
    """Hybrid dense/BM25/RRF/rerank RAG baseline with stable provenance."""

    def __init__(self, documents: list[BenchmarkDocument], config: StandardRagConfig):
        self.documents = documents
        self.config = config
        self.dense_backend: DenseVectorBackend | None = None
        self.lexical_backend: LexicalBackend | None = None
        self.reranker: OptionalCrossEncoderReranker | None = None
        self.backend_metadata: dict[str, Any] = {}

    def build(self) -> None:
        if not self.config.use_dense:
            raise ValueError("Standard industry baseline requires dense FAISS retrieval.")
        if not self.config.use_bm25:
            raise ValueError("Standard industry baseline requires BM25 lexical retrieval.")
        if not self.config.use_reranker:
            raise ValueError("Standard industry baseline requires cross-encoder reranking.")

        if self.config.use_dense:
            vectors: list[list[float]] = []
            texts = [document.text for document in self.documents]
            batches = [
                texts[start : start + self.config.embedding_batch_size]
                for start in range(0, len(texts), self.config.embedding_batch_size)
            ]
            if cfg.EMBEDDING_CONCURRENT_REQUESTS <= 1 or len(batches) <= 1:
                for batch in batches:
                    vectors.extend(embed_texts(batch))
            else:
                with ThreadPoolExecutor(max_workers=cfg.EMBEDDING_CONCURRENT_REQUESTS) as executor:
                    for batch_vectors in executor.map(embed_texts, batches):
                        vectors.extend(batch_vectors)
            self.dense_backend = DenseVectorBackend(np.array(vectors), backend=self.config.vector_backend)
            self.backend_metadata["dense_backend"] = self.dense_backend.backend
            self.backend_metadata["embedding_concurrent_requests"] = cfg.EMBEDDING_CONCURRENT_REQUESTS

        if self.config.use_bm25:
            self.lexical_backend = LexicalBackend(self.documents)
            self.backend_metadata["lexical_backend"] = self.lexical_backend.backend

        if self.config.use_reranker:
            self.reranker = OptionalCrossEncoderReranker(self.config.reranker_model)
            self.backend_metadata["reranker_backend"] = (
                self.config.reranker_model if self.reranker.available else "unavailable"
            )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        dense_hits = []
        lexical_hits = []
        if self.dense_backend is not None:
            query_vector = embed_texts([question])[0]
            dense_hits = self.dense_backend.search(query_vector, self.config.dense_top_k)
        if self.lexical_backend is not None:
            lexical_hits = self.lexical_backend.search(question, self.config.lexical_top_k)

        rankings = [
            [index for index, _ in dense_hits],
            [index for index, _ in lexical_hits],
        ]
        fused = _rrf([ranking for ranking in rankings if ranking], k=self.config.fusion_k)
        score_by_index: dict[int, dict[str, float]] = {}
        for index, score in dense_hits:
            score_by_index.setdefault(index, {})["dense_score"] = score
        for index, score in lexical_hits:
            score_by_index.setdefault(index, {})["lexical_score"] = score

        candidates = [
            {
                "document": self.documents[index],
                "score": score,
                **score_by_index.get(index, {}),
            }
            for index, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
        ]
        if self.reranker is not None and self.reranker.available:
            candidates = self.reranker.rerank(question, candidates, self.config.top_k)
        else:
            candidates = candidates[: self.config.top_k]
        return candidates

    def _format_context(self, candidates: list[dict[str, Any]]) -> str:
        sections = []
        total = 0
        for rank, candidate in enumerate(candidates, start=1):
            document = candidate["document"]
            header = f"[{rank}] id={document.id} type={document.source_type}"
            text = f"{header}\n{document.text}".strip()
            if total + len(text) > self.config.max_context_chars:
                remaining = self.config.max_context_chars - total
                if remaining > 300:
                    sections.append(text[:remaining])
                break
            sections.append(text)
            total += len(text)
        return "\n\n".join(sections)

    def answer(self, question: BenchmarkQuestion, candidates: list[dict[str, Any]]) -> str:
        context = self._format_context(candidates)
        prompt = f"""Answer the question using only the provided context.

Rules:
- Return the shortest exact answer possible.
- If the answer is not supported by the context, return: INSUFFICIENT EVIDENCE
- Do not include explanations, markdown, or citations.

Context:
{context}

Question: {question.question}

Answer:"""
        return generate_text(prompt, max_tokens=self.config.max_answer_tokens).strip()

    def query(self, question: BenchmarkQuestion) -> SystemPrediction:
        started = time.perf_counter()
        try:
            candidates = self.retrieve(question.question)
            answer = self.answer(question, candidates)
            contexts = []
            for rank, candidate in enumerate(candidates, start=1):
                document = candidate["document"]
                contexts.append(
                    RetrievedContext(
                        id=document.id,
                        text=document.text,
                        score=float(candidate.get("rerank_score", candidate.get("score", 0.0))),
                        rank=rank,
                        source_type=document.source_type,
                        metadata={
                            **document.metadata,
                            "dense_score": candidate.get("dense_score"),
                            "lexical_score": candidate.get("lexical_score"),
                            "rrf_score": candidate.get("score"),
                            "rerank_score": candidate.get("rerank_score"),
                        },
                    )
                )
            return SystemPrediction(
                question_id=question.question_id,
                system_name="standard_hybrid_rag",
                predicted_answer=answer,
                retrieved_contexts=contexts,
                latency_seconds=time.perf_counter() - started,
                metadata={"backend": self.backend_metadata, "config": self.config.__dict__},
            )
        except Exception as exc:
            return SystemPrediction(
                question_id=question.question_id,
                system_name="standard_hybrid_rag",
                predicted_answer="ERROR",
                retrieved_contexts=[],
                latency_seconds=time.perf_counter() - started,
                error=str(exc),
                metadata={"backend": self.backend_metadata, "config": self.config.__dict__},
            )
