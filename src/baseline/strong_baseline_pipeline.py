"""
Strong baseline RAG pipeline for Phase 3.

End-to-end pipeline that:
  1. Builds corpus from parsed HybridQA records (rows + passages)
  2. Creates vector index
  3. Retrieves with row-aware passage expansion
  4. Reranks with lexical+semantic hybrid
  5. Generates grounded answers with provenance tracking
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.baseline import corpus_builder, retriever
from src.baseline.strong_answer_generator import generate_answer_simple_baseline, generate_answer_strong
from src.baseline.strong_retriever import retrieve_strong
from src.baseline.vector_index import LocalVectorIndex
from src.evaluation.schemas import RetrievedContext, SystemPrediction


class StrongBaselinePipeline:
    """
    Strong baseline RAG pipeline with all Phase 3 enhancements.
    """
    
    def __init__(
        self,
        variant: str = "strong",  # "simple" or "strong"
        embedding_model: str = "intfloat/e5-base-v2",
        top_k: int = 8,
        use_lexical: bool = True,
        use_reranking: bool = True,
        include_citations: bool = False,
        max_context_chars: int = 12000,
        max_answer_tokens: int = 256,
    ):
        """
        Initialize strong baseline pipeline.
        
        Args:
            variant: "simple" for ablation, "strong" for full pipeline
            embedding_model: Model for vector index
            top_k: Number of evidence units to retrieve
            use_lexical: Enable hybrid lexical+semantic retrieval
            use_reranking: Enable cross-encoder reranking
            include_citations: Include evidence citations in answers
            max_context_chars: Maximum context length for answer generation
            max_answer_tokens: Maximum answer length
        """
        self.variant = variant
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.use_lexical = use_lexical and (variant == "strong")
        self.use_reranking = use_reranking and (variant == "strong")
        self.include_citations = include_citations
        self.max_context_chars = max_context_chars
        self.max_answer_tokens = max_answer_tokens
        
        self.index = None
        self.passage_lookup = None
        self.corpus = None
    
    def build_corpus(self, records: list[dict], max_passages: int = 30) -> list[dict]:
        """Build retrieval corpus from parsed HybridQA records."""
        print(f"Building corpus from {len(records)} records (variant={self.variant})...")
        self.corpus = corpus_builder.build_corpus(records, max_passages=max_passages)
        print(f"Created {len(self.corpus)} retrieval units")
        return self.corpus
    
    def build_index(self, corpus: list[dict] = None, cache_dir: Path = None, force_rebuild: bool = False):
        """Build vector index from corpus."""
        if corpus is None:
            if self.corpus is None:
                raise ValueError("Must build corpus first or provide corpus")
            corpus = self.corpus
        
        # Determine index path
        if cache_dir is None:
            cache_dir = Path("cache/indexes/default")
        index_path = Path(cache_dir) / "vector_index"
        
        print(f"Building vector index with {self.embedding_model}...")
        self.index = LocalVectorIndex(index_path=index_path)

        metadata_is_complete = all(
            meta.get("text")
            and (meta.get("type") != "table_row" or "row_links" in meta)
            for meta in self.index.metadata
        )
        if force_rebuild or (self.index.metadata and not metadata_is_complete):
            reason = "force rebuild requested" if force_rebuild else "cached metadata is missing retrieval fields"
            print(f"  Rebuilding vector index ({reason}).")
            self.index.embeddings = None
            self.index.metadata = []
        
        # Extract texts and metadata for indexing
        texts = [doc["text"] for doc in corpus]
        metadatas = [dict(doc) for doc in corpus]
        
        self.index.add(texts=texts, metadatas=metadatas)
        print(f"Index built with {len(corpus)} documents")
    
    def build_passage_lookup(self, records: list[dict]):
        """Build passage lookup for row-aware expansion."""
        print(f"Building passage lookup...")
        self.passage_lookup = retriever.build_passage_lookup(records)
        print(f"Passage lookup contains {len(self.passage_lookup)} linked passages")
    
    def prepare(
        self,
        records: list[dict],
        max_passages: int = None,
        cache_dir: Path = None,
        force_rebuild: bool = False,
    ):
        """
        Prepare the full pipeline: corpus, index, and passage lookup.
        """
        # Build corpus
        self.build_corpus(records, max_passages=max_passages)
        
        # Build index
        self.build_index(corpus=self.corpus, cache_dir=cache_dir, force_rebuild=force_rebuild)
        
        # Build passage lookup for strong variant
        if self.variant == "strong":
            self.build_passage_lookup(records)
    
    def retrieve(self, question: str) -> list[dict]:
        """Retrieve evidence for a question."""
        if self.index is None:
            raise ValueError("Must build index first")
        
        if self.variant == "simple":
            # Simple retrieval: just vector search, no expansion
            search_results = self.index.search(question, top_k=self.top_k)
            
            # Add text from corpus using IDs
            corpus_by_id = {doc["id"]: doc for doc in self.corpus}
            for result in search_results:
                doc_id = result.get("id")
                if doc_id and doc_id in corpus_by_id:
                    result["text"] = corpus_by_id[doc_id]["text"]
                    result["type"] = corpus_by_id[doc_id].get("type", "unknown")
                else:
                    result["text"] = "[Text not found]"
                    result["type"] = "unknown"
            
            return search_results
        
        elif self.variant == "strong":
            # Strong retrieval: row-aware, lexical, reranking
            if self.passage_lookup is None:
                raise ValueError("Must build passage lookup for strong variant")
            
            return retrieve_strong(
                question=question,
                index=self.index,
                passage_lookup=self.passage_lookup,
                top_k=self.top_k,
                use_lexical=self.use_lexical,
                use_reranking=self.use_reranking,
            )
        
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
    
    def generate_answer(self, question: str, evidence_units: list[dict]) -> dict[str, Any]:
        """Generate answer from question and evidence."""
        if self.variant == "simple":
            return generate_answer_simple_baseline(
                question=question,
                evidence_units=evidence_units,
                max_answer_tokens=self.max_answer_tokens,
            )
        
        elif self.variant == "strong":
            return generate_answer_strong(
                question=question,
                evidence_units=evidence_units,
                max_context_chars=self.max_context_chars,
                max_answer_tokens=self.max_answer_tokens,
                include_citations=self.include_citations,
            )
        
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
    
    def query(self, question: str) -> dict[str, Any]:
        """
        End-to-end query: retrieve evidence and generate answer.
        
        Returns full prediction with provenance.
        """
        start_time = time.time()
        
        # Retrieve
        evidence_units = self.retrieve(question)
        retrieve_time = time.time() - start_time
        
        # Generate answer
        answer_result = self.generate_answer(question, evidence_units)
        total_time = time.time() - start_time
        
        # Build full prediction
        prediction = {
            "question": question,
            "answer": answer_result["generated_answer"],
            "evidence_units": evidence_units,
            "num_evidence_units": len(evidence_units),
            "variant": self.variant,
            "retrieve_time_sec": retrieve_time,
            "total_time_sec": total_time,
        }
        
        # Add provenance if strong variant
        if self.variant == "strong":
            prediction["provenance"] = answer_result.get("provenance", {})
            prediction["context_summary"] = answer_result.get("context_summary", {})
            if self.include_citations:
                prediction["cited_evidence"] = answer_result.get("cited_evidence", [])
                prediction["citation_map"] = answer_result.get("citation_map", {})
        
        return prediction
    
    def to_system_prediction(self, question: str, question_id: str, gold_answer: str = None) -> SystemPrediction:
        """
        Run query and convert to SystemPrediction schema for evaluation.
        """
        result = self.query(question)
        
        # Convert evidence units to RetrievedContext
        contexts = [
            RetrievedContext(
                text=unit.get("text", ""),
                id=unit.get("id", ""),
                score=unit.get("rerank_score", unit.get("hybrid_score", unit.get("score", 0.0))),
                source_type=unit.get("type"),
                metadata={
                    "type": unit.get("type"),
                    "table_id": unit.get("table_id"),
                },
            )
            for unit in result["evidence_units"]
        ]
        
        return SystemPrediction(
            question_id=question_id,
            system_name=f"baseline_{self.variant}",
            predicted_answer=result["answer"],
            retrieved_contexts=contexts,
            latency_seconds=result["total_time_sec"],
            metadata={
                "question": question,
                "variant": self.variant,
                "embedding_model": self.embedding_model,
                "top_k": self.top_k,
                "use_lexical": self.use_lexical,
                "use_reranking": self.use_reranking,
                "retrieve_time_sec": result["retrieve_time_sec"],
                "total_time_sec": result["total_time_sec"],
                "provenance": result.get("provenance"),
                "context_summary": result.get("context_summary"),
            },
        )


def run_strong_baseline_on_questions(
    questions: list[dict],
    pipeline: StrongBaselinePipeline,
    output_path: Path = None,
) -> list[SystemPrediction]:
    """
    Run strong baseline pipeline on a list of questions.
    
    Args:
        questions: List of dicts with at least {question_id, question, answer}
        pipeline: Prepared StrongBaselinePipeline
        output_path: Optional path to save predictions
    
    Returns:
        List of SystemPrediction objects
    """
    predictions = []
    
    print(f"\nRunning {pipeline.variant} baseline on {len(questions)} questions...")
    
    for i, q in enumerate(questions, 1):
        question_id = q.get("question_id")
        question_text = q.get("question")
        gold_answer = q.get("answer", q.get("answer-text"))
        
        print(f"[{i}/{len(questions)}] {question_id}: {question_text[:80]}...")
        
        try:
            pred = pipeline.to_system_prediction(
                question=question_text,
                question_id=question_id,
                gold_answer=gold_answer,
            )
            predictions.append(pred)
            
            print(f"  → Predicted: {pred.predicted_answer}")
            print(f"  → Gold: {gold_answer}")
            print(f"  → Retrieved {len(pred.retrieved_contexts)} evidence units")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            # Create empty prediction on error
            predictions.append(
                SystemPrediction(
                    question_id=question_id,
                    system_name=f"baseline_{pipeline.variant}",
                    predicted_answer="ERROR",
                    retrieved_contexts=[],
                    metadata={
                        "question": question_text,
                        "gold_answer": gold_answer,
                        "error": str(e),
                    },
                )
            )
    
    # Save predictions if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred.dict(), ensure_ascii=False) + "\n")
        print(f"\nSaved {len(predictions)} predictions to {output_path}")
    
    return predictions
