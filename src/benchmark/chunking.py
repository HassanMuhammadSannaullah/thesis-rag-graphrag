"""Shared text-unit construction for fair and industry-default experiments."""
from __future__ import annotations

from dataclasses import dataclass

from src.benchmark.schemas import BenchmarkDocument


@dataclass
class ChunkingConfig:
    strategy: str = "sentence"
    chunk_size: int = 900
    chunk_overlap: int = 120
    keep_original: bool = False


def _llama_index_chunks(text: str, chunk_size: int, chunk_overlap: int, strategy: str) -> list[str]:
    try:
        from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
    except Exception as exc:
        raise RuntimeError("LlamaIndex is required for benchmark chunking.") from exc
    if strategy in {"sentence", "llama_index_sentence"}:
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy in {"token", "llama_index_token"}:
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def chunk_documents(documents: list[BenchmarkDocument], config: ChunkingConfig) -> list[BenchmarkDocument]:
    """Return canonical index units, preferring LlamaIndex chunkers when available."""
    if config.strategy in {"none", "prechunked"}:
        return documents

    chunked: list[BenchmarkDocument] = []
    for document in documents:
        if config.keep_original:
            chunked.append(document)
        parts = _llama_index_chunks(document.text, config.chunk_size, config.chunk_overlap, config.strategy)

        for index, text in enumerate(parts):
            chunk_id = f"{document.id}::chunk::{index}"
            chunked.append(
                BenchmarkDocument(
                    id=chunk_id,
                    text=text,
                    source_type=f"{document.source_type}_chunk",
                    metadata={
                        **document.metadata,
                        "parent_id": document.id,
                        "chunk_index": index,
                        "chunk_count": len(parts),
                        "chunking_strategy": config.strategy,
                    },
                )
            )
    return chunked
