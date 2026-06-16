"""Dataset-agnostic Microsoft GraphRAG wrapper."""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from src.benchmark.schemas import BenchmarkDocument, BenchmarkQuestion
from src.config import settings as cfg
from src.evaluation.schemas import RetrievedContext, SystemPrediction
from src.graphrag_system.runner import create_graphrag_config, has_graphrag_index, run_graphrag_index


def _local_graphrag_search_kwargs() -> dict[str, Any]:
    return {
        "community_prop": cfg.GRAPHRAG_LOCAL_COMMUNITY_PROP,
        "text_unit_prop": cfg.GRAPHRAG_LOCAL_TEXT_UNIT_PROP,
        "top_k_mapped_entities": cfg.GRAPHRAG_LOCAL_TOP_K_MAPPED_ENTITIES,
        "top_k_relationships": cfg.GRAPHRAG_LOCAL_TOP_K_RELATIONSHIPS,
        "max_context_tokens": cfg.GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS,
        "include_relationship_weight": cfg.GRAPHRAG_LOCAL_INCLUDE_RELATIONSHIP_WEIGHT,
    }


def write_graphrag_input(documents: list[BenchmarkDocument], input_dir: Path, *, clean: bool = False) -> None:
    if clean and input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for index, document in enumerate(documents):
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in document.id)
        path = input_dir / f"{index:06d}_{safe_id[:80]}.txt"
        metadata_lines = [
            f"[DOCUMENT_ID: {document.id}]",
            f"[SOURCE_TYPE: {document.source_type}]",
        ]
        for key, value in sorted(document.metadata.items()):
            if isinstance(value, (str, int, float, bool)) and len(str(value)) < 300:
                metadata_lines.append(f"[{str(key).upper()}: {value}]")
        path.write_text("\n".join(metadata_lines) + "\n\n" + document.text, encoding="utf-8")


class StandardGraphRagPipeline:
    """GraphRAG project builder/query wrapper for canonical benchmark docs."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        api_key: str,
        query_method: str = "local",
        response_type: str = "Single sentence",
        force_rebuild: bool = False,
        index_method: str = "standard",
        extract_graph_max_gleanings: int = 0,
    ):
        self.workspace_dir = workspace_dir
        self.api_key = api_key
        self.query_method = query_method
        self.response_type = response_type
        self.force_rebuild = force_rebuild
        self.index_method = index_method
        self.extract_graph_max_gleanings = extract_graph_max_gleanings
        self._search_engine: Any | None = None

    def build(self, documents: list[BenchmarkDocument]) -> None:
        input_dir = self.workspace_dir / "input"
        write_graphrag_input(documents, input_dir, clean=self.force_rebuild)
        create_graphrag_config(
            self.workspace_dir,
            api_key=self.api_key,
            force=self.force_rebuild,
            extract_graph_max_gleanings=self.extract_graph_max_gleanings,
        )
        if self.force_rebuild or not has_graphrag_index(self.workspace_dir):
            ok = run_graphrag_index(
                self.workspace_dir,
                method=self.index_method,
                use_cache=not self.force_rebuild,
            )
            if not ok:
                raise RuntimeError(f"GraphRAG indexing failed for workspace: {self.workspace_dir}")
        self._search_engine = asyncio.run(self._build_search_engine())

    async def _load_dataframes(self) -> tuple[Any, dict[str, Any]]:
        from graphrag.config.load_config import load_config
        from graphrag.data_model.data_reader import DataReader
        from graphrag_storage import create_storage
        from graphrag_storage.tables.table_provider_factory import create_table_provider

        config = load_config(root_dir=self.workspace_dir)
        storage_obj = create_storage(config.output_storage)
        table_provider = create_table_provider(config.table_provider, storage=storage_obj)
        reader = DataReader(table_provider)

        if self.query_method == "basic":
            required = ["text_units"]
            optional: list[str] = []
        elif self.query_method == "local":
            required = ["communities", "community_reports", "text_units", "relationships", "entities"]
            optional = ["covariates"]
        else:
            raise RuntimeError(f"Unsupported GraphRAG query method: {self.query_method}")

        dataframes: dict[str, Any] = {}
        for name in required:
            dataframes[name] = await getattr(reader, name)()
        for name in optional:
            dataframes[name] = await getattr(reader, name)() if await table_provider.has(name) else None
        return config, dataframes

    async def _build_search_engine(self) -> Any:
        from graphrag.query.factory import get_basic_search_engine, get_local_search_engine
        from graphrag.query.indexer_adapters import (
            read_indexer_covariates,
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_reports,
            read_indexer_text_units,
        )
        from graphrag.utils.api import get_embedding_store, load_search_prompt

        config, dataframes = await self._load_dataframes()
        if self.query_method == "basic":
            embedding_store = get_embedding_store(config.vector_store, "text_unit_text")
            prompt = load_search_prompt(config.basic_search.prompt)
            return get_basic_search_engine(
                config=config,
                text_units=read_indexer_text_units(dataframes["text_units"]),
                text_unit_embeddings=embedding_store,
                response_type=self.response_type,
                system_prompt=prompt,
                callbacks=[],
            )

        if self.query_method == "local":
            description_embedding_store = get_embedding_store(config.vector_store, "entity_description")
            prompt = load_search_prompt(config.local_search.prompt)
            engine = get_local_search_engine(
                config=config,
                reports=read_indexer_reports(
                    dataframes["community_reports"],
                    dataframes["communities"],
                    community_level=cfg.GRAPHRAG_LOCAL_REPORT_COMMUNITY_LEVEL,
                ),
                text_units=read_indexer_text_units(dataframes["text_units"]),
                entities=read_indexer_entities(
                    dataframes["entities"],
                    dataframes["communities"],
                    community_level=None,
                ),
                relationships=read_indexer_relationships(dataframes["relationships"]),
                covariates={
                    "claims": (
                        read_indexer_covariates(dataframes["covariates"])
                        if dataframes.get("covariates") is not None
                        else []
                    )
                },
                description_embedding_store=description_embedding_store,
                response_type=self.response_type,
                system_prompt=prompt,
                callbacks=[],
            )
            engine.context_builder_params.update(_local_graphrag_search_kwargs())
            return engine

        raise RuntimeError(f"Unsupported GraphRAG query method: {self.query_method}")

    @staticmethod
    def _normalize_response(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, (dict, list)):
            return json.dumps(response, ensure_ascii=False).strip()
        return str(response).strip()

    @staticmethod
    def _serialize_contexts(search_result: Any) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        context_text = getattr(search_result, "context_text", None)
        if isinstance(context_text, dict):
            for rank, (section_name, text_value) in enumerate(context_text.items(), start=1):
                text = str(text_value).strip()
                if text:
                    contexts.append(
                        RetrievedContext(
                            id=f"{section_name}:{rank}",
                            text=text,
                            score=None,
                            rank=rank,
                            source_type=str(section_name),
                            metadata={},
                        )
                    )
            return contexts

        if isinstance(context_text, list):
            for rank, text_value in enumerate(context_text, start=1):
                text = str(text_value).strip()
                if text:
                    contexts.append(
                        RetrievedContext(
                            id=f"context:{rank}",
                            text=text,
                            score=None,
                            rank=rank,
                            source_type="context_chunk",
                            metadata={},
                        )
                    )
            return contexts

        text = str(context_text or "").strip()
        if text:
            contexts.append(
                RetrievedContext(
                    id="context:1",
                    text=text,
                    score=None,
                    rank=1,
                    source_type="context_chunk",
                    metadata={},
                )
            )
        return contexts

    async def _search(self, question: BenchmarkQuestion) -> Any:
        if self._search_engine is None:
            self._search_engine = await self._build_search_engine()
        return await self._search_engine.search(question.question)

    async def _query_async(self, question: BenchmarkQuestion) -> SystemPrediction:
        started = time.perf_counter()
        try:
            search_result = await self._search(question)
            answer = self._normalize_response(getattr(search_result, "response", ""))
            contexts = self._serialize_contexts(search_result)
            return SystemPrediction(
                question_id=question.question_id,
                system_name="standard_graphrag",
                predicted_answer=answer,
                retrieved_contexts=contexts,
                latency_seconds=time.perf_counter() - started,
                metadata={
                    "workspace_dir": str(self.workspace_dir),
                    "query_method": self.query_method,
                    "response_type": self.response_type,
                    "context_count": len(contexts),
                },
            )
        except Exception as exc:
            return SystemPrediction(
                question_id=question.question_id,
                system_name="standard_graphrag",
                predicted_answer="ERROR",
                retrieved_contexts=[],
                latency_seconds=time.perf_counter() - started,
                error=str(exc),
                metadata={"workspace_dir": str(self.workspace_dir), "query_method": self.query_method},
            )

    def query(self, question: BenchmarkQuestion) -> SystemPrediction:
        return asyncio.run(self._query_async(question))

    def query_many(self, questions: list[BenchmarkQuestion], *, max_concurrent: int = 1) -> list[SystemPrediction]:
        if max_concurrent <= 1 or len(questions) <= 1:
            return [self.query(question) for question in questions]

        async def run_all() -> list[SystemPrediction]:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def guarded_query(question: BenchmarkQuestion) -> SystemPrediction:
                async with semaphore:
                    return await self._query_async(question)

            return list(await asyncio.gather(*(guarded_query(question) for question in questions)))

        return asyncio.run(run_all())


def default_graphrag_api_key() -> str:
    return cfg.LOCAL_LLM_API_KEY if cfg.MODEL_BACKEND == "local_openai" else cfg.GOOGLE_API_KEY
