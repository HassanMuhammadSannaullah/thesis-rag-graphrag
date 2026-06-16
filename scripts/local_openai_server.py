"""
Minimal OpenAI-compatible local server backed by Hugging Face models.
"""
import gc
import json
import math
import os
import sys
import time
import traceback
from threading import Lock
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from json_repair import repair_json
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings as cfg


os.environ.setdefault("HF_HOME", str(cfg.LOCAL_MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cfg.LOCAL_MODELS_DIR))


app = FastAPI(title="Local OpenAI-Compatible Model Server")
_generation_lock = Lock()
_embedding_lock = Lock()
_generation_tokenizer = None
_generation_model = None
_generation_model_name = None
_embedding_tokenizer = None
_embedding_model = None
_embedding_model_name = None
_device = None
_log_path = None
_generation_load_lock = Lock()
_embedding_load_lock = Lock()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = Field(default=256, alias="max_tokens")
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] | str


def _resolve_device() -> str:
    if cfg.LOCAL_MODEL_DEVICE != "auto":
        return cfg.LOCAL_MODEL_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype_for_device(device: str):
    if device == "cuda":
        return torch.float16
    return torch.float32


def _cache_dir_arg() -> str:
    """
    Keep Hugging Face cache paths relative when the server is launched from the
    project tree. SentencePiece on Windows can fail on absolute paths containing
    non-ASCII characters, even when the file exists.
    """
    cache_dir = cfg.LOCAL_MODELS_DIR.resolve()
    cwd = Path.cwd().resolve()
    project_root = cfg.PROJECT_ROOT.resolve()
    if cwd == project_root or project_root in cwd.parents:
        return os.path.relpath(cache_dir, cwd)
    return str(cache_dir)


def _load_tokenizer(model_name: str):
    tokenizer_kwargs = {
        "cache_dir": _cache_dir_arg(),
        "trust_remote_code": cfg.LOCAL_MODEL_TRUST_REMOTE_CODE,
    }
    try:
        return AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    except OSError as exc:
        if any(ord(char) > 127 for char in str(cfg.LOCAL_MODELS_DIR)):
            print(
                "Fast tokenizer load failed with a non-ASCII cache path; "
                "retrying with use_fast=False."
            )
            return AutoTokenizer.from_pretrained(model_name, use_fast=False, **tokenizer_kwargs)
        raise exc


def _use_device_map() -> bool:
    return cfg.LOCAL_MODEL_DEVICE_MAP.strip().lower() not in {"", "none"}


def _release_model(model) -> None:
    if model is None:
        return
    if not _use_device_map():
        try:
            model.to("cpu")
        except Exception:
            pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_generation_model(model_name: str | None = None):
    global _generation_model, _generation_tokenizer, _generation_model_name, _device
    model_name = model_name or cfg.LOCAL_GENERATION_MODEL
    if _generation_model is not None and _generation_tokenizer is not None and _generation_model_name == model_name:
        return
    with _generation_load_lock:
        if _generation_model is not None and _generation_tokenizer is not None and _generation_model_name == model_name:
            return
        _load_generation_model_unlocked(model_name)


def _load_generation_model_unlocked(model_name: str) -> None:
    global _generation_model, _generation_tokenizer, _generation_model_name, _device
    _device = _resolve_device()
    print(f"Loading generation model: {model_name} on {_device}")
    _release_model(_generation_model)
    _generation_model = None
    _generation_tokenizer = None
    _generation_model_name = None
    _generation_tokenizer = _load_tokenizer(model_name)
    if _generation_tokenizer.pad_token is None:
        _generation_tokenizer.pad_token = _generation_tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": _dtype_for_device(_device),
        "low_cpu_mem_usage": True,
        "trust_remote_code": cfg.LOCAL_MODEL_TRUST_REMOTE_CODE,
        "cache_dir": _cache_dir_arg(),
    }
    if _use_device_map():
        model_kwargs["device_map"] = cfg.LOCAL_MODEL_DEVICE_MAP

    _generation_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if not _use_device_map():
        _generation_model.to(_device)
    _generation_model.eval()
    _generation_model_name = model_name


def _load_embedding_model(model_name: str | None = None):
    global _embedding_model, _embedding_tokenizer, _embedding_model_name, _device
    model_name = model_name or cfg.LOCAL_EMBEDDING_MODEL
    if _embedding_model is not None and _embedding_tokenizer is not None and _embedding_model_name == model_name:
        return
    with _embedding_load_lock:
        if _embedding_model is not None and _embedding_tokenizer is not None and _embedding_model_name == model_name:
            return
        _load_embedding_model_unlocked(model_name)


def _load_embedding_model_unlocked(model_name: str) -> None:
    global _embedding_model, _embedding_tokenizer, _embedding_model_name, _device
    _device = _resolve_device()
    print(f"Loading embedding model: {model_name} on {_device}")
    _release_model(_embedding_model)
    _embedding_model = None
    _embedding_tokenizer = None
    _embedding_model_name = None
    _embedding_tokenizer = _load_tokenizer(model_name)
    _embedding_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=_dtype_for_device(_device),
        trust_remote_code=cfg.LOCAL_MODEL_TRUST_REMOTE_CODE,
        cache_dir=_cache_dir_arg(),
    )
    _embedding_model.to(_device)
    _embedding_model.eval()
    _embedding_model_name = model_name


def _request_log_path() -> Path:
    global _log_path
    if _log_path is None:
        configured = os.getenv("LOCAL_SERVER_LOG_PATH", "").strip()
        if configured:
            _log_path = Path(configured)
        else:
            _log_path = cfg.LOGS_DIR / "local_openai_server.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
    return _log_path


def _log_event(event_type: str, payload: dict) -> None:
    record = {
        "ts": int(time.time()),
        "event": event_type,
        **payload,
    }
    try:
        with _request_log_path().open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    if hasattr(_generation_tokenizer, "apply_chat_template"):
        return _generation_tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = []
    for message in messages:
        parts.append(f"{message.role.upper()}: {message.content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vector, p=2, dim=1)


def _looks_like_query(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower.startswith("query:"):
        return True
    if lower.startswith("passage:"):
        return False
    if stripped.endswith("?"):
        return True
    if "\n" in stripped:
        return False
    query_starts = (
        "who ",
        "what ",
        "when ",
        "where ",
        "why ",
        "how ",
        "which ",
        "whom ",
        "whose ",
        "name ",
        "in what ",
        "what is ",
        "what was ",
        "what year ",
        "who became ",
    )
    if lower.startswith(query_starts):
        return True
    return False


def _format_embedding_text(text: str, *, query_mode: bool = False) -> str:
    stripped = text.strip()
    lower = stripped.lower()
    if lower.startswith("query:") or lower.startswith("passage:"):
        return stripped
    prefix = "query" if query_mode else "passage"
    return f"{prefix}: {stripped}"


# Entity types that the model sometimes inserts as junk words in relationship fields
_GRAPHRAG_ENTITY_TYPE_WORDS = frozenset({"entity", "organization", "person", "geo", "event"})


def _normalize_graphrag_extraction(text: str) -> str:
    """
    Qwen models produce a close-but-wrong format for GraphRAG entity extraction:
      (entity|Name||type)<|>(relationship|Src|entity|Tgt|type|desc|strength)<|>|COMPLETE|
    This converts it to the exact GraphRAG-expected format:
      ("entity"<|>NAME<|>TYPE<|>desc)
      ##
      ("relationship"<|>SRC<|>TGT<|>desc<|>strength)
      <|COMPLETE|>
    """
    # Already in correct GraphRAG format — don't touch it
    if '("entity"<|>' in text:
        return text
    # Not an entity extraction response at all
    if '(entity|' not in text and '(relationship|' not in text:
        return text

    # Strip COMPLETE marker variants at the end
    stripped = text.strip()
    for marker in ("<|COMPLETE|>", "|COMPLETE|", "<|COMPLETE", "COMPLETE|"):
        if stripped.endswith(marker):
            stripped = stripped[: -len(marker)].rstrip("<|> \n")
            break

    # Split into individual records — the model uses <|> as record separator
    raw_parts = stripped.split("<|>")

    output_records: list[str] = []
    seen_entity_keys: set[tuple[str, str]] = set()

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        # Remove outer parentheses
        inner = part
        if inner.startswith("("):
            inner = inner[1:]
        if inner.endswith(")"):
            inner = inner[:-1]

        fields = inner.split("|")
        if not fields:
            continue

        record_type = fields[0].strip().lower()

        if record_type == "entity":
            if len(fields) < 3:
                continue
            name = fields[1].strip().upper()
            if not name:
                continue
            # Handle both (entity|name|type) and (entity|name||type)
            if len(fields) >= 4 and fields[2].strip() == "":
                entity_type = fields[3].strip().upper()
            else:
                entity_type = fields[2].strip().upper()
            if not entity_type:
                entity_type = "ENTITY"

            key = (name, entity_type)
            if key in seen_entity_keys:
                continue
            seen_entity_keys.add(key)

            desc = f"{name.title()} is a {entity_type.lower()}"
            output_records.append(f'("entity"<|>{name}<|>{entity_type}<|>{desc})')

        elif record_type == "relationship":
            if len(fields) < 4:
                continue
            source = fields[1].strip().upper()
            # If fields[2] is a junk word the model inserts, skip it
            idx = 2
            if fields[idx].strip().lower() in _GRAPHRAG_ENTITY_TYPE_WORDS:
                idx += 1
            if idx >= len(fields):
                continue
            target = fields[idx].strip().upper()
            idx += 1
            # Skip another junk type word if present
            if idx < len(fields) and fields[idx].strip().lower() in _GRAPHRAG_ENTITY_TYPE_WORDS:
                idx += 1
            description = fields[idx].strip() if idx < len(fields) else f"{source} is related to {target}"
            idx += 1
            strength_raw = fields[idx].strip().rstrip(")") if idx < len(fields) else "5"
            try:
                strength = str(int(float(strength_raw)))
            except (ValueError, TypeError):
                strength = "5"

            if source and target:
                output_records.append(f'("relationship"<|>{source}<|>{target}<|>{description}<|>{strength})')

    if not output_records:
        return text  # Couldn't parse anything — return unchanged

    result = "\n##\n".join(output_records)
    result += "\n<|COMPLETE|>"
    return result


def _clean_json_like_response(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = stripped[first_brace:last_brace + 1].strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                return repair_json(candidate, return_objects=False).strip()
            except Exception:
                return candidate
    return stripped


def _generate_chat_text(request: ChatCompletionRequest) -> tuple[str, int, int, str]:
    _load_generation_model(request.model)
    prompt = _messages_to_prompt(request.messages)

    with _generation_lock:
        inputs = _generation_tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(_generation_model.device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            generated = _generation_model.generate(
                **inputs,
                max_new_tokens=min(request.max_tokens, cfg.LOCAL_MAX_NEW_TOKENS),
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=_generation_tokenizer.pad_token_id,
                eos_token_id=_generation_tokenizer.eos_token_id,
            )

        new_tokens = generated[0][input_tokens:]
        text = _generation_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = _normalize_graphrag_extraction(text)
        text = _clean_json_like_response(text)
        completion_tokens = int(new_tokens.shape[0])

    return text, input_tokens, completion_tokens, prompt


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "generation_model": cfg.LOCAL_GENERATION_MODEL,
        "embedding_model": cfg.LOCAL_EMBEDDING_MODEL,
        "device": _resolve_device(),
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    started_at = time.time()
    try:
        text, input_tokens, completion_tokens, prompt = _generate_chat_text(request)
    except Exception as exc:
        _log_event(
            "chat_completion_error",
            {
                "model": request.model,
                "temperature": request.temperature,
                "requested_max_tokens": request.max_tokens,
                "error": repr(exc),
                "traceback": traceback.format_exc()[-4000:],
                "duration_sec": round(time.time() - started_at, 3),
            },
        )
        raise

    _log_event(
        "chat_completion",
        {
            "model": request.model,
            "temperature": request.temperature,
            "requested_max_tokens": request.max_tokens,
            "served_max_tokens": min(request.max_tokens, cfg.LOCAL_MAX_NEW_TOKENS),
            "prompt_chars": len(prompt),
            "prompt_preview": prompt[:3000],
            "response_chars": len(text),
            "response_preview": text[:1500],
            "duration_sec": round(time.time() - started_at, 3),
        },
    )

    if request.stream:
        chunk_id = f"chatcmpl-local-{int(time.time() * 1000)}"

        def stream_chunks():
            first_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            }
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_chunks(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-local-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": input_tokens + completion_tokens,
        },
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest) -> dict:
    texts = request.input if isinstance(request.input, list) else [request.input]
    query_mode = len(texts) == 1 and _looks_like_query(str(texts[0]))
    formatted = [_format_embedding_text(text, query_mode=query_mode) for text in texts]
    started_at = time.time()

    try:
        _load_embedding_model(request.model)

        with _embedding_lock:
            encoded = _embedding_tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(_embedding_model.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = _embedding_model(**encoded)
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                normalized = _normalize(pooled).float().cpu()
    except Exception as exc:
        _log_event(
            "embedding_error",
            {
                "model": request.model,
                "count": len(texts),
                "query_mode": query_mode,
                "max_chars": max((len(text) for text in texts), default=0),
                "error": repr(exc),
                "traceback": traceback.format_exc()[-4000:],
                "duration_sec": round(time.time() - started_at, 3),
            },
        )
        raise

    data = []
    total_tokens = 0
    for idx, (text, vector) in enumerate(zip(texts, normalized)):
        token_count = max(1, math.ceil(len(text.split()) * 1.3))
        total_tokens += token_count
        data.append(
            {
                "object": "embedding",
                "index": idx,
                "embedding": vector.tolist(),
            }
        )

    _log_event(
        "embedding",
        {
            "model": request.model,
            "count": len(texts),
            "query_mode": query_mode,
            "max_chars": max((len(text) for text in texts), default=0),
            "duration_sec": round(time.time() - started_at, 3),
        },
    )

    return {
        "object": "list",
        "model": request.model,
        "data": data,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=cfg.LOCAL_SERVER_HOST,
        port=cfg.LOCAL_SERVER_PORT,
        reload=False,
    )
