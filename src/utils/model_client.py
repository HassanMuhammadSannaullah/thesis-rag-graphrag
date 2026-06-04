"""
Shared model client for both Gemini and local OpenAI-compatible backends.
"""
import hashlib
import json
import time
from pathlib import Path

from openai import OpenAI

from src.config import settings as cfg

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None


_gemini_client = None
_openai_client = None


def _cache_path(prefix: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return cfg.GEN_CACHE_DIR / f"{prefix}_{digest}.json"


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _active_generation_model(model: str | None = None) -> str:
    if model:
        return model
    if cfg.MODEL_BACKEND == "local_openai":
        return cfg.LOCAL_GENERATION_MODEL
    return cfg.GENERATION_MODEL


def _active_embedding_model(model: str | None = None) -> str:
    if model:
        return model
    if cfg.MODEL_BACKEND == "local_openai":
        return cfg.LOCAL_EMBEDDING_MODEL
    return cfg.EMBEDDING_MODEL


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if genai is None:
            raise RuntimeError("google-genai is not installed for Gemini backend.")
        _gemini_client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
    return _gemini_client


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=cfg.LOCAL_LLM_API_KEY,
            base_url=cfg.LOCAL_LLM_BASE_URL,
            timeout=60.0,  # Prevent indefinite hangs on network/server issues
        )
    return _openai_client


def _generate_text_gemini(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    client = _get_gemini_client()
    _sleep(cfg.SLEEP_BETWEEN_REQ)
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text or ""
        except Exception as exc:
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s ...")
                _sleep(wait)
            elif "503" in err or "UNAVAILABLE" in err:
                wait = 30 * (attempt + 1)
                print(f"  Model unavailable (503), waiting {wait}s ...")
                _sleep(wait)
            else:
                raise
    raise RuntimeError("generate_text failed after 5 retries")


def _generate_text_local(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    return_reasoning: bool = False,
) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    msg = response.choices[0].message
    content = msg.content or ""
    if return_reasoning:
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            return reasoning.strip()
    return content


def generate_text(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    use_cache: bool = True,
) -> str:
    """Generate text with the configured backend, caching responses to disk."""
    active_model = _active_generation_model(model)
    cache_key = f"{cfg.MODEL_BACKEND}::{active_model}::{temperature}::{max_tokens}::{prompt}"
    cache_path = _cache_path("gen", cache_key)

    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

    if cfg.MODEL_BACKEND == "local_openai":
        text = _generate_text_local(prompt, active_model, temperature, max_tokens)
    else:
        text = _generate_text_gemini(prompt, active_model, temperature, max_tokens)

    cache_path.write_text(
        json.dumps(
            {
                "backend": cfg.MODEL_BACKEND,
                "model": active_model,
                "text": text,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return text


def _embed_texts_gemini(texts: list[str], model: str) -> list[list[float]]:
    client = _get_gemini_client()
    vectors: list[list[float]] = []
    for text in texts:
        _sleep(cfg.SLEEP_BETWEEN_EMB)
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=text,
                )
                vectors.append(response.embeddings[0].values)
                break
            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 30 * (attempt + 1)
                    print(f"  Embedding rate limited, waiting {wait}s ...")
                    _sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError("embed_texts failed after 3 retries")
    return vectors


def _embed_texts_local(texts: list[str], model: str) -> list[list[float]]:
    import requests
    headers = {}
    if cfg.LOCAL_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {cfg.LOCAL_LLM_API_KEY}"
    url = cfg.LOCAL_LLM_BASE_URL.rstrip("/") + "/embeddings"
    response = requests.post(
        url,
        json={"model": model, "input": texts},
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return [row["embedding"] for row in data["data"]]


def embed_texts(
    texts: list[str],
    model: str | None = None,
    use_cache: bool = True,
) -> list[list[float]]:
    """Embed a list of texts using the configured backend."""
    active_model = _active_embedding_model(model)
    results: list[list[float] | None] = [None] * len(texts)
    uncached_items: list[tuple[int, str, Path]] = []

    for idx, text in enumerate(texts):
        cache_key = f"{cfg.MODEL_BACKEND}::{active_model}::emb::{text}"
        cache_path = _cache_path("emb", cache_key)
        if use_cache and cache_path.exists():
            results[idx] = json.loads(cache_path.read_text(encoding="utf-8"))["vec"]
        else:
            uncached_items.append((idx, text, cache_path))

    if uncached_items:
        uncached_texts = [text for _, text, _ in uncached_items]
        if cfg.MODEL_BACKEND == "local_openai":
            vectors = _embed_texts_local(uncached_texts, active_model)
        else:
            vectors = _embed_texts_gemini(uncached_texts, active_model)

        for (idx, _, cache_path), vector in zip(uncached_items, vectors):
            cache_path.write_text(
                json.dumps(
                    {
                        "backend": cfg.MODEL_BACKEND,
                        "model": active_model,
                        "vec": vector,
                    }
                ),
                encoding="utf-8",
            )
            results[idx] = vector

    return [vector for vector in results if vector is not None]


def probe_local_backend(
    generation_model: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, str | int]:
    """
    Force a tiny completion + embedding request against the local OpenAI server.

    The health endpoint only proves the HTTP server is alive; it does not prove
    the underlying generation and embedding models can be loaded successfully.
    Warming both paths up here prevents the first GraphRAG request from paying
    the full cold-start cost or surfacing a model-load failure later.
    """
    if cfg.MODEL_BACKEND != "local_openai":
        return {"status": "skipped", "reason": "backend is not local_openai"}

    active_generation_model = _active_generation_model(generation_model)
    active_embedding_model = _active_embedding_model(embedding_model)

    response = _generate_text_local(
        "This is an LLM connectivity test. Say Hello World",
        active_generation_model,
        temperature=0.1,
        max_tokens=128,
        return_reasoning=True,
    ).strip()
    if not response:
        raise RuntimeError("Local generation probe returned an empty response.")

    vectors = _embed_texts_local(
        ["GraphRAG connectivity probe"],
        active_embedding_model,
    )
    if not vectors or not vectors[0]:
        raise RuntimeError("Local embedding probe returned no vectors.")

    return {
        "status": "ok",
        "generation_model": active_generation_model,
        "embedding_model": active_embedding_model,
        "embedding_dimension": len(vectors[0]),
    }
