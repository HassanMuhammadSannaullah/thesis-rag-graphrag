"""Local model registry for thesis experiment planning and run metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import settings as cfg


@dataclass
class ModelSpec:
    model_id: str
    role: str
    tier: str
    quantization: str
    family: str


GENERATION_MODELS: list[ModelSpec] = [
    ModelSpec("Qwen/Qwen2.5-14B-Instruct", role="generation", tier="A", quantization="fp16", family="qwen"),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct", role="generation", tier="A", quantization="int4", family="qwen"),
    ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", role="generation", tier="A", quantization="fp16", family="mistral"),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", role="generation", tier="A", quantization="int4", family="llama"),
]


EMBEDDING_MODELS: list[ModelSpec] = [
    ModelSpec("intfloat/e5-base-v2", role="embedding", tier="A", quantization="fp16", family="e5"),
    ModelSpec("BAAI/bge-base-en-v1.5", role="embedding", tier="A", quantization="fp16", family="bge"),
    ModelSpec("intfloat/e5-small-v2", role="embedding", tier="B", quantization="fp16", family="e5"),
    ModelSpec("BAAI/bge-small-en-v1.5", role="embedding", tier="B", quantization="fp16", family="bge"),
    ModelSpec("thenlper/gte-base", role="embedding", tier="B", quantization="fp16", family="gte"),
]


def _spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    payload = asdict(spec)
    local_hint = spec.model_id.replace("/", "--")
    payload["local_cache_hint"] = str(Path(cfg.LOCAL_MODELS_DIR) / f"models--{local_hint}")
    return payload


def build_local_model_registry_snapshot() -> dict[str, Any]:
    """Return a hardware-aware model registry snapshot for experiment metadata."""
    return {
        "hardware": {
            "gpu_vram_gb": cfg.LOCAL_MODEL_VRAM_GB,
            "ram_gb": cfg.LOCAL_MODEL_RAM_GB,
            "storage_budget_gb": cfg.LOCAL_MODEL_STORAGE_BUDGET_GB,
        },
        "policies": {
            "large_model_quantization": "Prefer fp16 for 14B on 24GB VRAM (RTX 4090); int4 for 7B/8B",
            "pilot_first": "Run small question subsets before larger runs",
            "storage_cap": "Keep active models and caches within configured storage budget",
        },
        "generation_models": [_spec_to_dict(spec) for spec in GENERATION_MODELS],
        "embedding_models": [_spec_to_dict(spec) for spec in EMBEDDING_MODELS],
        "active_generation_model": cfg.LOCAL_GENERATION_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL,
        "active_embedding_model": cfg.LOCAL_EMBEDDING_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.EMBEDDING_MODEL,
    }
