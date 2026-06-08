"""Local model registry for thesis experiment planning and run metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from src.config import settings as cfg
from src.utils.runtime import detect_hardware_snapshot, detect_runtime_environment


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
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", role="generation", tier="A", quantization="fp16", family="qwen"),
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


def get_active_model_metadata() -> dict[str, Any]:
    """
    Get metadata for currently configured models.
    
    Returns model names, families, and configuration for experiment tracking.
    """
    from src.config.model_variants import ALL_VARIANTS
    
    # Try to find matching variant
    matching_variant = None
    for variant in ALL_VARIANTS.values():
        if (variant.generation_model == cfg.LOCAL_GENERATION_MODEL and
            variant.embedding_model == cfg.LOCAL_EMBEDDING_MODEL):
            matching_variant = variant
            break
    
    metadata = {
        "backend": cfg.MODEL_BACKEND,
        "generation_model": cfg.LOCAL_GENERATION_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL,
        "embedding_model": cfg.LOCAL_EMBEDDING_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.EMBEDDING_MODEL,
        "graphrag_index_model": cfg.LOCAL_GRAPHRAG_INDEX_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL,
    }
    
    if matching_variant:
        metadata["variant_name"] = matching_variant.name
        metadata["model_family"] = matching_variant.model_family
        metadata["model_size_b"] = matching_variant.model_size_b
        metadata["quantization"] = matching_variant.quantization
        metadata["vram_requirement_gb"] = matching_variant.vram_requirement_gb
    else:
        metadata["variant_name"] = "custom"
        metadata["model_family"] = "unknown"
    
    return metadata


def build_experiment_metadata(
    experiment_name: str = "",
    additional_metadata: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Build complete experiment metadata including hardware, models, and runtime info.
    
    Args:
        experiment_name: Name of the experiment
        additional_metadata: Additional metadata to include
    
    Returns:
        Complete metadata dict for experiment tracking
    """
    detected_hardware = detect_hardware_snapshot(disk_path=cfg.PROJECT_ROOT)
    runtime_env = detect_runtime_environment()
    model_metadata = get_active_model_metadata()
    
    metadata = {
        "experiment_name": experiment_name,
        "hardware": {
            "gpu_vram_gb": detected_hardware.get("primary_gpu_total_memory_gb") or cfg.LOCAL_MODEL_VRAM_GB,
            "gpu_name": detected_hardware.get("primary_gpu_name", "unknown"),
            "cuda_available": detected_hardware.get("cuda_available", False),
            "ram_gb": detected_hardware.get("ram_total_gb") or cfg.LOCAL_MODEL_RAM_GB,
            "cpu_count": detected_hardware.get("cpu_count", 0),
        },
        "runtime": {
            "python_executable": runtime_env.get("python_executable"),
            "conda_env": runtime_env.get("conda_env"),
            "cuda_version": detected_hardware.get("cuda_version"),
            "torch_version": detected_hardware.get("torch_version"),
            "torch_cuda_available": detected_hardware.get("torch_cuda_available", False),
        },
        "models": model_metadata,
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return metadata


def build_local_model_registry_snapshot() -> dict[str, Any]:
    """Return a hardware-aware model registry snapshot for experiment metadata."""
    detected_hardware = detect_hardware_snapshot(disk_path=cfg.PROJECT_ROOT)
    configured_hardware = {
        "gpu_vram_gb": cfg.LOCAL_MODEL_VRAM_GB,
        "ram_gb": cfg.LOCAL_MODEL_RAM_GB,
        "storage_budget_gb": cfg.LOCAL_MODEL_STORAGE_BUDGET_GB,
    }
    return {
        "hardware": {
            "gpu_vram_gb": detected_hardware.get("primary_gpu_total_memory_gb") or cfg.LOCAL_MODEL_VRAM_GB,
            "ram_gb": detected_hardware.get("ram_total_gb") or cfg.LOCAL_MODEL_RAM_GB,
            "storage_budget_gb": cfg.LOCAL_MODEL_STORAGE_BUDGET_GB,
        },
        "configured_hardware": configured_hardware,
        "detected_hardware": detected_hardware,
        "runtime_environment": detect_runtime_environment(),
        "policies": {
            "large_model_quantization": "Prefer fp16 for 14B only when detected VRAM is sufficient; otherwise prefer int4 or smaller local models",
            "pilot_first": "Run small question subsets before larger runs",
            "storage_cap": "Keep active models and caches within configured storage budget",
        },
        "generation_models": [_spec_to_dict(spec) for spec in GENERATION_MODELS],
        "embedding_models": [_spec_to_dict(spec) for spec in EMBEDDING_MODELS],
        "active_generation_model": cfg.LOCAL_GENERATION_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.GENERATION_MODEL,
        "active_embedding_model": cfg.LOCAL_EMBEDDING_MODEL if cfg.MODEL_BACKEND == "local_openai" else cfg.EMBEDDING_MODEL,
    }


def save_experiment_with_metadata(
    predictions: list,
    metrics: dict,
    output_dir: Path,
    metadata: dict,
) -> None:
    """
    Save experiment results with full metadata tracking.
    
    Args:
        predictions: List of prediction objects
        metrics: Evaluation metrics dict
        output_dir: Directory to save results
        metadata: Experiment metadata dict
    """
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(
            [p.to_dict() if hasattr(p, "to_dict") else p for p in predictions],
            f,
            indent=2,
            default=str,
        )
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"  Saved results to {output_dir}")
    print(f"    - predictions.json ({len(predictions)} predictions)")
    print(f"    - metrics.json")
    print(f"    - metadata.json")
