"""Prepare local model cache for thesis experiments.

This script inspects or downloads generation/embedding models from the
registry based on hardware-aware tiers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings as cfg
from src.config.model_registry import EMBEDDING_MODELS, GENERATION_MODELS, ModelSpec

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("huggingface_hub is required. Install with: pip install huggingface_hub") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-tiers", default="A,B", help="Comma-separated generation tiers")
    parser.add_argument("--embedding-tiers", default="A,B", help="Comma-separated embedding tiers")
    parser.add_argument("--max-models", type=int, default=0, help="0 means no limit")
    parser.add_argument("--download", action="store_true", help="Actually download missing models")
    parser.add_argument("--allow-pattern", action="append", default=[], help="Extra download allow pattern")
    return parser.parse_args()


def _split_tiers(value: str) -> set[str]:
    return {part.strip().upper() for part in value.split(",") if part.strip()}


def _selected_specs(generation_tiers: set[str], embedding_tiers: set[str]) -> list[ModelSpec]:
    selected: list[ModelSpec] = []
    selected.extend(spec for spec in GENERATION_MODELS if spec.tier in generation_tiers)
    selected.extend(spec for spec in EMBEDDING_MODELS if spec.tier in embedding_tiers)
    return selected


def _cache_path_for(model_id: str) -> Path:
    return cfg.LOCAL_MODELS_DIR / f"models--{model_id.replace('/', '--')}"


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _default_allow_patterns(spec: ModelSpec) -> list[str]:
    patterns = [
        "*.json",
        "*.txt",
        "tokenizer.*",
        "special_tokens_map.json",
        "generation_config.json",
        "*.safetensors",
        "*.model",
        "*.bin",
    ]
    if spec.quantization == "int4":
        patterns.extend(["*gptq*", "*awq*", "*gguf*"])
    return patterns


def main() -> None:
    args = parse_args()
    gen_tiers = _split_tiers(args.generation_tiers)
    emb_tiers = _split_tiers(args.embedding_tiers)
    specs = _selected_specs(gen_tiers, emb_tiers)
    if args.max_models > 0:
        specs = specs[: args.max_models]

    print("=" * 72)
    print("LOCAL MODEL PREP")
    print("=" * 72)
    print(f"Model backend: {cfg.MODEL_BACKEND}")
    print(f"Hardware hint: VRAM={cfg.LOCAL_MODEL_VRAM_GB}GB RAM={cfg.LOCAL_MODEL_RAM_GB}GB")
    print(f"Storage budget: {cfg.LOCAL_MODEL_STORAGE_BUDGET_GB}GB")
    print(f"Local model cache: {cfg.LOCAL_MODELS_DIR}")
    print(f"Selected models: {len(specs)}")
    print(f"Disk free now: {_disk_free_gb(cfg.PROJECT_ROOT):.1f}GB")

    rows: list[dict] = []
    for idx, spec in enumerate(specs, start=1):
        cache_dir = _cache_path_for(spec.model_id)
        exists = cache_dir.exists()
        row = {
            "index": idx,
            "model_id": spec.model_id,
            "role": spec.role,
            "tier": spec.tier,
            "quantization": spec.quantization,
            "exists": exists,
            "cache_path": str(cache_dir),
            "status": "present" if exists else "missing",
            "downloaded_path": None,
            "error": None,
        }
        rows.append(row)
        print(f"[{idx:02d}] {spec.role:<10} {spec.tier} {spec.quantization:<5} {spec.model_id} -> {row['status']}")

    if args.download:
        print("\nStarting downloads for missing models ...")
        for row, spec in zip(rows, specs):
            if row["exists"]:
                continue
            free_gb = _disk_free_gb(cfg.PROJECT_ROOT)
            if free_gb < 10:
                row["error"] = f"Low disk space: {free_gb:.1f}GB free"
                print(f"  Skip {spec.model_id}: {row['error']}")
                continue
            allow_patterns = _default_allow_patterns(spec) + args.allow_pattern
            try:
                local_path = snapshot_download(
                    repo_id=spec.model_id,
                    cache_dir=str(cfg.LOCAL_MODELS_DIR),
                    local_files_only=False,
                    allow_patterns=allow_patterns,
                    resume_download=True,
                )
                row["status"] = "downloaded"
                row["downloaded_path"] = local_path
                print(f"  Downloaded {spec.model_id}")
            except Exception as exc:  # pragma: no cover
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"  Failed {spec.model_id}: {exc}")

    out = {
        "hardware": {
            "vram_gb": cfg.LOCAL_MODEL_VRAM_GB,
            "ram_gb": cfg.LOCAL_MODEL_RAM_GB,
            "storage_budget_gb": cfg.LOCAL_MODEL_STORAGE_BUDGET_GB,
        },
        "selection": {
            "generation_tiers": sorted(gen_tiers),
            "embedding_tiers": sorted(emb_tiers),
            "max_models": args.max_models,
            "download": args.download,
        },
        "models": rows,
    }
    out_path = cfg.METRICS_DIR / "local_model_registry_snapshot.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved snapshot: {out_path}")


if __name__ == "__main__":
    main()
