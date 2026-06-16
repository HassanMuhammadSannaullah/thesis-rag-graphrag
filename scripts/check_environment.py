"""Check whether the active Python environment can run the benchmark pipeline."""
from __future__ import annotations

import importlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _check_import(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
        return {
            "module": module_name,
            "ok": True,
            "version": getattr(module, "__version__", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "module": module_name,
            "ok": False,
            "version": None,
            "error": str(exc),
        }


def main() -> None:
    checks = [
        _check_import("faiss"),
        _check_import("rank_bm25"),
        _check_import("sentence_transformers"),
        _check_import("llama_index.core"),
        _check_import("graphrag"),
        _check_import("torch"),
    ]

    torch_info = {}
    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}

    payload = {
        "python": sys.executable,
        "python_version": sys.version,
        "imports": checks,
        "torch": torch_info,
        "ready": all(row["ok"] for row in checks),
    }
    print(json.dumps(payload, indent=2))
    if not payload["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
