"""Runtime helpers for choosing the correct Python/CLI executable."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONDA_ENV_NAME = "thesis_rag_gpu"


def _infer_conda_env_from_python(python_path: Path) -> str | None:
    parts = [part.lower() for part in python_path.parts]
    try:
        envs_idx = parts.index("envs")
    except ValueError:
        return None
    original_parts = python_path.parts
    if envs_idx + 1 < len(original_parts):
        return original_parts[envs_idx + 1]
    return None


def preferred_conda_env_name() -> str:
    return os.getenv("THESIS_RAG_CONDA_ENV", DEFAULT_CONDA_ENV_NAME).strip() or DEFAULT_CONDA_ENV_NAME


def current_conda_env_name() -> str | None:
    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if not conda_prefix:
        return _infer_conda_env_from_python(Path(sys.executable).resolve())
    return Path(conda_prefix).name


def _candidate_python_from_conda_root(conda_root: Path, env_name: str) -> Path:
    if os.name == "nt":
        return conda_root / "envs" / env_name / "python.exe"
    return conda_root / "envs" / env_name / "bin" / "python"


def resolve_project_python() -> Path:
    """
    Prefer the dedicated project conda env when it exists.

    This keeps subprocesses on the same runtime even if the parent command was
    launched with a different `python` on PATH.
    """
    current = Path(sys.executable).resolve()
    env_name = preferred_conda_env_name()

    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if conda_prefix and Path(conda_prefix).name.lower() == env_name.lower():
        return current

    candidates: list[Path] = []

    conda_exe = os.getenv("CONDA_EXE", "").strip()
    if conda_exe:
        conda_exe_path = Path(conda_exe).resolve()
        if conda_exe_path.parent.name.lower() == "scripts":
            candidates.append(_candidate_python_from_conda_root(conda_exe_path.parent.parent, env_name))
        else:
            candidates.append(_candidate_python_from_conda_root(conda_exe_path.parent, env_name))

    if os.name == "nt":
        candidates.extend(
            [
                current.parent.parent / "envs" / env_name / "python.exe",
                current.parent / "envs" / env_name / "python.exe",
            ]
        )
    else:
        candidates.extend(
            [
                current.parent.parent / "envs" / env_name / "bin" / "python",
                current.parent / "envs" / env_name / "bin" / "python",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return current


def resolve_graphrag_cli() -> str:
    python_executable = resolve_project_python()
    python_dir = python_executable.parent
    windows_candidate = python_dir / "Scripts" / "graphrag.exe"
    posix_candidate = python_dir / "graphrag"
    if windows_candidate.exists():
        return str(windows_candidate)
    if posix_candidate.exists():
        return str(posix_candidate)
    return "graphrag"


def _round_gb(value_bytes: int | float | None) -> float | None:
    if value_bytes is None:
        return None
    return round(float(value_bytes) / (1024**3), 2)


def detect_runtime_environment() -> dict[str, Any]:
    current_python = Path(sys.executable).resolve()
    resolved_project_python = resolve_project_python()
    current_env = current_conda_env_name()
    preferred_env = preferred_conda_env_name()
    conda_prefix = os.getenv("CONDA_PREFIX", "").strip() or None
    conda_exe = os.getenv("CONDA_EXE", "").strip() or None
    return {
        "current_python": str(current_python),
        "resolved_project_python": str(resolved_project_python),
        "current_conda_env": current_env,
        "preferred_conda_env": preferred_env,
        "using_preferred_conda_env": (current_env or "").lower() == preferred_env.lower(),
        "current_python_matches_resolved": current_python == resolved_project_python,
        "conda_prefix": conda_prefix,
        "conda_exe": conda_exe,
    }


def detect_hardware_snapshot(*, disk_path: str | Path | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        snapshot["ram_total_gb"] = _round_gb(vm.total)
        snapshot["ram_available_gb"] = _round_gb(vm.available)
    except Exception as exc:  # pragma: no cover - environment-dependent
        snapshot["ram_error"] = repr(exc)

    if disk_path is not None:
        try:
            total, _used, free = shutil.disk_usage(str(disk_path))
            snapshot["disk_total_gb"] = _round_gb(total)
            snapshot["disk_free_gb"] = _round_gb(free)
        except Exception as exc:  # pragma: no cover - environment-dependent
            snapshot["disk_error"] = repr(exc)

    try:
        import torch  # type: ignore

        snapshot["torch_version"] = getattr(torch, "__version__", None)
        snapshot["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        snapshot["torch_cuda_available"] = cuda_available
        snapshot["gpu_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available and torch.cuda.device_count() > 0:
            device_properties = torch.cuda.get_device_properties(0)
            snapshot["primary_gpu_name"] = device_properties.name
            snapshot["primary_gpu_total_memory_gb"] = _round_gb(device_properties.total_memory)
            snapshot["suggested_local_model_device"] = "cuda"
        else:
            snapshot["primary_gpu_name"] = None
            snapshot["primary_gpu_total_memory_gb"] = None
            snapshot["suggested_local_model_device"] = "cpu"
    except Exception as exc:  # pragma: no cover - environment-dependent
        snapshot["torch_error"] = repr(exc)

    return snapshot
