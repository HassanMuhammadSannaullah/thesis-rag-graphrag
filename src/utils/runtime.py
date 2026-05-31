"""Runtime helpers for choosing the correct Python/CLI executable."""
from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_CONDA_ENV_NAME = "thesis_rag_gpu"


def preferred_conda_env_name() -> str:
    return os.getenv("THESIS_RAG_CONDA_ENV", DEFAULT_CONDA_ENV_NAME).strip() or DEFAULT_CONDA_ENV_NAME


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
