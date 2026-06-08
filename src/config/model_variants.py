"""
Model variants configuration for Phase 3.5.

Defines named model configurations for thesis experiments.
Enables easy model switching without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelVariant:
    """A named model configuration for experiments."""
    
    name: str
    description: str
    
    # Generation model
    generation_model: str
    generation_backend: str = "local_openai"  # "local_openai" or "api"
    
    # Embedding model
    embedding_model: str = "intfloat/e5-base-v2"
    
    # Model metadata
    model_family: str = "unknown"
    model_size_b: float = 0.0  # Model size in billions of parameters
    quantization: str = "fp16"
    
    # Performance hints
    vram_requirement_gb: int = 8
    estimated_speed: str = "medium"  # fast, medium, slow
    
    # Additional metadata
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    
    def to_env_dict(self) -> dict[str, str]:
        """Convert to environment variable dict for subprocess execution."""
        return {
            "MODEL_BACKEND": self.generation_backend,
            "LOCAL_GENERATION_MODEL": self.generation_model,
            "LOCAL_GRAPHRAG_INDEX_MODEL": self.generation_model,
            "LOCAL_EMBEDDING_MODEL": self.embedding_model,
        }
    
    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata dict for experiment tracking."""
        return {
            "variant_name": self.name,
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "model_family": self.model_family,
            "model_size_b": self.model_size_b,
            "quantization": self.quantization,
            "vram_requirement_gb": self.vram_requirement_gb,
            "backend": self.generation_backend,
            "tags": self.tags,
        }


# ==============================================================================
# LLM-focused Variants (testing different generation models)
# ==============================================================================

MISTRAL_7B_E5_BASE = ModelVariant(
    name="mistral_7b_e5_base",
    description="Mistral 7B Instruct v0.3 with E5-base-v2 embeddings",
    generation_model="mistralai/Mistral-7B-Instruct-v0.3",
    embedding_model="intfloat/e5-base-v2",
    model_family="mistral",
    model_size_b=7.0,
    quantization="fp16",
    vram_requirement_gb=16,
    estimated_speed="fast",
    tags=["small", "baseline"],
    notes="Standard small LLM for initial testing",
)

QWEN_3B_E5_BASE = ModelVariant(
    name="qwen_3b_e5_base",
    description="Qwen 2.5 3B Instruct with E5-base-v2 embeddings",
    generation_model="Qwen/Qwen2.5-3B-Instruct",
    embedding_model="intfloat/e5-base-v2",
    model_family="qwen",
    model_size_b=3.0,
    quantization="fp16",
    vram_requirement_gb=8,
    estimated_speed="fast",
    tags=["small", "efficient"],
    notes="Smallest LLM for quick experiments",
)

QWEN_14B_E5_BASE = ModelVariant(
    name="qwen_14b_e5_base",
    description="Qwen 2.5 14B Instruct with E5-base-v2 embeddings",
    generation_model="Qwen/Qwen2.5-14B-Instruct",
    embedding_model="intfloat/e5-base-v2",
    model_family="qwen",
    model_size_b=14.0,
    quantization="fp16",
    vram_requirement_gb=28,
    estimated_speed="medium",
    tags=["medium", "powerful"],
    notes="Medium-sized LLM for better quality",
)

QWEN_7B_E5_BASE = ModelVariant(
    name="qwen_7b_e5_base",
    description="Qwen 2.5 7B Instruct with E5-base-v2 embeddings",
    generation_model="Qwen/Qwen2.5-7B-Instruct",
    embedding_model="intfloat/e5-base-v2",
    model_family="qwen",
    model_size_b=7.0,
    quantization="fp16",
    vram_requirement_gb=16,
    estimated_speed="fast",
    tags=["small", "qwen-family"],
    notes="Qwen 7B alternative to Mistral",
)

# ==============================================================================
# Embedding-focused Variants (testing different embedding models)
# ==============================================================================

MISTRAL_7B_BGE_BASE = ModelVariant(
    name="mistral_7b_bge_base",
    description="Mistral 7B with BGE-base-en-v1.5 embeddings",
    generation_model="mistralai/Mistral-7B-Instruct-v0.3",
    embedding_model="BAAI/bge-base-en-v1.5",
    model_family="mistral",
    model_size_b=7.0,
    quantization="fp16",
    vram_requirement_gb=16,
    estimated_speed="fast",
    tags=["baseline", "bge-embeddings"],
    notes="Test BGE embeddings as alternative to E5",
)

QWEN_7B_BGE_BASE = ModelVariant(
    name="qwen_7b_bge_base",
    description="Qwen 2.5 7B with BGE-base-en-v1.5 embeddings",
    generation_model="Qwen/Qwen2.5-7B-Instruct",
    embedding_model="BAAI/bge-base-en-v1.5",
    model_family="qwen",
    model_size_b=7.0,
    quantization="fp16",
    vram_requirement_gb=16,
    estimated_speed="fast",
    tags=["qwen-family", "bge-embeddings"],
    notes="Qwen 7B with BGE embeddings",
)

MISTRAL_7B_GTE_BASE = ModelVariant(
    name="mistral_7b_gte_base",
    description="Mistral 7B with GTE-base embeddings",
    generation_model="mistralai/Mistral-7B-Instruct-v0.3",
    embedding_model="thenlper/gte-base",
    model_family="mistral",
    model_size_b=7.0,
    quantization="fp16",
    vram_requirement_gb=16,
    estimated_speed="fast",
    tags=["baseline", "gte-embeddings"],
    notes="Test GTE embeddings as third alternative",
)

# ==============================================================================
# Variant Collections for Thesis Phases
# ==============================================================================

# Phase 6.5: Model Ablation - Minimum Required Variants
MINIMUM_ABLATION_VARIANTS = [
    QWEN_3B_E5_BASE,        # Small LLM
    MISTRAL_7B_E5_BASE,     # Medium LLM (different family)
    QWEN_14B_E5_BASE,       # Large LLM
    MISTRAL_7B_BGE_BASE,    # Alternative embedding
]

# Extended ablation for excellence
EXTENDED_ABLATION_VARIANTS = [
    QWEN_3B_E5_BASE,
    MISTRAL_7B_E5_BASE,
    QWEN_7B_E5_BASE,
    QWEN_14B_E5_BASE,
    MISTRAL_7B_BGE_BASE,
    QWEN_7B_BGE_BASE,
    MISTRAL_7B_GTE_BASE,
]

# Quick smoke test variants
SMOKE_TEST_VARIANTS = [
    QWEN_3B_E5_BASE,
    MISTRAL_7B_E5_BASE,
]

# ==============================================================================
# Variant Registry
# ==============================================================================

ALL_VARIANTS: dict[str, ModelVariant] = {
    "mistral_7b_e5_base": MISTRAL_7B_E5_BASE,
    "qwen_3b_e5_base": QWEN_3B_E5_BASE,
    "qwen_7b_e5_base": QWEN_7B_E5_BASE,
    "qwen_14b_e5_base": QWEN_14B_E5_BASE,
    "mistral_7b_bge_base": MISTRAL_7B_BGE_BASE,
    "qwen_7b_bge_base": QWEN_7B_BGE_BASE,
    "mistral_7b_gte_base": MISTRAL_7B_GTE_BASE,
}


def get_variant(name: str) -> ModelVariant:
    """Get a model variant by name."""
    if name not in ALL_VARIANTS:
        available = ", ".join(ALL_VARIANTS.keys())
        raise ValueError(f"Unknown variant: {name}. Available: {available}")
    return ALL_VARIANTS[name]


def list_variants(
    tags: list[str] = None,
    max_vram_gb: int = None,
) -> list[ModelVariant]:
    """
    List variants matching criteria.
    
    Args:
        tags: Filter by tags (any match)
        max_vram_gb: Filter by VRAM requirement
    
    Returns:
        List of matching variants
    """
    variants = list(ALL_VARIANTS.values())
    
    if tags:
        variants = [v for v in variants if any(t in v.tags for t in tags)]
    
    if max_vram_gb:
        variants = [v for v in variants if v.vram_requirement_gb <= max_vram_gb]
    
    return variants


def get_variant_for_hardware(
    available_vram_gb: int,
    prefer_family: str = None,
    prefer_size: str = "medium",  # "small", "medium", "large"
) -> ModelVariant:
    """
    Recommend a variant for available hardware.
    
    Args:
        available_vram_gb: Available VRAM in GB
        prefer_family: Preferred model family ("mistral", "qwen", etc.)
        prefer_size: Preferred model size
    
    Returns:
        Recommended ModelVariant
    """
    # Filter by VRAM
    candidates = list_variants(max_vram_gb=available_vram_gb)
    
    if not candidates:
        raise ValueError(
            f"No variants available for {available_vram_gb}GB VRAM. "
            f"Minimum requirement: {min(v.vram_requirement_gb for v in ALL_VARIANTS.values())}GB"
        )
    
    # Filter by family if specified
    if prefer_family:
        family_candidates = [v for v in candidates if v.model_family == prefer_family]
        if family_candidates:
            candidates = family_candidates
    
    # Filter by size preference
    size_map = {
        "small": (0, 5),
        "medium": (5, 10),
        "large": (10, 100),
    }
    
    if prefer_size in size_map:
        min_size, max_size = size_map[prefer_size]
        size_candidates = [
            v for v in candidates
            if min_size <= v.model_size_b < max_size
        ]
        if size_candidates:
            candidates = size_candidates
    
    # Return the largest model that fits
    return max(candidates, key=lambda v: v.model_size_b)


def print_variant_table():
    """Print a formatted table of all available variants."""
    print("\n" + "="*100)
    print("AVAILABLE MODEL VARIANTS")
    print("="*100)
    print(f"{'Name':<25} {'LLM':<30} {'Embed':<20} {'Size':>6} {'VRAM':>6} {'Speed':<8}")
    print("-"*100)
    
    for variant in sorted(ALL_VARIANTS.values(), key=lambda v: (v.model_size_b, v.name)):
        print(
            f"{variant.name:<25} "
            f"{variant.generation_model.split('/')[-1][:30]:<30} "
            f"{variant.embedding_model.split('/')[-1][:20]:<20} "
            f"{variant.model_size_b:>6.1f}B "
            f"{variant.vram_requirement_gb:>6}GB "
            f"{variant.estimated_speed:<8}"
        )
    
    print("="*100)
    print(f"Total variants: {len(ALL_VARIANTS)}")
    print("="*100 + "\n")


if __name__ == "__main__":
    print_variant_table()
