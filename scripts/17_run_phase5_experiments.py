"""
Phase 5: Split Experiment Types Clearly

This script runs two distinct experiment families with clear methodological separation:

Family A: Controlled Retrieval Comparison
  - Purpose: Compare vector-only vs GraphRAG retrieval using shared text units
  - Systems: baseline_shared_text_unit_rag vs pure_graphrag
  - Question: Does graph-aware retrieval help when evidence units are controlled?
  
Family B: Realistic End-to-End System Comparison
  - Purpose: Compare practical baseline vs practical GraphRAG as full systems
  - Systems: baseline_strong_rag vs graphrag_practical
  - Question: Which system is better as an actual QA pipeline on HybridQA?

Usage:
    # Run Family A (controlled comparison)
    python scripts/17_run_phase5_experiments.py --family A --split dev --limit 20
    
    # Run Family B (realistic comparison)
    python scripts/17_run_phase5_experiments.py --family B --split dev --limit 50
    
    # Run both families
    python scripts/17_run_phase5_experiments.py --family both --split dev --limit 30
    
    # Use specific corpus strategy for GraphRAG
    python scripts/17_run_phase5_experiments.py --family B --corpus-strategy hybrid
    
    # Use specific model variant
    python scripts/17_run_phase5_experiments.py --family B --variant qwen_14b_e5_base
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline.strong_baseline_pipeline import StrongBaselinePipeline, run_strong_baseline_on_questions
from src.config import model_variants
from src.config.model_registry import build_experiment_metadata, save_experiment_with_metadata
from src.data_pipeline.hybridqa_parser import load_hybridqa_data
from src.evaluation.evaluator import Evaluator
from src.evaluation.schemas import EvaluationExample
from src.graphrag_system.improved_corpus_prep import hybridqa_to_improved_graphrag_docs
from src.graphrag_system.query_runner import run_graphrag_experiment
from src.graphrag_system.runner import create_graphrag_config, has_graphrag_index, run_graphrag_index


ExperimentFamily = Literal["A", "B", "both"]


def _overall_metric(metrics: dict, key: str, default: float = 0.0) -> float:
    """Read an aggregate evaluator metric from either legacy or current shapes."""
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)

    aggregate = metrics.get("aggregate_metrics", {})
    for section in ("overall", "system_name"):
        rows = aggregate.get(section, [])
        if rows:
            value = rows[0].get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return default


def _answer_f1(metrics: dict) -> float:
    return _overall_metric(metrics, "token_f1", _overall_metric(metrics, "answer_f1"))


def _answer_em(metrics: dict) -> float:
    return _overall_metric(metrics, "normalized_exact_match", _overall_metric(metrics, "answer_em"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: Split Experiment Types Clearly")
    
    parser.add_argument(
        "--family",
        type=str,
        choices=["A", "B", "both"],
        required=True,
        help="Experiment family: A (controlled), B (realistic), or both"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "train"],
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (None = all)"
    )
    
    parser.add_argument(
        "--corpus-strategy",
        type=str,
        default="hybrid",
        choices=["row_centric", "table_centric", "entity_centric", "hybrid"],
        help="GraphRAG corpus construction strategy (Phase 4)"
    )
    
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant name from Phase 3.5 registry (default: use env vars)"
    )
    
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/experiments/phase5"),
        help="Root directory for experiment outputs"
    )
    
    parser.add_argument(
        "--force-rebuild-corpus",
        action="store_true",
        help="Force rebuild GraphRAG corpus even if it exists"
    )
    
    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="Force rebuild GraphRAG index even if it exists"
    )
    
    parser.add_argument(
        "--skip-graphrag",
        action="store_true",
        help="Skip GraphRAG experiments (only run baseline)"
    )
    
    return parser.parse_args()


def load_model_variant(variant_name: str = None) -> model_variants.ModelVariant:
    """Load model variant from registry."""
    if variant_name:
        variant = model_variants.get_variant(variant_name)
        if not variant:
            raise ValueError(f"Unknown variant: {variant_name}")
        return variant
    
    # Use default based on env vars
    return model_variants.ModelVariant(
        name="env_based",
        description="Environment-based configuration from LOCAL_GENERATION_MODEL and LOCAL_EMBEDDING_MODEL",
        generation_model=os.getenv("LOCAL_GENERATION_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/e5-base-v2"),
        model_family="mixed",
        model_size_b=3.0,
        vram_requirement_gb=12,
        estimated_speed="medium",
        notes="Using environment variables for model configuration",
    )


def run_family_a_experiments(
    records: list[dict],
    variant: model_variants.ModelVariant,
    corpus_strategy: str,
    output_dir: Path,
    force_rebuild_corpus: bool = False,
    force_rebuild_index: bool = False,
    skip_graphrag: bool = False,
) -> dict:
    """
    Run Family A: Controlled Retrieval Comparison
    
    Systems:
    - baseline_shared_text_unit_rag: Basic RAG using same text units as GraphRAG
    - pure_graphrag: GraphRAG retrieval with graph-aware ranking
    
    Returns:
        Dict with results for each system
    """
    print("\n" + "="*80)
    print("FAMILY A: CONTROLLED RETRIEVAL COMPARISON")
    print("="*80)
    print("Purpose: Compare vector-only vs GraphRAG retrieval using shared text units")
    print("Question: Does graph-aware retrieval help when evidence units are controlled?")
    print("="*80 + "\n")
    
    results = {}
    
    # ===========================================================================
    # System 1: Baseline Shared Text Unit RAG
    # ===========================================================================
    
    print("\n" + "-"*80)
    print("System 1: baseline_shared_text_unit_rag")
    print("-"*80)
    print("Using simple retrieval (no lexical scoring, no reranking)")
    print("Same text units as GraphRAG for controlled comparison")
    print("-"*80 + "\n")
    
    baseline_shared_start = time.time()
    
    # Use "simple" variant of baseline (ablation control)
    pipeline_shared = StrongBaselinePipeline(
        variant="simple",  # No lexical, no reranking
        embedding_model=variant.embedding_model,
        top_k=8,
        use_lexical=False,
        use_reranking=False,
        include_citations=False,
        max_context_chars=12000,
        max_answer_tokens=256,
    )
    
    pipeline_shared.prepare(
        records=records,
        max_passages=None,  # Index ALL passages - let semantic search decide!
        cache_dir=Path(f"cache/embeddings/family_a_{variant.name}"),
        force_rebuild=False,
    )
    
    predictions_shared = run_strong_baseline_on_questions(
        questions=records,
        pipeline=pipeline_shared,
        output_path=None,
    )
    
    baseline_shared_time = time.time() - baseline_shared_start
    
    # Prepare output directory
    baseline_shared_dir = output_dir / "family_a_baseline_shared_text_unit_rag"
    baseline_shared_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    examples = [EvaluationExample.from_dict(r) for r in records]
    evaluator = Evaluator(
        dataset_name="hybridqa",
        system_name="baseline_shared_text_unit_rag",
        model_backend=variant.generation_backend,
        generation_model=variant.generation_model,
        embedding_model=variant.embedding_model,
        experiment_id=f"phase5_family_a_baseline_{time.strftime('%Y%m%d_%H%M%S')}",
        query_mode="shared_text_unit_rag",
    )
    metrics_shared = evaluator.evaluate(
        examples=examples,
        predictions=predictions_shared,
        output_dir=baseline_shared_dir,
    )
    
    metadata_shared = build_experiment_metadata(
        experiment_name="family_a_baseline_shared_text_unit_rag",
        additional_metadata={
            "experiment_family": "A",
            "family_purpose": "controlled_retrieval_comparison",
            "system_name": "baseline_shared_text_unit_rag",
            "baseline_variant": "simple",
            "model_variant": variant.to_metadata(),
            "num_questions": len(records),
            "total_time_seconds": baseline_shared_time,
        }
    )
    
    save_experiment_with_metadata(
        predictions=predictions_shared,
        metrics=metrics_shared,
        output_dir=baseline_shared_dir,
        metadata=metadata_shared,
    )
    
    results["baseline_shared_text_unit_rag"] = {
        "metrics": metrics_shared,
        "num_predictions": len(predictions_shared),
        "time_seconds": baseline_shared_time,
        "output_dir": str(baseline_shared_dir),
    }
    
    print(f"\n✓ baseline_shared_text_unit_rag completed in {baseline_shared_time:.1f}s")
    print(f"  Answer F1: {_answer_f1(metrics_shared):.3f}")
    print(f"  Output: {baseline_shared_dir}")
    
    # ===========================================================================
    # System 2: Pure GraphRAG
    # ===========================================================================
    
    if skip_graphrag:
        print("\n⊘ Skipping GraphRAG (--skip-graphrag flag set)")
        return results
    
    print("\n" + "-"*80)
    print("System 2: pure_graphrag")
    print("-"*80)
    print("Using GraphRAG with graph-aware retrieval")
    print("Same text units as baseline for controlled comparison")
    print("-"*80 + "\n")
    
    graphrag_start = time.time()
    
    # Build GraphRAG corpus (if needed)
    graphrag_workspace = output_dir / "family_a_graphrag_workspace" / corpus_strategy
    graphrag_input_dir = graphrag_workspace / "input"
    
    if force_rebuild_corpus or not graphrag_input_dir.exists():
        print(f"Building GraphRAG corpus (strategy={corpus_strategy})...")
        hybridqa_to_improved_graphrag_docs(
            records=records,
            output_dir=graphrag_input_dir,
            strategy=corpus_strategy,
            format="txt",
        )
    else:
        print(f"✓ GraphRAG corpus already exists: {graphrag_input_dir}")
    
    # Create GraphRAG config
    print("Creating GraphRAG configuration...")
    create_graphrag_config(
        project_dir=graphrag_workspace,
        api_key=os.getenv("GRAPHRAG_API_KEY", "sk-local-dummy-key-for-testing"),
        model=variant.generation_model,
        embedding_model=variant.embedding_model,
    )
    
    # Run GraphRAG indexing
    if force_rebuild_index or not has_graphrag_index(graphrag_workspace):
        print("Running GraphRAG indexing...")
        success = run_graphrag_index(
            project_dir=graphrag_workspace,
            dry_run=False,
            verbose=True,
        )
        if not success:
            raise RuntimeError(f"GraphRAG indexing failed for workspace: {graphrag_workspace}")
        
        # Verify index was created
        if not has_graphrag_index(graphrag_workspace):
            raise RuntimeError(f"GraphRAG index files missing after indexing: {graphrag_workspace}")
    else:
        print(f"✓ GraphRAG index already exists")
    
    # Run GraphRAG queries
    print(f"Running GraphRAG queries on {len(records)} questions...")
    
    predictions_graphrag, query_stats = run_graphrag_experiment(
        questions=records,
        workspace_dir=graphrag_workspace,
        output_path=None,
        method="local",
        response_type="Single sentence",
        verbose=True,
    )
    
    graphrag_time = time.time() - graphrag_start
    
    # Prepare output directory
    graphrag_dir = output_dir / "family_a_pure_graphrag"
    graphrag_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    examples = [EvaluationExample.from_dict(r) for r in records]
    evaluator_graphrag = Evaluator(
        dataset_name="hybridqa",
        system_name="pure_graphrag",
        model_backend=variant.generation_backend,
        generation_model=variant.generation_model,
        embedding_model=variant.embedding_model,
        experiment_id=f"phase5_family_a_graphrag_{time.strftime('%Y%m%d_%H%M%S')}",
        query_mode="local",
    )
    metrics_graphrag = evaluator_graphrag.evaluate(
        examples=examples,
        predictions=predictions_graphrag,
        output_dir=graphrag_dir,
    )
    
    metadata_graphrag = build_experiment_metadata(
        experiment_name="family_a_pure_graphrag",
        additional_metadata={
            "experiment_family": "A",
            "family_purpose": "controlled_retrieval_comparison",
            "system_name": "pure_graphrag",
            "corpus_strategy": corpus_strategy,
            "model_variant": variant.to_metadata(),
            "num_questions": len(records),
            "total_time_seconds": graphrag_time,
            "query_stats": query_stats,
        }
    )
    
    save_experiment_with_metadata(
        predictions=predictions_graphrag,
        metrics=metrics_graphrag,
        output_dir=graphrag_dir,
        metadata=metadata_graphrag,
    )
    
    print(f"\n✓ pure_graphrag completed in {graphrag_time:.1f}s")
    print(f"  Answer F1: {_answer_f1(metrics_graphrag):.3f}")
    print(f"  Output: {graphrag_dir}")
    
    results["pure_graphrag"] = {
        "metrics": metrics_graphrag,
        "num_predictions": len(predictions_graphrag),
        "time_seconds": graphrag_time,
        "output_dir": str(graphrag_dir),
        "workspace_dir": str(graphrag_workspace),
    }
    
    return results


def run_family_b_experiments(
    records: list[dict],
    variant: model_variants.ModelVariant,
    corpus_strategy: str,
    output_dir: Path,
    force_rebuild_corpus: bool = False,
    force_rebuild_index: bool = False,
    skip_graphrag: bool = False,
) -> dict:
    """
    Run Family B: Realistic End-to-End System Comparison
    
    Systems:
    - baseline_strong_rag: Full-featured baseline with lexical + reranking
    - graphrag_practical: GraphRAG with practical configuration
    
    Returns:
        Dict with results for each system
    """
    print("\n" + "="*80)
    print("FAMILY B: REALISTIC END-TO-END SYSTEM COMPARISON")
    print("="*80)
    print("Purpose: Compare practical baseline vs practical GraphRAG as full systems")
    print("Question: Which system is better as an actual QA pipeline on HybridQA?")
    print("="*80 + "\n")
    
    results = {}
    
    # ===========================================================================
    # System 1: Baseline Strong RAG
    # ===========================================================================
    
    print("\n" + "-"*80)
    print("System 1: baseline_strong_rag")
    print("-"*80)
    print("Using strong baseline with lexical scoring and reranking (Phase 3)")
    print("Represents best-practice RAG system")
    print("-"*80 + "\n")
    
    baseline_strong_start = time.time()
    
    # Use "strong" variant of baseline (full-featured)
    pipeline_strong = StrongBaselinePipeline(
        variant="strong",
        embedding_model=variant.embedding_model,
        top_k=8,
        use_lexical=True,
        use_reranking=True,
        include_citations=False,
        max_context_chars=12000,
        max_answer_tokens=256,
    )
    
    pipeline_strong.prepare(
        records=records,
        max_passages=None,  # Index ALL passages - let semantic search decide!
        cache_dir=Path(f"cache/embeddings/family_b_{variant.name}"),
        force_rebuild=False,
    )
    
    predictions_strong = run_strong_baseline_on_questions(
        questions=records,
        pipeline=pipeline_strong,
        output_path=None,
    )
    
    baseline_strong_time = time.time() - baseline_strong_start
    
    # Prepare output directory
    baseline_strong_dir = output_dir / "family_b_baseline_strong_rag"
    baseline_strong_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    examples = [EvaluationExample.from_dict(r) for r in records]
    evaluator = Evaluator(
        dataset_name="hybridqa",
        system_name="baseline_strong_rag",
        model_backend=variant.generation_backend,
        generation_model=variant.generation_model,
        embedding_model=variant.embedding_model,
        experiment_id=f"phase5_family_b_baseline_{time.strftime('%Y%m%d_%H%M%S')}",
        query_mode="strong_rag",
    )
    metrics_strong = evaluator.evaluate(
        examples=examples,
        predictions=predictions_strong,
        output_dir=baseline_strong_dir,
    )
    
    metadata_strong = build_experiment_metadata(
        experiment_name="family_b_baseline_strong_rag",
        additional_metadata={
            "experiment_family": "B",
            "family_purpose": "realistic_endtoend_comparison",
            "system_name": "baseline_strong_rag",
            "baseline_variant": "strong",
            "model_variant": variant.to_metadata(),
            "num_questions": len(records),
            "total_time_seconds": baseline_strong_time,
        }
    )
    
    save_experiment_with_metadata(
        predictions=predictions_strong,
        metrics=metrics_strong,
        output_dir=baseline_strong_dir,
        metadata=metadata_strong,
    )
    
    results["baseline_strong_rag"] = {
        "metrics": metrics_strong,
        "num_predictions": len(predictions_strong),
        "time_seconds": baseline_strong_time,
        "output_dir": str(baseline_strong_dir),
    }
    
    print(f"\n✓ baseline_strong_rag completed in {baseline_strong_time:.1f}s")
    print(f"  Answer F1: {_answer_f1(metrics_strong):.3f}")
    print(f"  Output: {baseline_strong_dir}")
    
    # ===========================================================================
    # System 2: GraphRAG Practical
    # ===========================================================================
    
    if skip_graphrag:
        print("\n⊘ Skipping GraphRAG (--skip-graphrag flag set)")
        return results
    
    print("\n" + "-"*80)
    print("System 2: graphrag_practical")
    print("-"*80)
    print("Using GraphRAG with improved corpus construction (Phase 4)")
    print("Represents best-practice GraphRAG system")
    print("-"*80 + "\n")
    
    graphrag_start = time.time()
    
    # Build GraphRAG corpus with improved strategy
    graphrag_workspace = output_dir / "family_b_graphrag_workspace" / corpus_strategy
    graphrag_input_dir = graphrag_workspace / "input"
    
    if force_rebuild_corpus or not graphrag_input_dir.exists():
        print(f"Building GraphRAG corpus (strategy={corpus_strategy})...")
        hybridqa_to_improved_graphrag_docs(
            records=records,
            output_dir=graphrag_input_dir,
            strategy=corpus_strategy,
            format="txt",
        )
    else:
        print(f"✓ GraphRAG corpus already exists: {graphrag_input_dir}")
    
    # Create GraphRAG config
    print("Creating GraphRAG configuration...")
    create_graphrag_config(
        project_dir=graphrag_workspace,
        api_key=os.getenv("GRAPHRAG_API_KEY", "sk-local-dummy-key-for-testing"),
        model=variant.generation_model,
        embedding_model=variant.embedding_model,
    )
    
    # Run GraphRAG indexing
    if force_rebuild_index or not has_graphrag_index(graphrag_workspace):
        print("Running GraphRAG indexing...")
        success = run_graphrag_index(
            project_dir=graphrag_workspace,
            dry_run=False,
            verbose=True,
        )
        if not success:
            raise RuntimeError(f"GraphRAG indexing failed for workspace: {graphrag_workspace}")
        
        # Verify index was created
        if not has_graphrag_index(graphrag_workspace):
            raise RuntimeError(f"GraphRAG index files missing after indexing: {graphrag_workspace}")
    else:
        print(f"✓ GraphRAG index already exists")
    
    # Run GraphRAG queries
    print(f"Running GraphRAG queries on {len(records)} questions...")
    
    predictions_graphrag, query_stats = run_graphrag_experiment(
        questions=records,
        workspace_dir=graphrag_workspace,
        output_path=None,
        method="local",
        response_type="Single sentence",
        verbose=True,
    )
    
    graphrag_time = time.time() - graphrag_start
    
    # Prepare output directory
    graphrag_dir = output_dir / "family_b_graphrag_practical"
    graphrag_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    examples = [EvaluationExample.from_dict(r) for r in records]
    evaluator_graphrag = Evaluator(
        dataset_name="hybridqa",
        system_name="graphrag_practical",
        model_backend=variant.generation_backend,
        generation_model=variant.generation_model,
        embedding_model=variant.embedding_model,
        experiment_id=f"phase5_family_b_graphrag_{time.strftime('%Y%m%d_%H%M%S')}",
        query_mode="local",
    )
    metrics_graphrag = evaluator_graphrag.evaluate(
        examples=examples,
        predictions=predictions_graphrag,
        output_dir=graphrag_dir,
    )
    
    metadata_graphrag = build_experiment_metadata(
        experiment_name="family_b_graphrag_practical",
        additional_metadata={
            "experiment_family": "B",
            "family_purpose": "realistic_endtoend_comparison",
            "system_name": "graphrag_practical",
            "corpus_strategy": corpus_strategy,
            "model_variant": variant.to_metadata(),
            "num_questions": len(records),
            "total_time_seconds": graphrag_time,
            "query_stats": query_stats,
        }
    )
    
    save_experiment_with_metadata(
        predictions=predictions_graphrag,
        metrics=metrics_graphrag,
        output_dir=graphrag_dir,
        metadata=metadata_graphrag,
    )
    
    print(f"\n✓ graphrag_practical completed in {graphrag_time:.1f}s")
    print(f"  Answer F1: {_answer_f1(metrics_graphrag):.3f}")
    print(f"  Output: {graphrag_dir}")
    
    results["graphrag_practical"] = {
        "metrics": metrics_graphrag,
        "num_predictions": len(predictions_graphrag),
        "time_seconds": graphrag_time,
        "output_dir": str(graphrag_dir),
        "workspace_dir": str(graphrag_workspace),
    }
    
    return results


def print_comparison_summary(family: str, results: dict) -> None:
    """Print comparison summary for experiment family."""
    print("\n" + "="*80)
    print(f"FAMILY {family} RESULTS SUMMARY")
    print("="*80)
    
    if not results:
        print("\n⚠️  No results - experiments did not complete successfully.")
        print("="*80)
        return
    
    for system_name, result in results.items():
        print(f"\n{system_name}:")
        print(f"  Status: {result.get('status', 'complete')}")
        print(f"  Time: {result.get('time_seconds', 0):.1f}s")
        print(f"  Predictions: {result.get('num_predictions', 0)}")
        
        metrics = result.get("metrics", {})
        if metrics:
            print(f"  Answer F1: {_answer_f1(metrics):.3f}")
            print(f"  Answer EM: {_answer_em(metrics):.3f}")
        
        output_dir = result.get("output_dir") or result.get("workspace_dir")
        if output_dir:
            print(f"  Output: {output_dir}")
    
    print("\n" + "="*80)


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("PHASE 5: SPLIT EXPERIMENT TYPES CLEARLY")
    print("="*80)
    print(f"Family: {args.family}")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit or 'all questions'}")
    print(f"Corpus Strategy: {args.corpus_strategy}")
    print(f"Variant: {args.variant or 'env-based'}")
    print("="*80 + "\n")
    
    # Load data
    print(f"Loading HybridQA {args.split} split...")
    records = load_hybridqa_data(args.split)
    
    if args.limit:
        records = records[:args.limit]
    
    print(f"✓ Loaded {len(records)} questions")
    
    # Load model variant
    variant = load_model_variant(args.variant)
    print(f"✓ Using model variant: {variant.name}")
    print(f"  Generation: {variant.generation_model}")
    print(f"  Embedding:  {variant.embedding_model}")
    
    # Set environment variables for variant
    env_vars = variant.to_env_dict()
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = args.output_root / f"{args.family}_{args.split}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directory: {experiment_dir}")
    
    # Run experiments
    all_results = {}
    
    if args.family in ["A", "both"]:
        print("\n" + "#"*80)
        print("# RUNNING FAMILY A EXPERIMENTS")
        print("#"*80)
        
        family_a_dir = experiment_dir / "family_a"
        family_a_dir.mkdir(parents=True, exist_ok=True)
        
        results_a = run_family_a_experiments(
            records=records,
            variant=variant,
            corpus_strategy=args.corpus_strategy,
            output_dir=family_a_dir,
            force_rebuild_corpus=args.force_rebuild_corpus,
            force_rebuild_index=args.force_rebuild_index,
            skip_graphrag=args.skip_graphrag,
        )
        
        all_results["family_a"] = results_a
        print_comparison_summary("A", results_a)
    
    if args.family in ["B", "both"]:
        print("\n" + "#"*80)
        print("# RUNNING FAMILY B EXPERIMENTS")
        print("#"*80)
        
        family_b_dir = experiment_dir / "family_b"
        family_b_dir.mkdir(parents=True, exist_ok=True)
        
        results_b = run_family_b_experiments(
            records=records,
            variant=variant,
            corpus_strategy=args.corpus_strategy,
            output_dir=family_b_dir,
            force_rebuild_corpus=args.force_rebuild_corpus,
            force_rebuild_index=args.force_rebuild_index,
            skip_graphrag=args.skip_graphrag,
        )
        
        all_results["family_b"] = results_b
        print_comparison_summary("B", results_b)
    
    # Save master summary
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "args": vars(args),
            "variant": variant.to_metadata(),
            "num_questions": len(records),
            "results": all_results,
        }, f, indent=2, default=str)
    
    print(f"\n✓ Experiment summary saved: {summary_path}")
    
    print("\n" + "="*80)
    print("PHASE 5 EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Summary: {summary_path}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
