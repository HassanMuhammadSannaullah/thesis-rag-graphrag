"""
Model comparison runner for Phase 3.5 and Phase 6.5.

Runs the same experiment configuration across multiple model variants
to perform model ablation studies.

Usage:
    # Run smoke test with minimum ablation variants
    python scripts/16_run_model_comparison.py --split dev --limit 10 --variants minimum
    
    # Run full comparison with extended variants
    python scripts/16_run_model_comparison.py --split dev --limit 50 --variants extended
    
    # Run specific variants
    python scripts/16_run_model_comparison.py --split dev --limit 20 --variant-names mistral_7b_e5_base,qwen_14b_e5_base
    
    # List available variants
    python scripts/16_run_model_comparison.py --list-variants
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline.strong_baseline_pipeline import StrongBaselinePipeline, run_strong_baseline_on_questions
from src.config import model_variants
from src.config.model_registry import build_experiment_metadata
from src.data_pipeline.hybridqa_evidence import attach_proxy_evidence
from src.evaluation.evaluator import Evaluator


def load_parsed_records(split: str = "dev", limit: int = None) -> list[dict]:
    """Load parsed HybridQA records."""
    from src.config import settings as cfg
    
    parsed_path = Path(cfg.ORIGINAL_DIR) / f"{split}.jsonl"
    
    if not parsed_path.exists():
        raise FileNotFoundError(
            f"Parsed data not found: {parsed_path}\n"
            f"Run `python scripts/02_parse_hybridqa.py --split {split}` first."
        )
    
    records = []
    with open(parsed_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # Backfill proxy evidence if not present
                if "proxy_evidence" not in rec:
                    attach_proxy_evidence(rec)
                records.append(rec)
    
    if limit:
        records = records[:limit]
    
    return records


def run_experiment_with_variant(
    variant: model_variants.ModelVariant,
    records: list[dict],
    baseline_variant: str = "strong",
    top_k: int = 8,
    output_dir: Path = None,
) -> tuple[list, dict, dict]:
    """
    Run baseline experiment with a specific model variant.
    
    Returns:
        (predictions, metrics, experiment_metadata)
    """
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT WITH VARIANT: {variant.name}")
    print("="*80)
    print(f"Generation Model: {variant.generation_model}")
    print(f"Embedding Model:  {variant.embedding_model}")
    print(f"Model Family:     {variant.model_family}")
    print(f"Model Size:       {variant.model_size_b}B")
    print(f"VRAM Required:    {variant.vram_requirement_gb}GB")
    print("="*80 + "\n")
    
    # Set environment variables for this variant
    env_vars = variant.to_env_dict()
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key} = {value}")
    
    # Reload config to pick up new environment variables
    import importlib
    from src.config import settings

    importlib.reload(settings)
    
    print(f"\nCreating pipeline (baseline_variant={baseline_variant})...")
    
    # Create pipeline
    pipeline = StrongBaselinePipeline(
        variant=baseline_variant,
        embedding_model=variant.embedding_model,
        top_k=top_k,
        use_lexical=True,
        use_reranking=True,
        include_citations=False,
        max_context_chars=12000,
        max_answer_tokens=256,
    )
    
    # Prepare pipeline
    print("Preparing pipeline (building corpus, index, lookup)...")
    start_time = time.time()
    pipeline.prepare(
        records=records,
        max_passages=None,  # Index ALL passages
        cache_dir=Path(f"cache/embeddings/{variant.name}"),
        force_rebuild=False,
    )
    prep_time = time.time() - start_time
    print(f"Pipeline prepared in {prep_time:.1f}s")
    
    # Run on questions
    print(f"\nRunning on {len(records)} questions...")
    start_time = time.time()
    predictions = run_strong_baseline_on_questions(
        questions=records,
        pipeline=pipeline,
        output_path=None,  # Don't save yet
    )
    query_time = time.time() - start_time
    print(f"Queries completed in {query_time:.1f}s")
    
    # Evaluate
    print("\nEvaluating predictions...")
    evaluator = Evaluator()
    metrics = evaluator.evaluate_predictions(
        predictions=predictions,
        gold_answers=[r.get("answer", r.get("answer-text")) for r in records],
    )
    
    # Build experiment metadata
    experiment_metadata = build_experiment_metadata(
        experiment_name=f"model_comparison_{variant.name}",
        additional_metadata={
            "variant": variant.to_metadata(),
            "baseline_variant": baseline_variant,
            "top_k": top_k,
            "num_questions": len(records),
            "preparation_time_sec": prep_time,
            "query_time_sec": query_time,
            "avg_time_per_question_sec": query_time / len(records) if records else 0,
        }
    )
    
    # Print results
    print("\n" + "="*80)
    print(f"RESULTS FOR {variant.name.upper()}")
    print("="*80)
    print(f"Exact Match:        {metrics.get('exact_match', 0.0):.3f}")
    print(f"F1 Score:           {metrics.get('f1_score', 0.0):.3f}")
    print(f"Token Overlap:      {metrics.get('token_overlap', 0.0):.3f}")
    print(f"Avg Retrieved:      {metrics.get('avg_num_retrieved', 0):.1f}")
    print(f"Avg Time/Question:  {query_time / len(records):.3f}s")
    print("="*80 + "\n")
    
    # Save results if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        pred_path = output_dir / f"{variant.name}_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred.dict(), ensure_ascii=False) + "\n")
        
        # Save metrics
        metrics_path = output_dir / f"{variant.name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata_path = output_dir / f"{variant.name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved results to {output_dir}")
    
    return predictions, metrics, experiment_metadata


def run_model_comparison(
    variants: list[model_variants.ModelVariant],
    split: str = "dev",
    limit: int = 20,
    baseline_variant: str = "strong",
    top_k: int = 8,
    output_dir: Path = None,
):
    """
    Run comparison across multiple model variants.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON STUDY")
    print("="*80)
    print(f"Split:            {split}")
    print(f"Questions:        {limit}")
    print(f"Baseline Variant: {baseline_variant}")
    print(f"Top-K Retrieval:  {top_k}")
    print(f"Model Variants:   {len(variants)}")
    print("="*80)
    
    for i, variant in enumerate(variants, 1):
        print(f"\n[{i}/{len(variants)}] {variant.name}: {variant.description}")
    
    print("\n" + "="*80 + "\n")
    
    # Load data once
    print(f"Loading {split} split...")
    records = load_parsed_records(split=split, limit=limit)
    print(f"Loaded {len(records)} records\n")
    
    # Run experiments
    all_results = []
    
    for i, variant in enumerate(variants, 1):
        print(f"\n{'='*80}")
        print(f"VARIANT {i}/{len(variants)}: {variant.name}")
        print(f"{'='*80}\n")
        
        try:
            variant_output_dir = output_dir / variant.name if output_dir else None
            predictions, metrics, metadata = run_experiment_with_variant(
                variant=variant,
                records=records,
                baseline_variant=baseline_variant,
                top_k=top_k,
                output_dir=variant_output_dir,
            )
            
            all_results.append({
                "variant": variant,
                "predictions": predictions,
                "metrics": metrics,
                "metadata": metadata,
            })
            
        except Exception as e:
            print(f"\n✗ ERROR running variant {variant.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison table
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Variant':<25} {'EM':>8} {'F1':>8} {'Overlap':>8} {'Speed':>10} {'VRAM':>6}")
    print("-"*80)
    
    for result in all_results:
        variant = result["variant"]
        metrics = result["metrics"]
        metadata = result["metadata"]
        
        em = metrics.get("exact_match", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        overlap = metrics.get("token_overlap", 0.0)
        speed = metadata.get("avg_time_per_question_sec", 0.0)
        vram = variant.vram_requirement_gb
        
        print(
            f"{variant.name:<25} "
            f"{em:>8.3f} "
            f"{f1:>8.3f} "
            f"{overlap:>8.3f} "
            f"{speed:>9.2f}s "
            f"{vram:>5}GB"
        )
    
    print("="*80)
    
    # Save comparison summary
    if output_dir:
        summary_path = output_dir / "comparison_summary.json"
        summary = {
            "experiment_config": {
                "split": split,
                "num_questions": limit,
                "baseline_variant": baseline_variant,
                "top_k": top_k,
            },
            "variants": [
                {
                    "name": result["variant"].name,
                    "metrics": result["metrics"],
                    "metadata": result["metadata"],
                }
                for result in all_results
            ],
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved comparison summary to {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Model comparison runner for Phase 3.5/6.5")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--limit", type=int, default=20, help="Number of questions")
    parser.add_argument("--baseline-variant", choices=["simple", "strong"], default="strong")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--variants",
        choices=["minimum", "extended", "smoke"],
        help="Use predefined variant set"
    )
    parser.add_argument(
        "--variant-names",
        help="Comma-separated list of variant names to test"
    )
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    
    args = parser.parse_args()
    
    # List variants and exit
    if args.list_variants:
        model_variants.print_variant_table()
        print("\nPredefined variant sets:")
        print(f"  minimum:  {[v.name for v in model_variants.MINIMUM_ABLATION_VARIANTS]}")
        print(f"  extended: {[v.name for v in model_variants.EXTENDED_ABLATION_VARIANTS]}")
        print(f"  smoke:    {[v.name for v in model_variants.SMOKE_TEST_VARIANTS]}")
        return
    
    # Select variants
    if args.variants == "minimum":
        variants = model_variants.MINIMUM_ABLATION_VARIANTS
    elif args.variants == "extended":
        variants = model_variants.EXTENDED_ABLATION_VARIANTS
    elif args.variants == "smoke":
        variants = model_variants.SMOKE_TEST_VARIANTS
    elif args.variant_names:
        variant_names = [n.strip() for n in args.variant_names.split(",")]
        variants = [model_variants.get_variant(name) for name in variant_names]
    else:
        print("ERROR: Must specify --variants or --variant-names")
        print("Use --list-variants to see available options")
        sys.exit(1)
    
    # Set output directory
    if not args.output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(f"results/experiments/model_comparison_{timestamp}")
    
    # Run comparison
    run_model_comparison(
        variants=variants,
        split=args.split,
        limit=args.limit,
        baseline_variant=args.baseline_variant,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
