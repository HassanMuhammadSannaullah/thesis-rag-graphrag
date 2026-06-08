"""
Test script for Phase 3 strong baseline RAG.

Tests both simple baseline (for ablation) and strong baseline (thesis system).

Usage:
    python scripts/test_strong_baseline.py --split dev --limit 5 --variant strong
    python scripts/test_strong_baseline.py --split dev --limit 10 --variant simple
    python scripts/test_strong_baseline.py --split dev --limit 10 --compare-variants
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline.strong_baseline_pipeline import StrongBaselinePipeline, run_strong_baseline_on_questions
from src.config import settings as cfg
from src.data_pipeline.hybridqa_evidence import attach_proxy_evidence
from src.evaluation.evaluator import Evaluator


def load_parsed_records(split: str = "dev") -> list[dict]:
    """Load parsed HybridQA records."""
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
    
    return records


def test_strong_baseline(
    variant: str = "strong",
    split: str = "dev",
    limit: int = 5,
    save_results: bool = True,
):
    """
    Test strong baseline on a small subset of questions.
    """
    print(f"=== Testing {variant.upper()} Baseline ===\n")
    
    # Load records
    print(f"Loading {split} split...")
    records = load_parsed_records(split)
    
    if limit:
        records = records[:limit]
    
    print(f"Loaded {len(records)} records\n")
    
    # Create pipeline
    print(f"Creating pipeline (variant={variant})...")
    pipeline = StrongBaselinePipeline(
        variant=variant,
        embedding_model=cfg.EMBEDDING_MODEL_NAME,
        top_k=8,
        use_lexical=True,
        use_reranking=True,
        include_citations=False,
        max_context_chars=12000,
        max_answer_tokens=256,
    )
    
    # Prepare pipeline
    print("\nPreparing pipeline (building corpus, index, lookup)...")
    pipeline.prepare(
        records=records,
        max_passages=None,  # Index ALL passages
        cache_dir=Path("cache/embeddings"),
        force_rebuild=False,
    )
    
    # Run on questions
    print("\n" + "="*60)
    predictions = run_strong_baseline_on_questions(
        questions=records,
        pipeline=pipeline,
        output_path=None,  # Don't save during test
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("\nEvaluating predictions...")
    
    evaluator = Evaluator()
    metrics = evaluator.evaluate_predictions(
        predictions=predictions,
        gold_answers=[r.get("answer", r.get("answer-text")) for r in records],
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"RESULTS ({variant.upper()} baseline, {len(predictions)} questions):")
    print("="*60)
    print(f"Exact Match:        {metrics.get('exact_match', 0.0):.3f}")
    print(f"F1 Score:           {metrics.get('f1_score', 0.0):.3f}")
    print(f"Token Overlap:      {metrics.get('token_overlap', 0.0):.3f}")
    print(f"Avg Retrieved:      {metrics.get('avg_num_retrieved', 0):.1f}")
    print(f"Avg Retrieval Time: {metrics.get('avg_retrieve_time_sec', 0.0):.3f}s")
    print(f"Avg Total Time:     {metrics.get('avg_total_time_sec', 0.0):.3f}s")
    print("="*60)
    
    # Save results if requested
    if save_results:
        output_dir = Path("results/experiments/strong_baseline_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pred_path = output_dir / f"{variant}_{split}_{limit}q_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred.dict(), ensure_ascii=False) + "\n")
        
        metrics_path = output_dir / f"{variant}_{split}_{limit}q_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved results to {output_dir}")
    
    return predictions, metrics


def compare_variants(split: str = "dev", limit: int = 10):
    """Compare simple vs strong baseline side by side."""
    print("\n" + "="*60)
    print("COMPARING SIMPLE vs STRONG BASELINE")
    print("="*60 + "\n")
    
    # Test simple
    print(">>> Testing SIMPLE baseline...")
    simple_preds, simple_metrics = test_strong_baseline(
        variant="simple",
        split=split,
        limit=limit,
        save_results=True,
    )
    
    print("\n\n")
    
    # Test strong
    print(">>> Testing STRONG baseline...")
    strong_preds, strong_metrics = test_strong_baseline(
        variant="strong",
        split=split,
        limit=limit,
        save_results=True,
    )
    
    # Comparison table
    print("\n\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Simple':<15} {'Strong':<15} {'Improvement'}")
    print("-"*60)
    
    metrics_to_compare = [
        ("Exact Match", "exact_match"),
        ("F1 Score", "f1_score"),
        ("Token Overlap", "token_overlap"),
        ("Avg Retrieved", "avg_num_retrieved"),
        ("Avg Total Time (s)", "avg_total_time_sec"),
    ]
    
    for label, key in metrics_to_compare:
        simple_val = simple_metrics.get(key, 0.0)
        strong_val = strong_metrics.get(key, 0.0)
        
        if key == "avg_total_time_sec":
            # For time, lower is better
            improvement = f"{((simple_val - strong_val) / simple_val * 100):.1f}% faster" if simple_val > 0 else "N/A"
        else:
            # For other metrics, higher is better
            improvement = f"+{((strong_val - simple_val) / max(simple_val, 0.001) * 100):.1f}%" if simple_val > 0 else "N/A"
        
        print(f"{label:<25} {simple_val:<15.3f} {strong_val:<15.3f} {improvement}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Test Phase 3 strong baseline RAG")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to test")
    parser.add_argument("--variant", choices=["simple", "strong"], default="strong")
    parser.add_argument("--compare-variants", action="store_true", help="Compare simple vs strong")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    if args.compare_variants:
        compare_variants(split=args.split, limit=args.limit)
    else:
        test_strong_baseline(
            variant=args.variant,
            split=args.split,
            limit=args.limit,
            save_results=not args.no_save,
        )


if __name__ == "__main__":
    main()
