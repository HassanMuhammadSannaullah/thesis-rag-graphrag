"""
Phase 5 Smoke Test - Split Experiment Types

Quick test to verify both experiment families (A and B) work correctly.
Uses a small sample (5 questions) to test infrastructure without long delays.

Usage:
    python scripts/test_phase5_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.hybridqa_parser import load_hybridqa_data


def test_phase5_structure():
    """Test: Phase 5 module imports and basic structure."""
    print("\n" + "="*80)
    print("TEST 1: Phase 5 Structure")
    print("="*80)
    
    try:
        # Import the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_phase5_experiments",
            Path(__file__).parent / "17_run_phase5_experiments.py"
        )
        phase5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(phase5)
        
        print("✓ Phase 5 module imports successfully")
        
        # Check key functions exist
        assert hasattr(phase5, "run_family_a_experiments")
        assert hasattr(phase5, "run_family_b_experiments")
        assert hasattr(phase5, "load_model_variant")
        assert hasattr(phase5, "print_comparison_summary")
        print("✓ All required functions present")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 1 PASSED")
    return True


def test_graphrag_query_runner():
    """Test: GraphRAG query runner imports."""
    print("\n" + "="*80)
    print("TEST 2: GraphRAG Query Runner")
    print("="*80)
    
    try:
        from src.graphrag_system.query_runner import (
            run_graphrag_batch_queries,
            run_graphrag_experiment,
        )
        print("✓ GraphRAG query runner imports successfully")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 2 PASSED")
    return True


def test_data_loading():
    """Test: Can load HybridQA data."""
    print("\n" + "="*80)
    print("TEST 3: Data Loading")
    print("="*80)
    
    try:
        print("Loading HybridQA dev split...")
        records = load_hybridqa_data("dev")
        sample = records[:5]
        
        print(f"✓ Loaded {len(records)} records")
        print(f"✓ Sample size: {len(sample)}")
        
        # Verify structure
        for rec in sample:
            assert "question_id" in rec
            assert "question" in rec
            assert "answer" in rec or "answer-text" in rec
        
        print("✓ All sample records have required fields")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 3 PASSED")
    return True


def test_model_variant_loading():
    """Test: Model variant loading from Phase 3.5."""
    print("\n" + "="*80)
    print("TEST 4: Model Variant Loading")
    print("="*80)
    
    try:
        from src.config import model_variants
        
        # Try to load a specific variant
        variant = model_variants.get_variant("qwen_3b_e5_base")
        
        if variant:
            print(f"✓ Loaded variant: {variant.name}")
            print(f"  Generation: {variant.generation_model}")
            print(f"  Embedding:  {variant.embedding_model}")
        else:
            print("⚠ Variant not found, will use env-based defaults")
        
        # List available variants
        variants = model_variants.list_variants()
        print(f"✓ Available variants: {len(variants)}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 4 PASSED")


def test_baseline_pipeline():
    """Test: Strong baseline pipeline can be created."""
    print("\n" + "="*80)
    print("TEST 5: Baseline Pipeline Creation")
    print("="*80)
    
    try:
        from src.baseline.strong_baseline_pipeline import StrongBaselinePipeline
        
        # Create simple variant (for Family A)
        pipeline_simple = StrongBaselinePipeline(
            variant="simple",
            embedding_model="intfloat/e5-base-v2",
            top_k=5,
            use_lexical=False,
            use_reranking=False,
        )
        print("✓ Created simple baseline pipeline (Family A)")
        
        # Create strong variant (for Family B)
        pipeline_strong = StrongBaselinePipeline(
            variant="strong",
            embedding_model="intfloat/e5-base-v2",
            top_k=5,
            use_lexical=True,
            use_reranking=True,
        )
        print("✓ Created strong baseline pipeline (Family B)")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 5 PASSED")
    return True


def test_improved_corpus_prep():
    """Test: Improved corpus preparation works."""
    print("\n" + "="*80)
    print("TEST 6: Improved Corpus Preparation")
    print("="*80)
    
    try:
        from src.graphrag_system.improved_corpus_prep import (
            build_graphrag_corpus,
            compare_strategies,
        )
        from src.data_pipeline.hybridqa_parser import load_hybridqa_data
        
        # Load small sample
        records = load_hybridqa_data("dev")[:2]
        
        # Test each strategy
        strategies = ["row_centric", "table_centric", "entity_centric", "hybrid"]
        
        for strategy in strategies:
            docs = build_graphrag_corpus(records, strategy=strategy)
            print(f"✓ Strategy '{strategy}': generated {len(docs)} documents")
        
        # Test comparison
        comparison = compare_strategies(records, sample_size=2)
        print(f"✓ Strategy comparison completed for {len(comparison)} strategies")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 6 PASSED")
    return True


def main():
    """Run all Phase 5 smoke tests."""
    print("\n" + "="*80)
    print("PHASE 5: SPLIT EXPERIMENT TYPES - SMOKE TEST SUITE")
    print("="*80)
    print("Testing experiment infrastructure without running full experiments")
    print("="*80 + "\n")
    
    tests = [
        test_phase5_structure,
        test_graphrag_query_runner,
        test_data_loading,
        test_model_variant_loading,
        test_baseline_pipeline,
        test_improved_corpus_prep,
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {test_func.__name__}")
            print(f"   Error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 5 infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Start local LLM server (if needed)")
        print("  2. Run Family A smoke test:")
        print("     python scripts/17_run_phase5_experiments.py --family A --split dev --limit 5 --skip-graphrag")
        print("  3. Run Family B smoke test:")
        print("     python scripts/17_run_phase5_experiments.py --family B --split dev --limit 5 --skip-graphrag")
        print("  4. Test with GraphRAG (requires indexing - takes longer):")
        print("     python scripts/17_run_phase5_experiments.py --family A --split dev --limit 5")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
