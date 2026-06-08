"""
Test script for Phase 4: GraphRAG Source Construction Strategies

This script tests all four strategies for building GraphRAG documents:
1. Row-centric: One document per row + linked passages
2. Table-centric: One comprehensive document per table
3. Entity-centric: One document per entity with table context
4. Hybrid: Table overview + detailed row documents

Tests:
- Each strategy produces valid documents
- Document IDs are stable and unique
- Metadata is preserved
- Documents can be saved and loaded
- Strategies can be compared

Usage:
    python scripts/test_graphrag_strategies.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.data_pipeline.hybridqa_parser import load_hybridqa_data
from src.graphrag_system.improved_corpus_prep import (
    build_graphrag_corpus,
    save_graphrag_corpus,
    compare_strategies,
    hybridqa_to_improved_graphrag_docs,
)


def test_basic_corpus_building():
    """Test: Each strategy can build documents."""
    print("\n" + "="*80)
    print("TEST 1: Basic Corpus Building")
    print("="*80)
    
    # Load sample data
    print("\n📂 Loading HybridQA dev data...")
    records = load_hybridqa_data("dev")
    sample_records = records[:5]
    print(f"✓ Loaded {len(sample_records)} sample records")
    
    strategies = ["row_centric", "table_centric", "entity_centric", "hybrid"]
    
    for strategy in strategies:
        print(f"\n🔧 Testing strategy: {strategy}")
        try:
            docs = build_graphrag_corpus(sample_records, strategy=strategy)
            print(f"  ✓ Generated {len(docs)} documents")
            
            # Verify all docs have required fields
            for doc in docs:
                assert "id" in doc, f"Missing 'id' in document"
                assert "title" in doc, f"Missing 'title' in document"
                assert "text" in doc, f"Missing 'text' in document"
                assert "metadata" in doc, f"Missing 'metadata' in document"
                assert doc["metadata"].get("strategy") == strategy
            
            print(f"  ✓ All documents have required fields")
            
            # Check average document length
            avg_len = sum(len(d["text"]) for d in docs) / len(docs)
            print(f"  ℹ Average document length: {avg_len:.0f} chars")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False
    
    print("\n✅ TEST 1 PASSED: All strategies build valid documents")
    return True


def test_document_id_stability():
    """Test: Document IDs are stable and unique."""
    print("\n" + "="*80)
    print("TEST 2: Document ID Stability")
    print("="*80)
    
    print("\n📂 Loading HybridQA dev data...")
    records = load_hybridqa_data("dev")
    sample_records = records[:3]
    
    strategies = ["row_centric", "table_centric", "entity_centric", "hybrid"]
    
    for strategy in strategies:
        print(f"\n🔧 Testing strategy: {strategy}")
        
        # Build twice
        docs1 = build_graphrag_corpus(sample_records, strategy=strategy)
        docs2 = build_graphrag_corpus(sample_records, strategy=strategy)
        
        ids1 = [d["id"] for d in docs1]
        ids2 = [d["id"] for d in docs2]
        
        # Check stability
        if ids1 != ids2:
            print(f"  ✗ FAILED: IDs not stable across builds")
            print(f"    First:  {ids1[:3]}...")
            print(f"    Second: {ids2[:3]}...")
            return False
        
        print(f"  ✓ IDs are stable across builds")
        
        # Check uniqueness
        if len(ids1) != len(set(ids1)):
            print(f"  ✗ FAILED: Duplicate IDs found")
            return False
        
        print(f"  ✓ All {len(ids1)} IDs are unique")
        
        # Print sample IDs
        print(f"  ℹ Sample IDs: {ids1[:2]}")
    
    print("\n✅ TEST 2 PASSED: Document IDs are stable and unique")
    return True


def test_metadata_preservation():
    """Test: Metadata is preserved correctly."""
    print("\n" + "="*80)
    print("TEST 3: Metadata Preservation")
    print("="*80)
    
    print("\n📂 Loading HybridQA dev data...")
    records = load_hybridqa_data("dev")
    sample_records = records[:2]
    
    print("\n🔧 Testing metadata fields...")
    
    docs = build_graphrag_corpus(sample_records, strategy="row_centric")
    
    required_fields = ["table_id", "question_id", "strategy"]
    
    for doc in docs:
        metadata = doc["metadata"]
        
        for field in required_fields:
            if field not in metadata:
                print(f"  ✗ FAILED: Missing metadata field: {field}")
                print(f"    Document: {doc['id']}")
                print(f"    Metadata: {metadata}")
                return False
    
    print(f"  ✓ All {len(docs)} documents have required metadata fields")
    print(f"  ℹ Sample metadata: {docs[0]['metadata']}")
    
    print("\n✅ TEST 3 PASSED: Metadata is preserved correctly")
    return True


def test_corpus_saving_and_loading():
    """Test: Corpus can be saved and loaded."""
    print("\n" + "="*80)
    print("TEST 4: Corpus Saving and Loading")
    print("="*80)
    
    print("\n📂 Loading HybridQA dev data...")
    records = load_hybridqa_data("dev")
    sample_records = records[:2]
    
    temp_dir = PROJECT_ROOT / "results" / "test_graphrag_output"
    
    # Test txt format
    print("\n🔧 Testing .txt format...")
    txt_dir = temp_dir / "txt_format"
    docs = build_graphrag_corpus(sample_records, strategy="row_centric")
    
    try:
        save_graphrag_corpus(docs, txt_dir, format="txt")
        print(f"  ✓ Saved {len(docs)} documents as .txt files")
        
        # Verify files exist
        saved_files = list(txt_dir.glob("*.txt"))
        if len(saved_files) != len(docs):
            print(f"  ✗ FAILED: Expected {len(docs)} files, found {len(saved_files)}")
            return False
        
        print(f"  ✓ All {len(saved_files)} .txt files exist")
        
        # Verify summary exists
        summary_path = txt_dir / "corpus_summary.json"
        if not summary_path.exists():
            print(f"  ✗ FAILED: Summary file not found")
            return False
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        print(f"  ✓ Summary file exists")
        print(f"  ℹ Summary: {summary}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test jsonl format
    print("\n🔧 Testing .jsonl format...")
    jsonl_dir = temp_dir / "jsonl_format"
    
    try:
        save_graphrag_corpus(docs, jsonl_dir, format="jsonl")
        print(f"  ✓ Saved {len(docs)} documents as .jsonl")
        
        # Verify file exists
        jsonl_path = jsonl_dir / "corpus.jsonl"
        if not jsonl_path.exists():
            print(f"  ✗ FAILED: corpus.jsonl not found")
            return False
        
        # Count lines
        with open(jsonl_path) as f:
            lines = f.readlines()
        
        if len(lines) != len(docs):
            print(f"  ✗ FAILED: Expected {len(docs)} lines, found {len(lines)}")
            return False
        
        print(f"  ✓ corpus.jsonl has {len(lines)} lines")
        
        # Verify we can parse
        parsed_docs = [json.loads(line) for line in lines]
        print(f"  ✓ All {len(parsed_docs)} documents can be parsed")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 4 PASSED: Corpus can be saved and loaded")
    return True


def test_strategy_comparison():
    """Test: Strategies can be compared."""
    print("\n" + "="*80)
    print("TEST 5: Strategy Comparison")
    print("="*80)
    
    print("\n📂 Loading HybridQA dev data...")
    records = load_hybridqa_data("dev")
    
    print("\n🔧 Comparing strategies on 5 records...")
    
    try:
        comparison = compare_strategies(records, sample_size=5)
        
        print(f"\n✓ Comparison completed for {len(comparison)} strategies")
        
        # Pretty print comparison
        print("\n" + "-"*80)
        print(f"{'Strategy':<20} {'Docs':<10} {'Avg Len':<12} {'Total Chars':<15}")
        print("-"*80)
        
        for strategy, stats in comparison.items():
            print(
                f"{strategy:<20} "
                f"{stats['num_documents']:<10} "
                f"{stats['avg_doc_length_chars']:<12.0f} "
                f"{stats['total_chars']:<15}"
            )
        
        print("-"*80)
        
        # Verify all strategies have results
        expected_strategies = ["row_centric", "table_centric", "entity_centric", "hybrid"]
        for strategy in expected_strategies:
            if strategy not in comparison:
                print(f"  ✗ FAILED: Missing results for strategy: {strategy}")
                return False
        
        print(f"\n✓ All {len(expected_strategies)} strategies compared")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 5 PASSED: Strategies can be compared")
    return True


def test_convenience_function():
    """Test: Convenience function works end-to-end."""
    print("\n" + "="*80)
    print("TEST 6: Convenience Function")
    print("="*80)
    
    print("\n📂 Loading HybridQA dev data...")
    records = load_hybridqa_data("dev")
    sample_records = records[:3]
    
    output_dir = PROJECT_ROOT / "results" / "test_graphrag_output" / "convenience_test"
    
    print("\n🔧 Running convenience function...")
    
    try:
        docs = hybridqa_to_improved_graphrag_docs(
            sample_records,
            output_dir,
            strategy="hybrid",
            format="txt",
        )
        
        print(f"\n✓ Convenience function completed")
        print(f"  ℹ Generated {len(docs)} documents")
        
        # Verify output exists
        saved_files = list(output_dir.glob("*.txt"))
        if len(saved_files) != len(docs):
            print(f"  ✗ FAILED: Expected {len(docs)} files, found {len(saved_files)}")
            return False
        
        print(f"  ✓ All {len(saved_files)} files saved")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print("\n✅ TEST 6 PASSED: Convenience function works end-to-end")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 4: GraphRAG Source Construction Strategies - Test Suite")
    print("="*80)
    
    tests = [
        test_basic_corpus_building,
        test_document_id_stability,
        test_metadata_preservation,
        test_corpus_saving_and_loading,
        test_strategy_comparison,
        test_convenience_function,
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
        print("\n🎉 ALL TESTS PASSED! Phase 4 is ready for integration.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
