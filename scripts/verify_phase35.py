"""
Quick verification script for Phase 3.5 model switching.

Tests:
1. Model variants can be imported and listed
2. Model metadata is tracked correctly
3. Environment variables are set correctly
4. Experiment metadata includes model info

Usage:
    python scripts/verify_phase35.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model_variants():
    """Test 1: Model variants system"""
    print("\n" + "="*80)
    print("TEST 1: Model Variants System")
    print("="*80)
    
    try:
        from src.config import model_variants
        
        print(f"✓ Successfully imported model_variants")
        print(f"✓ Found {len(model_variants.ALL_VARIANTS)} variants")
        
        # Test getting a variant
        variant = model_variants.get_variant("mistral_7b_e5_base")
        print(f"✓ Retrieved variant: {variant.name}")
        print(f"  - Generation model: {variant.generation_model}")
        print(f"  - Embedding model: {variant.embedding_model}")
        print(f"  - Family: {variant.model_family}")
        print(f"  - Size: {variant.model_size_b}B")
        
        # Test env dict conversion
        env_dict = variant.to_env_dict()
        print(f"✓ Environment dict has {len(env_dict)} variables")
        for key, value in env_dict.items():
            print(f"  - {key}={value}")
        
        # Test variant collections
        print(f"✓ Minimum ablation variants: {len(model_variants.MINIMUM_ABLATION_VARIANTS)}")
        print(f"✓ Extended ablation variants: {len(model_variants.EXTENDED_ABLATION_VARIANTS)}")
        print(f"✓ Smoke test variants: {len(model_variants.SMOKE_TEST_VARIANTS)}")
        
        print("\n✓ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_metadata():
    """Test 2: Model metadata tracking"""
    print("\n" + "="*80)
    print("TEST 2: Model Metadata Tracking")
    print("="*80)
    
    try:
        from src.config import model_registry
        
        # Test active model metadata
        metadata = model_registry.get_active_model_metadata()
        print(f"✓ Retrieved active model metadata")
        print(f"  - Backend: {metadata['backend']}")
        print(f"  - Generation model: {metadata['generation_model']}")
        print(f"  - Embedding model: {metadata['embedding_model']}")
        print(f"  - Variant name: {metadata.get('variant_name', 'custom')}")
        
        # Test experiment metadata building
        exp_metadata = model_registry.build_experiment_metadata(
            experiment_name="phase35_verification",
            additional_metadata={"test": True},
        )
        print(f"✓ Built experiment metadata")
        print(f"  - Has hardware info: {'hardware' in exp_metadata}")
        print(f"  - Has runtime info: {'runtime' in exp_metadata}")
        print(f"  - Has model info: {'models' in exp_metadata}")
        print(f"  - GPU VRAM: {exp_metadata['hardware'].get('gpu_vram_gb', 'unknown')}GB")
        print(f"  - CUDA available: {exp_metadata['hardware'].get('cuda_available', False)}")
        
        print("\n✓ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_switching():
    """Test 3: Environment variable switching"""
    print("\n" + "="*80)
    print("TEST 3: Environment Variable Switching")
    print("="*80)
    
    try:
        from src.config import model_variants
        
        # Get a variant
        variant = model_variants.get_variant("qwen_3b_e5_base")
        print(f"✓ Testing with variant: {variant.name}")
        
        # Save current env
        original_env = {
            "LOCAL_GENERATION_MODEL": os.environ.get("LOCAL_GENERATION_MODEL", ""),
            "LOCAL_EMBEDDING_MODEL": os.environ.get("LOCAL_EMBEDDING_MODEL", ""),
        }
        print(f"  - Original generation model: {original_env['LOCAL_GENERATION_MODEL']}")
        
        # Apply variant
        env_dict = variant.to_env_dict()
        for key, value in env_dict.items():
            os.environ[key] = value
        
        print(f"✓ Applied variant environment variables")
        print(f"  - New generation model: {os.environ.get('LOCAL_GENERATION_MODEL')}")
        print(f"  - New embedding model: {os.environ.get('LOCAL_EMBEDDING_MODEL')}")
        
        # Verify they match variant
        assert os.environ.get("LOCAL_GENERATION_MODEL") == variant.generation_model
        assert os.environ.get("LOCAL_EMBEDDING_MODEL") == variant.embedding_model
        print(f"✓ Environment variables match variant configuration")
        
        # Restore original env
        for key, value in original_env.items():
            if value:
                os.environ[key] = value
        
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variant_listing():
    """Test 4: Variant listing and filtering"""
    print("\n" + "="*80)
    print("TEST 4: Variant Listing and Filtering")
    print("="*80)
    
    try:
        from src.config import model_variants
        
        # Test listing by tag
        small_variants = model_variants.list_variants(tags=["small"])
        print(f"✓ Found {len(small_variants)} 'small' variants")
        for v in small_variants:
            print(f"  - {v.name} ({v.model_size_b}B)")
        
        # Test listing by VRAM
        low_vram_variants = model_variants.list_variants(max_vram_gb=16)
        print(f"✓ Found {len(low_vram_variants)} variants with ≤16GB VRAM")
        
        # Test hardware recommendation
        recommended = model_variants.get_variant_for_hardware(
            available_vram_gb=16,
            prefer_size="small",
        )
        print(f"✓ Recommended variant for 16GB VRAM: {recommended.name}")
        print(f"  - Model: {recommended.generation_model}")
        print(f"  - Size: {recommended.model_size_b}B")
        print(f"  - VRAM req: {recommended.vram_requirement_gb}GB")
        
        print("\n✓ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("PHASE 3.5 VERIFICATION: Model Registry and Switching System")
    print("="*80)
    
    tests = [
        test_model_variants,
        test_model_metadata,
        test_env_switching,
        test_variant_listing,
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ✓ ✓ ALL TESTS PASSED ✓ ✓ ✓")
        print("\nPhase 3.5 is working correctly!")
        print("\nNext steps:")
        print("  1. List available variants:  python scripts/16_run_model_comparison.py --list-variants")
        print("  2. Run smoke test:          python scripts/16_run_model_comparison.py --split dev --limit 10 --variants smoke")
        print("  3. Proceed to Phase 4")
        return 0
    else:
        print("\n✗ ✗ ✗ SOME TESTS FAILED ✗ ✗ ✗")
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
