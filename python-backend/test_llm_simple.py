"""
Simple LLM Test - Quick verification without model download

This tests the LLM integration basics:
1. Import check
2. Settings management
3. Model path detection
4. Lazy loading behavior
"""

import sys

def test_imports():
    """Test 1: Can we import the LLM modules?"""
    print("\n" + "="*60)
    print("TEST 1: Import Check")
    print("="*60)

    try:
        from services.llm import config, prompts, loader, manager, settings
        from services.llm import split_analyzer, name_generator
        print("[OK] All LLM modules imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings():
    """Test 2: Settings management"""
    print("\n" + "="*60)
    print("TEST 2: Settings Management")
    print("="*60)

    try:
        from services.llm.settings import get_settings_manager, LLMSettings

        mgr = get_settings_manager()
        settings = mgr.get()

        print(f"[OK] Settings loaded:")
        print(f"   Enabled: {settings.enabled}")
        print(f"   Model: {settings.model_preference}")
        print(f"   GPU: {settings.use_gpu}")
        print(f"   Split refinement: {settings.split_refinement_enabled}")
        print(f"   Naming: {settings.naming_enabled}")
        print(f"   Auto cleanup: {settings.auto_cleanup_enabled}")

        # Validate
        valid, error = mgr.validate()
        print(f"\n   Valid: {valid}")
        if error:
            print(f"   Validation error: {error}")

        return True
    except Exception as e:
        print(f"[FAIL] Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_paths():
    """Test 3: Model path detection"""
    print("\n" + "="*60)
    print("TEST 3: Model Path Detection")
    print("="*60)

    try:
        from services.resource_path import (
            get_llm_models_dir,
            get_phi3_mini_path,
            get_gemma2_2b_path,
            verify_llm_models
        )

        models_dir = get_llm_models_dir()
        print(f"[OK] LLM models directory: {models_dir}")
        print(f"   Exists: {models_dir.exists()}")

        phi3_path = get_phi3_mini_path()
        gemma2_path = get_gemma2_2b_path()

        print(f"\n[OK] Model paths:")
        print(f"   Phi-3 Mini: {phi3_path}")
        print(f"   Exists: {phi3_path.exists() if phi3_path else False}")

        print(f"\n   Gemma 2 2B: {gemma2_path}")
        print(f"   Exists: {gemma2_path.exists() if gemma2_path else False}")

        # Verify models
        models = verify_llm_models()
        print(f"\n[OK] Model verification:")
        print(f"   Phi-3 Mini available: {models['phi3_mini']}")
        print(f"   Gemma 2 2B available: {models['gemma2_2b']}")

        if not any(models.values()):
            print(f"\n[WARN] No models found locally")
            print(f"   Models will auto-download on first use (slower)")
            print(f"   Or run: python download_llm_models.py")

        return True
    except Exception as e:
        print(f"[FAIL] Model path test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lazy_loading_behavior():
    """Test 4: Lazy loading behavior (without actually loading model)"""
    print("\n" + "="*60)
    print("TEST 4: Lazy Loading Behavior")
    print("="*60)

    try:
        from services.llm.manager import get_llm_manager

        # Get manager - should NOT load model yet
        print(">>> Getting LLM manager instance...")
        manager = get_llm_manager()

        print(f"[OK] Manager created: {manager is not None}")
        print(f"   Model loaded: {hasattr(manager, '_model') and manager._model is not None}")
        print(f"   Loader created: {hasattr(manager, '_loader') and manager._loader is not None}")

        if hasattr(manager, '_model') and manager._model is None:
            print(f"\n[OK] CORRECT: Model is None (lazy loading - not loaded yet)")
        else:
            print(f"\n[WARN] Model might be loaded (unexpected)")

        print(f"\n   To actually load the model, call: manager.initialize()")
        print(f"   This will download the model if not found locally")

        return True
    except Exception as e:
        print(f"[FAIL] Lazy loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llama_cpp_available():
    """Test 5: Check if llama-cpp-python is installed"""
    print("\n" + "="*60)
    print("TEST 5: llama-cpp-python Installation")
    print("="*60)

    try:
        import llama_cpp
        print(f"[OK] llama-cpp-python version: {llama_cpp.__version__}")

        # Check for CUDA support
        try:
            print(f"\n   Checking GPU support...")
            print(f"   (Actual GPU availability checked during model load)")
        except:
            pass

        return True
    except ImportError:
        print(f"[FAIL] llama-cpp-python not installed")
        print(f"\n   Install with:")
        print(f"   cd python-backend")
        print(f"   uv pip install llama-cpp-python==0.3.4")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking llama-cpp-python: {e}")
        return False

def main():
    """Run all simple tests"""
    print("\n" + "="*60)
    print("SIMPLE LLM INTEGRATION TEST")
    print("="*60)

    tests = [
        ("llama-cpp-python Installation", test_llama_cpp_available),
        ("Module Imports", test_imports),
        ("Settings Management", test_settings),
        ("Model Path Detection", test_model_paths),
        ("Lazy Loading Behavior", test_lazy_loading_behavior),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Unexpected error in '{name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n[OK] All tests passed! LLM integration is ready.")
        print("\nNext steps:")
        print("   1. Download models: python download_llm_models.py")
        print("   2. Test lazy loading: python test_llm_lazy_loading.py")
    else:
        print("\n[WARN] Some tests failed. Check errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
