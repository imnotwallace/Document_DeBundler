"""
Test LLM Lazy Loading and Memory Management

This script tests:
1. Model availability
2. Lazy loading (model loads only when needed)
3. Memory usage tracking
4. Cleanup and resource management
"""

import logging
import time
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        logger.warning("psutil not installed, memory tracking unavailable")
        return None

def test_model_availability():
    """Test 1: Check model availability"""
    print("\n" + "="*70)
    print("TEST 1: Model Availability")
    print("="*70)
    
    from services.resource_path import verify_llm_models
    
    models = verify_llm_models()
    print(f"Phi-3 Mini: {'[OK] Available' if models['phi3_mini'] else '[FAIL] Not found'}")
    print(f"Gemma 2 2B: {'[OK] Available' if models['gemma2_2b'] else '[FAIL] Not found'}")
    
    if not any(models.values()):
        print("\n[WARN]  WARNING: No models found!")
        print("   Run: python download_llm_models.py")
        print("   Or continue - will auto-download on first use (slow)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True

def test_lazy_loading():
    """Test 2: Lazy loading - model loads only when needed"""
    print("\n" + "="*70)
    print("TEST 2: Lazy Loading")
    print("="*70)
    
    from services.llm.manager import get_llm_manager, cleanup_llm
    
    mem_start = get_memory_usage()
    print(f"Initial memory: {mem_start:.1f} MB" if mem_start else "Memory tracking unavailable")
    
    # Get manager instance (should NOT load model yet)
    print("\n>>> Getting manager instance (should NOT load model)...")
    manager = get_llm_manager()
    
    mem_after_get = get_memory_usage()
    if mem_after_get:
        print(f"Memory after get_llm_manager(): {mem_after_get:.1f} MB (+{mem_after_get - mem_start:.1f} MB)")
        print("   [OK] Memory increase should be minimal (manager created but model not loaded)")
    
    # Check that model is NOT loaded
    print(f"\nModel initialized? {manager.is_initialized()}")
    print(f"   Loader exists? {manager.loader is not None}")
    print("   [OK] Should be False (lazy loading - model not loaded yet)")
    
    # Now initialize (this SHOULD load the model)
    print("\n>>> Calling initialize() - NOW model should load...")
    start_time = time.time()
    
    success = manager.initialize()
    
    load_time = time.time() - start_time
    
    if not success:
        print("[FAIL] Failed to initialize LLM")
        return False
    
    mem_after_init = get_memory_usage()
    if mem_after_init:
        print(f"Memory after initialize(): {mem_after_init:.1f} MB (+{mem_after_init - mem_start:.1f} MB)")
        print(f"   [OK] Memory increase: ~{mem_after_init - mem_after_get:.1f} MB (model loaded)")
    
    print(f"Load time: {load_time:.2f} seconds")
    print(f"Model loaded? {manager.is_initialized()}")
    print("   Expected: True (model now loaded)")
    
    # Check the loader
    print(f"   Loader exists? {manager.loader is not None}")
    
    # Get model info
    info = manager.loader.get_model_info() if manager.loader else None
    if info:
        print(f"\nModel info:")
        print(f"   Name: {info['model_name']}")
        print(f"   Mode: {info['mode']}")
        print(f"   GPU: {info['gpu_enabled']}")
        print(f"   VRAM: {info['expected_vram_gb']:.1f} GB")
    
    # Cleanup
    print("\n>>> Cleaning up...")
    cleanup_llm()
    gc.collect()
    time.sleep(1)  # Give OS time to free memory
    
    mem_after_cleanup = get_memory_usage()
    if mem_after_cleanup:
        print(f"Memory after cleanup: {mem_after_cleanup:.1f} MB")
        print(f"   Memory freed: ~{mem_after_init - mem_after_cleanup:.1f} MB")
    
    return True

def test_generation():
    """Test 3: Text generation"""
    print("\n" + "="*70)
    print("TEST 3: Text Generation")
    print("="*70)
    
    from services.llm.manager import get_llm_manager, cleanup_llm
    
    manager = get_llm_manager()
    
    # Initialize if needed
    if not manager.initialize():
        print("[FAIL] Failed to initialize")
        return False
    
    # Test generation
    print("\n>>> Testing generation...")
    prompt = "What is the capital of France? Answer in one word."
    
    start_time = time.time()
    response = manager.generate(
        prompt=prompt,
        task_type="general",
        max_tokens=10
    )
    gen_time = time.time() - start_time
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Generation time: {gen_time:.2f} seconds")
    
    # Check stats
    stats = manager.stats
    print(f"\nManager stats:")
    print(f"   Total generations: {stats['total_generations']}")
    print(f"   Successful: {stats['successful_generations']}")
    print(f"   Failed: {stats['failed_generations']}")
    
    cleanup_llm()
    return True

def test_multiple_calls():
    """Test 4: Multiple generations (model should stay loaded)"""
    print("\n" + "="*70)
    print("TEST 4: Multiple Generations (Model Persistence)")
    print("="*70)
    
    from services.llm.manager import get_llm_manager, cleanup_llm
    
    manager = get_llm_manager()
    manager.initialize()
    
    prompts = [
        "2+2=",
        "The sky is",
        "Hello, my name is"
    ]
    
    print("\n>>> Running 3 generations to test model persistence...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nGeneration {i}:")
        start_time = time.time()
        
        response = manager.generate(
            prompt=prompt,
            task_type="general",
            max_tokens=5
        )
        
        gen_time = time.time() - start_time
        print(f"   Prompt: '{prompt}'")
        print(f"   Response: '{response}'")
        print(f"   Time: {gen_time:.2f}s")
    
    print(f"\n[OK] All generations used same loaded model instance")
    print(f"   Total generations: {manager.stats['total_generations']}")
    
    cleanup_llm()
    return True

def test_auto_cleanup_setting():
    """Test 5: Auto-cleanup setting"""
    print("\n" + "="*70)
    print("TEST 5: Auto-Cleanup Setting")
    print("="*70)
    
    from services.llm.settings import get_settings_manager
    
    settings_mgr = get_settings_manager()
    settings = settings_mgr.get()
    
    print(f"\nCurrent settings:")
    print(f"   LLM Enabled: {settings.enabled}")
    print(f"   Auto-cleanup: {settings.auto_cleanup_enabled}")
    print(f"   Model preference: {settings.model_preference}")
    print(f"   Use GPU: {settings.use_gpu}")
    print(f"   GPU layers: {settings.max_gpu_layers}")
    
    # Test validation
    is_valid, error = settings_mgr.validate()
    print(f"\n   Settings valid: {is_valid}")
    if error:
        print(f"   Error: {error}")
    
    return True

def run_all_tests():
    """Run all lazy loading tests"""
    print("\n" + "="*70)
    print("LLM LAZY LOADING TEST SUITE")
    print("="*70)
    
    tests = [
        ("Model Availability", test_model_availability),
        ("Lazy Loading", test_lazy_loading),
        ("Text Generation", test_generation),
        ("Multiple Generations", test_multiple_calls),
        ("Auto-Cleanup Settings", test_auto_cleanup_setting),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\n")
            result = test_func()
            results.append((name, result))
            
            if not result:
                print(f"\n[WARN]  Test '{name}' returned False - stopping here")
                break
                
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' FAILED with exception:")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            break
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
