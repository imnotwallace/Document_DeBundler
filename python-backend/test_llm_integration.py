"""
Integration Tests for LLM Features
Run with: python test_llm_integration.py
"""

import logging
logging.basicConfig(level=logging.INFO)

def test_model_availability():
    """Test 1: Check if models are available"""
    print("\n" + "="*60)
    print("Test 1: Model Availability")
    print("="*60)

    from services.resource_path import verify_llm_models

    models = verify_llm_models()
    print(f"Phi-3 Mini: {'✓ Found' if models['phi3_mini'] else '✗ Not found'}")
    print(f"Gemma 2 2B: {'✓ Found' if models['gemma2_2b'] else '✗ Not found'}")

    if not any(models.values()):
        print("\n⚠️  No models found. Run: python download_llm_models.py")
        return False

    return True

def test_loader():
    """Test 2: Load model and generate text"""
    print("\n" + "="*60)
    print("Test 2: Model Loading")
    print("="*60)

    from services.llm.loader import LlamaLoader

    with LlamaLoader(use_gpu=True) as loader:
        if not loader.load_model():
            print("✗ Failed to load model")
            return False

        info = loader.get_model_info()
        print(f"✓ Model loaded: {info['model_name']}")
        print(f"  Mode: {info['mode']}")
        print(f"  GPU: {info['gpu_enabled']}")

        # Test generation
        response = loader.generate("The capital of France is", max_tokens=10)
        print(f"  Test generation: {response}")

        return True

def test_manager():
    """Test 3: Manager singleton and generation"""
    print("\n" + "="*60)
    print("Test 3: LLM Manager")
    print("="*60)

    from services.llm.manager import get_llm_manager

    manager = get_llm_manager()

    if not manager.initialize():
        print("✗ Failed to initialize")
        return False

    response = manager.generate(
        "What is 2+2?",
        task_type="general",
        max_tokens=20
    )

    print(f"✓ Response: {response}")

    stats = manager.stats
    print(f"  Total generations: {stats['total_generations']}")
    print(f"  Successful: {stats['successful_generations']}")

    manager.cleanup()
    return True

def test_split_analyzer():
    """Test 4: Split analysis"""
    print("\n" + "="*60)
    print("Test 4: Split Analyzer")
    print("="*60)

    from services.llm.split_analyzer import SplitAnalyzer

    page_texts = [
        "Page 1: This is a contract. Terms and conditions...",
        "Page 2: Continued contract terms...",
        "Page 3: Final contract page. Signatures...",
        "Page 4: INVOICE #001 Date: 2024-01-15 Amount: $500",
        "Page 5: Invoice line items...",
    ]

    analyzer = SplitAnalyzer()
    should_split, confidence, reasoning = analyzer.analyze_split(
        split_page=3,
        page_texts=page_texts,
        heuristic_signals=["Content type changed", "Low similarity (0.15)"]
    )

    print(f"✓ Split decision: {should_split}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Reasoning: {reasoning}")

    return True

def test_name_generator():
    """Test 5: Name generation"""
    print("\n" + "="*60)
    print("Test 5: Name Generator")
    print("="*60)

    from services.llm.name_generator import NameGenerator

    invoice_text = """
    INVOICE

    Invoice Number: INV-2024-001
    Date: January 15, 2024

    Bill To: Acme Corporation
    Services: Consulting Q4 2023
    """

    generator = NameGenerator()
    filename = generator.generate_name(
        first_page_text=invoice_text,
        start_page=0,
        end_page=2
    )

    print(f"✓ Generated filename: {filename}")

    metadata = generator.extract_metadata(filename)
    print(f"  Date: {metadata['date']}")
    print(f"  Type: {metadata['doctype']}")
    print(f"  Description: {metadata['description']}")

    return True

def test_settings():
    """Test 6: Settings management"""
    print("\n" + "="*60)
    print("Test 6: Settings Management")
    print("="*60)

    from services.llm.settings import get_settings_manager

    manager = get_settings_manager()
    settings = manager.get()

    print(f"✓ LLM Enabled: {settings.enabled}")
    print(f"  Split Refinement: {settings.split_refinement_enabled}")
    print(f"  Naming: {settings.naming_enabled}")
    print(f"  Model Preference: {settings.model_preference}")
    print(f"  Use GPU: {settings.use_gpu}")

    valid, error = manager.validate()
    print(f"  Valid: {valid}")

    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("LLM Integration Test Suite")
    print("="*60)

    tests = [
        ("Model Availability", test_model_availability),
        ("Model Loading", test_loader),
        ("LLM Manager", test_manager),
        ("Split Analyzer", test_split_analyzer),
        ("Name Generator", test_name_generator),
        ("Settings", test_settings),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
