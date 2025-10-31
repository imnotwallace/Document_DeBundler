"""
Simple test for LLM module - imports only what's needed
"""

# Direct imports to avoid services.__init__.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Mock the detect_hardware_capabilities function before importing
class MockHardwareCapabilities:
    @staticmethod
    def detect_hardware_capabilities():
        return {
            'gpu_available': True,
            'cuda_available': True,
            'directml_available': False,
            'gpu_memory_gb': 4.0,
            'system_memory_gb': 16.0,
            'cpu_count': 8,
            'platform': 'Windows'
        }

# Inject mock
sys.modules['services.ocr.config'] = MockHardwareCapabilities()

# Now we can import the llm config directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'llm'))

import config
import prompts


def test_config_selection():
    """Test LLM configuration selection"""
    print("=" * 80)
    print("LLM Configuration Selection Test")
    print("=" * 80)

    vram_levels = [
        (0, "No GPU (CPU only)"),
        (2, "2GB VRAM (Low-end GPU)"),
        (4, "4GB VRAM (Target hardware)"),
        (6, "6GB VRAM (Mid-range GPU)"),
        (8, "8GB VRAM (High-end GPU)"),
    ]

    for vram_gb, description in vram_levels:
        print(f"\n{description}:")
        print("-" * 80)
        cfg = config.select_optimal_llm_config(gpu_memory_gb=vram_gb)

        print(f"  Model: {cfg['model_name']}")
        print(f"  Model ID: {cfg['model_id']}")
        print(f"  Quantization: {cfg['quantization']}")
        print(f"  GPU Layers: {cfg['n_gpu_layers']}")
        print(f"  Expected VRAM: {cfg['expected_vram_gb']} GB")
        print(f"  Offload Strategy: {cfg['offload_strategy']}")

        if 'cpu_layers' in cfg:
            print(f"  CPU Layers: {cfg['cpu_layers']}")

        # Get download info
        download_info = config.get_model_download_info(cfg)
        print(f"  Model file: {download_info['model_file']}")

        # Get generation params
        split_params = config.get_generation_params('split_refinement')
        naming_params = config.get_generation_params('naming')
        print(f"  Split params: temp={split_params['temperature']}, max_tokens={split_params['max_tokens']}")
        print(f"  Naming params: temp={naming_params['temperature']}, max_tokens={naming_params['max_tokens']}")


def test_prompts():
    """Test prompt formatting"""
    print("\n" + "=" * 80)
    print("Prompt Formatting Test")
    print("=" * 80)

    # Test split prompt
    print("\n1. Split Refinement Prompt Example:")
    print("-" * 80)

    before_pages = [
        {'page_num': 4, 'text': 'This is page 4 of the first document. Contains important contract details.'},
    ]
    after_pages = [
        {'page_num': 5, 'text': 'Page 1. INVOICE - Acme Corporation. Date: 2024-06-15'},
    ]
    signals = ["Page number reset: 4 -> 1", "Header changed"]

    split_prompt = prompts.format_split_prompt(5, before_pages, after_pages, signals)
    print(split_prompt[:400] + "...")

    # Test naming prompt
    print("\n2. Document Naming Prompt Example:")
    print("-" * 80)

    first_page = "INVOICE\n\nAcme Corp\nDate: 2024-06-15\nInvoice #: INV-001"
    naming_prompt = prompts.format_naming_prompt(1, 3, first_page, "Line items...")
    print(naming_prompt[:400] + "...")


def test_parsing():
    """Test response parsing"""
    print("\n" + "=" * 80)
    print("Response Parsing Test")
    print("=" * 80)

    # Test split decision parsing
    print("\n1. Split Decision Parsing:")
    print("-" * 80)

    test_cases = [
        "YES - Page numbering resets",
        "NO - Content continues",
        "YES: Clear boundary",
        "Unclear response",
    ]

    for test in test_cases:
        should_split, reasoning = prompts.parse_split_decision(test)
        print(f"  '{test[:30]}...' -> Split: {should_split}")

    # Test filename parsing
    print("\n2. Filename Parsing:")
    print("-" * 80)

    filenames = [
        "2024-06-15_Invoice_Acme Corp Services",
        '"2024-03-20_Contract_Software License Agreement"',
        "UNDATED_Report_Financial Summary.pdf",
    ]

    for fn in filenames:
        parsed = prompts.parse_filename(fn)
        valid = prompts.validate_filename(parsed)
        print(f"  '{fn}' -> '{parsed}' (valid: {valid})")


def main():
    print("\n" + "=" * 80)
    print("LLM Module Test Suite")
    print("=" * 80)

    test_config_selection()
    test_prompts()
    test_parsing()

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
