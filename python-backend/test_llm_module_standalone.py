"""
Standalone test script for LLM module functionality
Does not require dependencies beyond standard library
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_config_selection():
    """Test LLM configuration selection for different VRAM levels"""
    print("=" * 80)
    print("Testing LLM Configuration Selection")
    print("=" * 80)

    # Mock the hardware detection to avoid dependency
    import unittest.mock as mock

    with mock.patch('services.ocr.config.detect_hardware_capabilities') as mock_detect:
        from services.llm.config import (
            select_optimal_llm_config,
            get_model_download_info,
            get_generation_params,
            estimate_inference_time
        )

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
            config = select_optimal_llm_config(gpu_memory_gb=vram_gb)

            print(f"  Model: {config['model_name']}")
            print(f"  Model ID: {config['model_id']}")
            print(f"  Quantization: {config['quantization']}")
            print(f"  GPU Layers: {config['n_gpu_layers']}")
            print(f"  Batch Size: {config['n_batch']}")
            print(f"  Context Size: {config['n_ctx']}")
            print(f"  Expected VRAM: {config['expected_vram_gb']} GB")
            print(f"  Offload Strategy: {config['offload_strategy']}")

            if 'cpu_layers' in config:
                print(f"  CPU Layers: {config['cpu_layers']}")
            if 'warning' in config:
                print(f"  WARNING: {config['warning']}")

            # Get download info
            download_info = get_model_download_info(config)
            print(f"  Download URL: {download_info['model_url']}")

            # Get generation params
            for task_type in ['split_refinement', 'naming']:
                params = get_generation_params(task_type)
                print(f"  {task_type.title()} params: temp={params['temperature']}, max_tokens={params['max_tokens']}")

            # Estimate inference time
            est_time = estimate_inference_time(config, num_tokens=100)
            print(f"  Estimated time (100 tokens): {est_time:.2f}s")


def test_prompt_formatting():
    """Test prompt formatting functions"""
    print("\n" + "=" * 80)
    print("Testing Prompt Formatting")
    print("=" * 80)

    from services.llm.prompts import (
        format_split_prompt,
        format_naming_prompt,
        parse_split_decision,
        parse_filename,
        validate_filename
    )

    # Test split refinement prompt
    print("\n1. Split Refinement Prompt:")
    print("-" * 80)

    before_pages = [
        {'page_num': 3, 'text': 'This is the third page of the first document. Page 3'},
        {'page_num': 4, 'text': 'This is the fourth page. Page 4. Final page of first doc.'},
    ]

    after_pages = [
        {'page_num': 5, 'text': 'Page 1 of second document. New header: Acme Corp Invoice'},
        {'page_num': 6, 'text': 'Page 2. Invoice details continue here.'},
    ]

    heuristic_signals = [
        "Page number reset: 4 -> 1",
        "Both header and footer changed",
        "Low semantic similarity (0.25)",
    ]

    split_prompt = format_split_prompt(
        split_page=5,
        before_pages=before_pages,
        after_pages=after_pages,
        heuristic_signals=heuristic_signals
    )

    print(split_prompt[:500] + "...")

    # Test naming prompt
    print("\n2. Document Naming Prompt:")
    print("-" * 80)

    first_page = """
    INVOICE

    Acme Corporation
    123 Business St, Commerce City, CA 90210

    Date: 2024-06-15
    Invoice #: INV-2024-0615

    Bill To:
    Widget Industries
    456 Industry Ave, Manufacturing Town, TX 75001

    Services Rendered - June 2024
    """

    second_page = """
    Description                    Amount
    --------------------------------
    Consulting Services            $5,000
    Support & Maintenance          $2,500
    Cloud Hosting                  $1,200

    Total Due: $8,700
    """

    naming_prompt = format_naming_prompt(
        start_page=5,
        end_page=7,
        first_page_text=first_page,
        second_page_text=second_page
    )

    print(naming_prompt[:500] + "...")


def test_response_parsing():
    """Test parsing of LLM responses"""
    print("\n" + "=" * 80)
    print("Testing Response Parsing")
    print("=" * 80)

    from services.llm.prompts import (
        parse_split_decision,
        parse_filename,
        validate_filename
    )

    # Test split decision parsing
    print("\n1. Split Decision Parsing:")
    print("-" * 80)

    test_responses = [
        "YES - Page numbering resets and header changes indicate new document",
        "NO - Content continues same topic with consistent formatting",
        "YES: Clear document boundary detected",
        "NO: Pages are part of same document",
        "Maybe this is unclear",  # Edge case
    ]

    for response in test_responses:
        should_split, reasoning = parse_split_decision(response)
        print(f"  Input: '{response[:50]}...'")
        print(f"  -> Split: {should_split}, Reasoning: {reasoning}")

    # Test filename parsing
    print("\n2. Filename Parsing:")
    print("-" * 80)

    test_filenames = [
        "2024-06-15_Invoice_Acme Corp June Services",
        '"2024-03-20_Contract_Software License Agreement"',
        "UNDATED_Report_Annual Financial Summary.pdf",
        "2023-12-01_Letter_Employment Offer John Smith",
    ]

    for raw_filename in test_filenames:
        parsed = parse_filename(raw_filename)
        is_valid = validate_filename(parsed)
        print(f"  Input: '{raw_filename}'")
        print(f"  -> Parsed: '{parsed}'")
        print(f"  -> Valid: {is_valid}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("LLM Module Test Suite")
    print("=" * 80)

    test_config_selection()
    test_prompt_formatting()
    test_response_parsing()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
