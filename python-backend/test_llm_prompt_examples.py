"""
Display formatted examples of LLM prompts for documentation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'llm'))

import prompts


def show_split_prompt_example():
    """Show a complete split refinement prompt example"""
    print("=" * 80)
    print("SPLIT REFINEMENT PROMPT EXAMPLE")
    print("=" * 80)
    print()

    before_pages = [
        {
            'page_num': 10,
            'text': '''Employment Agreement

This Employment Agreement ("Agreement") is entered into between Acme Corporation
("Employer") and Jane Smith ("Employee"), effective as of January 15, 2024.

1. POSITION AND DUTIES
   Employee will serve as Senior Software Engineer...

Page 10'''
        },
        {
            'page_num': 11,
            'text': '''5. TERMINATION
   Either party may terminate this agreement with 30 days written notice...

6. CONFIDENTIALITY
   Employee agrees to maintain confidentiality of all proprietary information...

IN WITNESS WHEREOF, the parties have executed this Agreement.

Signed: ________________
Date: January 15, 2024

Page 11'''
        }
    ]

    after_pages = [
        {
            'page_num': 12,
            'text': '''INVOICE

Acme Corporation
123 Business Street
Commerce City, CA 90210

Date: June 15, 2024
Invoice Number: INV-2024-0615

BILL TO:
Widget Industries Inc.
456 Industry Avenue

Page 1'''
        },
        {
            'page_num': 13,
            'text': '''DESCRIPTION                           AMOUNT
--------------------------------------------------
Professional Services - June 2024     $15,000.00
Cloud Hosting                          $2,500.00
Support & Maintenance                  $3,000.00

TOTAL DUE:                            $20,500.00

Page 2'''
        }
    ]

    signals = [
        "Page number reset: 11 -> 1",
        "Both header and footer changed",
        "Low semantic similarity (0.23)",
        "Blank separator page detected"
    ]

    prompt = prompts.format_split_prompt(
        split_page=12,
        before_pages=before_pages,
        after_pages=after_pages,
        heuristic_signals=signals
    )

    print(prompt)
    print()
    print("Expected Response:")
    print("YES - Page numbering resets to 1, content switches from employment agreement to invoice, and header/footer completely changed")


def show_naming_prompt_example():
    """Show a complete document naming prompt example"""
    print("\n" + "=" * 80)
    print("DOCUMENT NAMING PROMPT EXAMPLE")
    print("=" * 80)
    print()

    first_page = '''INVOICE

Acme Corporation
123 Business Street, Commerce City, CA 90210
Phone: (555) 123-4567
Email: billing@acmecorp.com

Date: June 15, 2024
Invoice Number: INV-2024-0615
Due Date: July 15, 2024

BILL TO:
Widget Industries Inc.
456 Industry Avenue
Manufacturing Town, TX 75001

RE: Professional Services - June 2024'''

    second_page = '''DESCRIPTION                                    QUANTITY    RATE        AMOUNT
--------------------------------------------------------------------------------
Software Development Services                   160 hrs    $125/hr    $20,000.00
  - Backend API implementation
  - Frontend dashboard development
  - Database optimization

Technical Consulting                             40 hrs    $150/hr     $6,000.00
  - Architecture review
  - Performance optimization

Cloud Infrastructure                              1 mo    $2,500/mo    $2,500.00

SUBTOTAL:                                                              $28,500.00
TAX (8.5%):                                                             $2,422.50
TOTAL DUE:                                                            $30,922.50'''

    prompt = prompts.format_naming_prompt(
        start_page=12,
        end_page=14,
        first_page_text=first_page,
        second_page_text=second_page
    )

    print(prompt)
    print()
    print("Expected Response:")
    print("2024-06-15_Invoice_Acme Corp Professional Services")


def show_parsing_examples():
    """Show response parsing examples"""
    print("\n" + "=" * 80)
    print("RESPONSE PARSING EXAMPLES")
    print("=" * 80)
    print()

    print("1. Split Decision Parsing:")
    print("-" * 80)

    test_responses = [
        "YES - Page numbering resets and header changes indicate new document",
        "NO - Content continues same topic with consistent formatting",
        "YES: Clear document boundary detected with multiple signals",
        "NO: Pages are part of same contract, just different sections",
    ]

    for response in test_responses:
        should_split, reasoning = prompts.parse_split_decision(response)
        print(f"Response: '{response}'")
        print(f"  -> Should Split: {should_split}")
        print(f"  -> Reasoning: {reasoning}")
        print()

    print("\n2. Filename Parsing:")
    print("-" * 80)

    test_filenames = [
        "2024-06-15_Invoice_Acme Corp Professional Services",
        '"2024-03-20_Contract_Software License Agreement"',
        "UNDATED_Report_Annual Financial Summary.pdf",
        "2023-12-01_Letter_Employment Offer Jane Smith",
    ]

    for raw_filename in test_filenames:
        parsed = prompts.parse_filename(raw_filename)
        is_valid = prompts.validate_filename(parsed)
        print(f"Input:  '{raw_filename}'")
        print(f"Parsed: '{parsed}'")
        print(f"Valid:  {is_valid}")
        print()


def main():
    print("\n" + "=" * 80)
    print("LLM PROMPT EXAMPLES FOR DOCUMENT DE-BUNDLING")
    print("=" * 80)
    print()

    show_split_prompt_example()
    show_naming_prompt_example()
    show_parsing_examples()

    print("=" * 80)
    print("Examples Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
