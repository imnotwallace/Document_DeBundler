# LLM Module for Document De-bundling

This module provides LLM-based analysis for document de-bundling with VRAM-optimized model selection.

## Files

### `__init__.py`
Module initialization exposing main functions:
- `select_optimal_llm_config()` - VRAM-based model selection
- `get_model_download_info()` - Model download URLs
- `format_split_prompt()` - Format split refinement prompts
- `format_naming_prompt()` - Format document naming prompts

### `config.py`
LLM configuration and model selection based on available VRAM.

**Key Functions:**
- `select_optimal_llm_config(gpu_memory_gb)` - Auto-select best model
- `get_model_download_info(config)` - Get HuggingFace URLs
- `get_generation_params(task_type)` - Task-specific generation parameters
- `estimate_inference_time(config, num_tokens)` - Estimate generation time

**Supported Configurations:**

| VRAM | Model | Quantization | GPU Layers | Expected VRAM | Strategy |
|------|-------|--------------|------------|---------------|----------|
| 8GB+ | Phi-3 Mini | Q5_K_M | 32 | 3.2 GB | GPU only |
| 6GB+ | Phi-3 Mini | Q4_K_M | 32 | 2.5 GB | GPU only |
| 4GB+ | Phi-3 Mini | Q4_K_M | 28 | 2.3 GB | Hybrid (28 GPU + 4 CPU) |
| 2GB+ | Gemma 2 2B | Q4_K_M | 20 | 1.5 GB | Hybrid (20 GPU + 8 CPU) |
| CPU | Phi-3 Mini | Q4_K_M | 0 | 0 GB | CPU only |

**Target Hardware (4GB VRAM):**
- Model: Phi-3 Mini (Q4_K_M quantization)
- GPU Layers: 28 (out of 32 total)
- CPU Layers: 4
- Expected VRAM: 2.3 GB
- Batch Size: 256
- Context: 4096 tokens

### `prompts.py`
Prompt templates and formatting functions for LLM tasks.

**Templates:**
1. `SPLIT_REFINEMENT_PROMPT` - Analyze document boundaries
2. `DOCUMENT_NAMING_PROMPT` - Generate filenames in `{DATE}_{DOCTYPE}_{DESCRIPTION}` format
3. `SPLIT_REASONING_SYSTEM_PROMPT` - System prompt for split analysis
4. `NAMING_SYSTEM_PROMPT` - System prompt for naming

**Functions:**
- `format_split_prompt(split_page, before_pages, after_pages, signals)` - Format split analysis
- `format_naming_prompt(start_page, end_page, first_page_text, second_page_text)` - Format naming
- `parse_split_decision(response)` - Parse YES/NO decision from LLM
- `parse_filename(response)` - Extract and clean filename
- `validate_filename(filename)` - Validate filename format

## Usage Examples

### 1. Select Optimal Configuration

```python
from services.llm.config import select_optimal_llm_config

# Auto-detect VRAM and select model
config = select_optimal_llm_config()

# Or specify VRAM manually
config = select_optimal_llm_config(gpu_memory_gb=4.0)

print(f"Selected: {config['model_name']}")
print(f"Expected VRAM: {config['expected_vram_gb']} GB")
```

### 2. Format Split Refinement Prompt

```python
from services.llm.prompts import format_split_prompt

before_pages = [
    {'page_num': 4, 'text': 'Page 4 content...'},
    {'page_num': 5, 'text': 'Page 5 content...'},
]

after_pages = [
    {'page_num': 6, 'text': 'Page 1 new document...'},
    {'page_num': 7, 'text': 'Page 2 continued...'},
]

signals = [
    "Page number reset: 5 -> 1",
    "Header changed",
    "Low semantic similarity (0.28)"
]

prompt = format_split_prompt(
    split_page=6,
    before_pages=before_pages,
    after_pages=after_pages,
    heuristic_signals=signals
)

# Use with LLM to get YES/NO decision
```

### 3. Format Document Naming Prompt

```python
from services.llm.prompts import format_naming_prompt, parse_filename

first_page = """
INVOICE
Acme Corporation
Date: 2024-06-15
Invoice #: INV-001
"""

prompt = format_naming_prompt(
    start_page=1,
    end_page=3,
    first_page_text=first_page,
    second_page_text="Line items..."
)

# Use with LLM to generate filename
# Expected output: "2024-06-15_Invoice_Acme Corporation"
```

### 4. Parse LLM Responses

```python
from services.llm.prompts import parse_split_decision, parse_filename

# Parse split decision
response = "YES - Page numbering resets and header changes"
should_split, reasoning = parse_split_decision(response)
# -> (True, "Page numbering resets and header changes")

# Parse filename
response = "2024-06-15_Invoice_Acme Corp Services"
filename = parse_filename(response)
# -> "2024-06-15_Invoice_Acme Corp Services"
```

## Generation Parameters

### Split Refinement
- Temperature: 0.1 (very low for consistent YES/NO)
- Max Tokens: 150
- Task: Determine if split point is valid

### Document Naming
- Temperature: 0.3 (low but allows variation)
- Max Tokens: 100
- Task: Generate structured filename

## Performance Estimates

Based on configuration and hardware:

| Configuration | Tokens/Second | Time (100 tokens) |
|---------------|---------------|-------------------|
| 8GB GPU (Q5) | 50 | 2.0s |
| 6GB GPU (Q4) | 40 | 2.5s |
| 4GB Hybrid | 25 | 4.0s |
| 2GB Hybrid | 25 | 4.0s |
| CPU only | 5 | 20.0s |

## Integration with Hardware Detection

The config module integrates with the existing OCR config for hardware detection:

```python
from services.ocr.config import detect_hardware_capabilities
from services.llm.config import select_optimal_llm_config

# Auto-detect hardware
capabilities = detect_hardware_capabilities()
# Returns: gpu_memory_gb, system_memory_gb, cpu_count, etc.

# Select optimal LLM config
llm_config = select_optimal_llm_config()
# Uses detected VRAM automatically
```

## Filename Format Specification

Filenames follow the pattern: `{DATE}_{DOCTYPE}_{DESCRIPTION}`

**DATE (YYYY-MM-DD):**
- Document date (not today's date)
- Use "UNDATED" if not found

**DOCTYPE (one word):**
- Invoice, Contract, Agreement, Letter, Report, Receipt, Form, Certificate, Statement, Memo, Proposal, Notice, Policy, Other

**DESCRIPTION (2-5 words):**
- Parties, subject matter, identifiers
- No special characters
- Concise but descriptive

**Examples:**
- `2024-06-15_Invoice_Acme Corp June Services`
- `2024-03-20_Contract_Software License Agreement`
- `UNDATED_Report_Annual Financial Summary`
- `2023-12-01_Letter_Employment Offer John Smith`

## Testing

Run the test suite:

```bash
cd python-backend
python test_llm_simple.py
```

This verifies:
- Configuration selection for all VRAM levels
- Prompt formatting with sample data
- Response parsing for decisions and filenames
- Filename validation

## Next Steps

The following files will be implemented in future tasks:
- `loader.py` - LLM loading with memory monitoring
- `split_analyzer.py` - Split refinement using LLM
- `name_generator.py` - Filename generation using LLM

These will use the configurations and prompts defined in this module.
