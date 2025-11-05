# Documentation Index

Welcome to the Document De-Bundler documentation. This index helps you quickly find the information you need.

## Quick Start

- **[Developer Quick Start](DEVELOPER_QUICK_START.md)** - Get up and running with development
- **[Architecture Overview](ARCHITECTURE.md)** - Understand the system design
- **[Implementation Status](IMPLEMENTATION_STATUS.md)** - Current feature completion status

## Features

Core features and their documentation:

- **[OCR System](features/LLM_INTEGRATION.md)** - OCR engine architecture and usage (Note: Consolidated with LLM docs)
- **[Document De-Bundling](features/DEBUNDLING_QUICK_START.md)** - ML-powered document separation
- **[Embedding Service](features/EMBEDDING_SERVICE_IMPLEMENTATION.md)** - Semantic analysis with Nomic Embed v1.5
- **[Split Detection](features/SPLIT_DETECTION_IMPLEMENTATION_REPORT.md)** - Algorithm for detecting document boundaries
- **[LLM Integration](features/LLM_INTEGRATION.md)** - Local LLM integration with llama.cpp

## Guides

Step-by-step instructions for setup and configuration:

- **[Tesseract Bundling](guides/TESSERACT_BUNDLING.md)** - Bundle Tesseract OCR for offline use
- **[PaddlePaddle 3.0 Upgrade](guides/PADDLEPADDLE_3.0_UPGRADE_AND_CUDA_FIX.md)** - GPU acceleration setup and troubleshooting

## Implementation Details

Technical implementation reports:

- **[OCR Event Handlers](implementations/OCR_EVENT_HANDLERS_IMPLEMENTATION.md)** - Svelte event-driven UI architecture
- **[Phase 3 Step 2](implementations/PHASE_3_STEP_2_IMPLEMENTATION_REPORT.md)** - Rust Tauri commands implementation
- **[Python Bridge](implementations/PYTHON_BRIDGE_IMPLEMENTATION.md)** - Rust-Python async bridge

## Testing

- **[Latest Test Results](testing/TEST_RESULTS_2025-11-01.md)** - Current test suite status (93% pass rate)

## Archive

Historical documentation preserved for reference:

- **[Archive Index](archive/README.md)** - Organization of archived documentation
  - Session handoffs
  - Implementation checklists
  - Fix summaries and reports
  - Historical test results

## Documentation Standards

For guidelines on creating and organizing documentation, see the project memory: `documentation_structure_standards`

---

## Common Tasks

### I want to...

- **Get started developing** → [Developer Quick Start](DEVELOPER_QUICK_START.md)
- **Understand the architecture** → [Architecture Overview](ARCHITECTURE.md)
- **Check implementation status** → [Implementation Status](IMPLEMENTATION_STATUS.md)
- **Use OCR features** → [LLM Integration](features/LLM_INTEGRATION.md) (includes OCR)
- **Implement de-bundling** → [De-Bundling Quick Start](features/DEBUNDLING_QUICK_START.md)
- **Set up GPU acceleration** → [PaddlePaddle 3.0 Guide](guides/PADDLEPADDLE_3.0_UPGRADE_AND_CUDA_FIX.md)
- **Troubleshoot Tesseract** → [Tesseract Bundling](guides/TESSERACT_BUNDLING.md)
- **Review test results** → [Latest Test Results](testing/TEST_RESULTS_2025-11-01.md)
- **Find historical context** → [Archive](archive/)

## Contributing to Documentation

When adding new documentation:

1. Choose the correct folder based on document type (see structure above)
2. Follow the naming conventions: `FEATURE_NAME.md` for features, `TASK_NAME.md` for guides
3. Use cross-references to link related docs
4. Update this README.md with a link to your new document
5. Archive superseded documents rather than deleting them

See project memory `documentation_structure_standards` for detailed guidelines.

---

**Last Updated:** 2025-11-03
**Documentation Version:** 2.0 (Reorganized structure)
