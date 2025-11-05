# HANDOFF DOCUMENT: llama.cpp Integration for Document De-Bundler

**Date**: 2025-11-01
**Status**: Phase 3 (Integration) - 60% Complete
**Next Session Priority**: Complete Phase 3 & 4, then test

---

## Executive Summary

Successfully implemented llama.cpp integration into Document De-Bundler for AI-powered document naming and split refinement. **Phases 1 & 2 are 100% complete** with all core LLM services implemented. **Phase 3 is 60% complete** with split detection and naming service updated. Remaining work: update main.py pipeline, implement sequential GPU processing, frontend integration, testing, and documentation.

---

## ✅ COMPLETED WORK (Phases 1-2, Partial Phase 3)

### Phase 1: Dependencies & Model Management ✅ (100%)

**Files Created:**
1. ✅ `python-backend/requirements.txt` - Added `llama-cpp-python==0.3.4`
2. ✅ `python-backend/download_llm_models.py` - Interactive model downloader
   - Downloads Phi-3 Mini Q4_K_M (~2.3GB) and Gemma 2 2B Q4_K_M (~1.5GB)
   - HuggingFace integration with progress tracking
   - Resume support for interrupted downloads

3. ✅ `python-backend/bin/llama-cpp/README.md` - Binary bundling guide
   - Instructions for downloading pre-built binaries
   - Platform-specific paths (Windows/Linux/macOS)

4. ✅ `python-backend/services/resource_path.py` - Updated with LLM helpers
   - `get_llm_models_dir()` - Get models/llm directory
   - `get_phi3_mini_path()` - Get Phi-3 Mini model path
   - `get_gemma2_2b_path()` - Get Gemma 2 2B model path
   - `verify_llm_models()` - Check which models are installed
   - `get_llama_cpp_binary_path()` - Get platform-specific binary
   - `verify_llama_cpp_binary()` - Verify binary availability

5. ✅ `python-backend/models/llm/README.md` - Comprehensive model documentation
   - Model selection guide
   - Installation instructions (3 methods)
   - Performance benchmarks
   - Memory requirements
   - Troubleshooting guide

### Phase 2: Core LLM Services ✅ (100%)

**Files Created:**
1. ✅ `python-backend/services/llm/loader.py` - Model loading with dual integration
   - **Primary**: llama-cpp-python bindings
   - **Fallback**: Standalone binary via subprocess
   - GPU layer optimization (28 GPU + 4 CPU for 4GB VRAM)
   - Memory monitoring and cleanup
   - Test function included

2. ✅ `python-backend/services/llm/manager.py` - Singleton lifecycle manager
   - Lazy loading (initialize only when needed)
   - Thread-safe generation with queuing
   - Batch processing support
   - Statistics tracking
   - Auto cleanup and memory management
   - Test function included

3. ✅ `python-backend/services/llm/split_analyzer.py` - LLM-based split refinement
   - Analyzes DBSCAN candidates using LLM
   - Provides YES/NO decisions with reasoning
   - Confidence scoring
   - Caching support for LLM decisions
   - Batch processing with progress callbacks
   - Test function included

4. ✅ `python-backend/services/llm/name_generator.py` - Intelligent document naming
   - Generates `{DATE}_{DOCTYPE}_{DESCRIPTION}` format
   - Extracts dates, document types, and descriptions
   - Validation and auto-fixing of malformed names
   - Fallback to heuristic naming if LLM fails
   - Caching support
   - Test function included

5. ✅ `python-backend/services/llm/settings.py` - User configuration management
   - `LLMSettings` dataclass with all configurable options
   - `LLMSettingsManager` for persistence (JSON file)
   - Feature toggles (enable/disable LLM, split refinement, naming)
   - Performance tuning (GPU layers, context size, batch size)
   - Quality thresholds (confidence, fallback behavior)
   - Memory management settings
   - Validation and effective config resolution
   - Test function included

**Existing Files (Already Implemented):**
- ✅ `python-backend/services/llm/config.py` - VRAM-based model selection (already optimal)
- ✅ `python-backend/services/llm/prompts.py` - Prompt templates (already optimal)

### Phase 3: Integration with Existing Services 🔄 (60%)

**Files Updated:**
1. ✅ `python-backend/services/split_detection.py` - Added LLM refinement
   - Added `use_llm_refinement` parameter to `detect_splits_for_document()`
   - Integrates `SplitAnalyzer` to refine DBSCAN candidates
   - Filters by LLM decision and confidence threshold
   - Graceful fallback if LLM unavailable
   - Progress callback integration

2. ✅ `python-backend/services/naming_service.py` - Added LLM naming
   - Added `use_llm` parameter to `suggest_name()`
   - Integrates `NameGenerator` for intelligent naming
   - Checks settings for LLM availability
   - Fallback to heuristic naming preserved
   - Added `_heuristic_name()` helper method

3. ⏳ `python-backend/main.py` - **NEEDS UPDATING**
   - **Status**: NOT YET UPDATED
   - **Required Changes**: See "Next Steps" section below

---

## 🔴 REMAINING WORK

### Phase 3.3: Update main.py Processing Pipeline ⏳ (PRIORITY 1)

**File**: `python-backend/main.py`

**What Needs to be Done:**

1. **Update `handle_process()` function** to add LLM phases:

```python
# Current pipeline in main.py:
# Phase 1: OCR/Text Extraction ✅
# Phase 2: Embedding Generation ✅
# Phase 3: Split Detection ✅
# Phase 4: Document Extraction ✅
# Phase 5: Bundling ✅

# NEW: Add these integrations:
```

**Required Code Changes:**

**A. Add LLM option to command structure (around line 50)**:
```python
def handle_process(self, command: Dict[str, Any]):
    options = command.get("options", {})

    # Existing options
    force_ocr = options.get("force_ocr", False)
    skip_splitting = options.get("skip_splitting", False)

    # NEW: Add LLM options
    use_llm_refinement = options.get("use_llm_refinement", True)  # Default: True
    use_llm_naming = options.get("use_llm_naming", True)  # Default: True
```

**B. Update Phase 3 (Split Detection) - around line 200**:
```python
# ===== PHASE 3: Split Detection =====
if not skip_splitting:
    cache.log_phase(doc_id, "split_detection", "started", "Detecting document boundaries")
    self.send_progress(50, 100, "Phase 3: Detecting document boundaries...")

    def split_progress(current, total, message):
        percent = 50 + int((current / total) * 15)  # 50-65%
        self.send_progress(percent, 100, f"Phase 3: {message}")

    try:
        # UPDATED: Add use_llm_refinement parameter
        split_count = detect_splits_for_document(
            doc_id,
            use_llm_refinement=use_llm_refinement,  # NEW
            progress_callback=split_progress
        )
        cache.log_phase(doc_id, "split_detection", "completed", f"Detected {split_count} split points")
        cache.update_document_status(doc_id, "splits_detected")
        logger.info(f"Detected {split_count} split candidates")
    except Exception as e:
        logger.error(f"Split detection error: {e}", exc_info=True)
        cache.log_phase(doc_id, "split_detection", "failed", str(e))
        raise
```

**C. Update Phase 4 (Document Naming) - around line 250**:
```python
# ===== PHASE 4: Document Extraction and Naming =====
cache.log_phase(doc_id, "extraction", "started", "Extracting document segments")
self.send_progress(65, 100, "Phase 4: Extracting documents...")

# ... existing code to get split_points ...

# Extract document segments
extracted_files = []
naming_service = NamingService()

with PDFProcessor(file_path) as pdf:
    num_segments = len(split_points) - 1

    for idx in range(num_segments):
        start_page = split_points[idx]
        end_page = split_points[idx + 1] - 1  # Inclusive end

        percent = 65 + int((idx / num_segments) * 15)  # 65-80%
        self.send_progress(percent, 100, f"Extracting document {idx+1}/{num_segments}...")

        # Get text for naming
        first_page_text = cache.get_page_text(doc_id, start_page) or ""
        second_page_text = cache.get_page_text(doc_id, start_page + 1) if start_page + 1 <= end_page else None

        # UPDATED: Add LLM naming with second page context
        suggested_name = naming_service.suggest_name(
            text_content=first_page_text,
            page_num=idx + 1,
            fallback_prefix=file_path_obj.stem,
            second_page_text=second_page_text,  # NEW
            use_llm=use_llm_naming  # NEW
        )

        # ... rest of extraction code ...
```

**D. Add cleanup phase for LLM (around line 300, before final result)**:
```python
# ===== Phase 5.5: LLM Cleanup (if used) =====
if use_llm_refinement or use_llm_naming:
    try:
        from services.llm.manager import cleanup_llm
        from services.llm.settings import get_settings

        settings = get_settings()
        if settings.auto_cleanup_enabled:
            self.send_progress(95, 100, "Cleaning up LLM resources...")
            cleanup_llm()
            logger.info("LLM cleanup complete")
    except Exception as e:
        logger.warning(f"LLM cleanup error: {e}")
```

### Phase 4: Resource Management & Optimization ⏳ (PRIORITY 2)

**4.1 Sequential GPU Processing** - Already handled by existing code structure!
- OCR Phase → cleanup → Embeddings Phase → cleanup → LLM Phase
- Just need to ensure cleanup calls are in place (see Phase 3.3 above)

**4.2 Update cache_manager.py for LLM caching** - **OPTIONAL**
- Current implementation: `split_analyzer.py` and `name_generator.py` already have internal caching via MD5 hashes
- Cache manager update would be for centralized cache management (nice-to-have, not critical)

**If you want to implement it:**
```python
# Add to cache_manager.py

def save_llm_split_decision(self, doc_id: str, split_page: int, decision: bool, confidence: float, reasoning: str):
    """Save LLM split decision."""
    with self.get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO llm_split_decisions
            (doc_id, split_page, decision, confidence, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (doc_id, split_page, decision, confidence, reasoning))

def get_llm_split_decision(self, doc_id: str, split_page: int):
    """Get cached LLM split decision."""
    # Implementation...

def save_llm_name(self, doc_id: str, start_page: int, end_page: int, filename: str):
    """Save LLM-generated name."""
    # Implementation...
```

### Phase 5: Frontend Integration ⏳ (PRIORITY 3)

**5.1 Create LLMStatus.svelte component**

**File**: `src/lib/components/LLMStatus.svelte` (NEW FILE)

```svelte
<script lang="ts">
  import { invoke } from '@tauri-apps/api/tauri';
  import { onMount } from 'svelte';

  let llmAvailable = false;
  let llmInitialized = false;
  let modelName = '';
  let gpuEnabled = false;
  let vramUsage = 0;

  async function checkLLMStatus() {
    try {
      // Call Rust command to check LLM status via Python
      const status = await invoke('get_llm_status');
      llmAvailable = status.available;
      llmInitialized = status.initialized;
      modelName = status.model_name || 'Not loaded';
      gpuEnabled = status.gpu_enabled;
      vramUsage = status.expected_vram_gb || 0;
    } catch (e) {
      console.error('Failed to get LLM status:', e);
    }
  }

  onMount(() => {
    checkLLMStatus();
  });
</script>

<div class="llm-status-card">
  <h3>AI Features</h3>

  <div class="status-row">
    <span class="label">Status:</span>
    <span class="value {llmAvailable ? 'available' : 'unavailable'}">
      {llmAvailable ? '✓ Available' : '✗ Unavailable'}
    </span>
  </div>

  {#if llmAvailable}
    <div class="status-row">
      <span class="label">Model:</span>
      <span class="value">{modelName}</span>
    </div>

    <div class="status-row">
      <span class="label">GPU:</span>
      <span class="value">{gpuEnabled ? `✓ Enabled (${vramUsage.toFixed(1)}GB VRAM)` : 'CPU Only'}</span>
    </div>
  {/if}
</div>

<style>
  .llm-status-card {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    margin: 0.5rem 0;
  }

  .available {
    color: green;
  }

  .unavailable {
    color: red;
  }
</style>
```

**5.2 Update ProcessingOptions.svelte**

**File**: `src/lib/components/ProcessingOptions.svelte` (or wherever processing options are)

Add these checkboxes:
```svelte
<script lang="ts">
  export let useLLMRefinement = true;
  export let useLLMNaming = true;
</script>

<!-- Add to existing options -->
<div class="option-group">
  <h4>AI Features</h4>

  <label>
    <input type="checkbox" bind:checked={useLLMRefinement} />
    Use AI to refine split points
    <span class="help-text">LLM analyzes content to validate splits (~20s overhead)</span>
  </label>

  <label>
    <input type="checkbox" bind:checked={useLLMNaming} />
    Use AI for document naming
    <span class="help-text">LLM generates intelligent filenames (~5s per document)</span>
  </label>
</div>
```

**5.3 Pass options to backend**

Update wherever you call `invoke("start_processing", ...)`:
```typescript
await invoke("start_processing", {
  filePath: selectedFile,
  options: {
    force_ocr: forceOCR,
    skip_splitting: skipSplitting,
    use_llm_refinement: useLLMRefinement,  // NEW
    use_llm_naming: useLLMNaming,  // NEW
    output_format: outputFormat,
    // ... other options
  }
});
```

### Phase 6: Testing & Documentation ⏳ (PRIORITY 4)

**6.1 Create test_llm_integration.py**

**File**: `python-backend/test_llm_integration.py` (NEW FILE)

```python
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
```

**6.2 Create LLM_INTEGRATION.md documentation** - See template in next section

**6.3 Update CLAUDE.md and README.md** - Add LLM features section

---

## 📝 DOCUMENTATION TEMPLATE

**File**: `docs/LLM_INTEGRATION.md` (CREATE THIS)

Use this structure:
```markdown
# LLM Integration for Document De-Bundler

## Overview
[Explain llama.cpp integration, model selection, use cases]

## Installation

### Prerequisites
[System requirements, dependencies]

### Model Download
[Instructions for download_llm_models.py]

### First-Time Setup
[Step-by-step setup]

## Usage

### Basic Usage
[Code examples for split refinement and naming]

### Advanced Configuration
[Settings customization, performance tuning]

## Architecture

### Components
[Explain loader, manager, split_analyzer, name_generator]

### Data Flow
[Diagram or description of processing pipeline]

## Performance

### Benchmarks
[Performance metrics on 4GB VRAM]

### Memory Management
[Sequential GPU processing explanation]

## Troubleshooting
[Common issues and solutions]

## API Reference
[Key functions and classes]
```

---

## 🧪 TESTING INSTRUCTIONS

### Before Testing
1. Install llama-cpp-python:
   ```bash
   cd python-backend
   .venv\Scripts\activate
   pip install llama-cpp-python==0.3.4
   ```

2. Download models:
   ```bash
   python download_llm_models.py
   # Select option 1 (Phi-3 Mini) - ~2.3GB download
   ```

### Run Tests
```bash
# Unit tests for individual components
python python-backend/services/llm/loader.py
python python-backend/services/llm/manager.py
python python-backend/services/llm/split_analyzer.py
python python-backend/services/llm/name_generator.py
python python-backend/services/llm/settings.py

# Integration test (once created)
python python-backend/test_llm_integration.py

# Full pipeline test
npm run tauri:dev
# Then use UI to process a test PDF
```

### What to Verify
- ✅ Models download successfully
- ✅ Loader initializes with GPU (check logs for "GPU: enabled")
- ✅ Manager generates coherent text
- ✅ Split analyzer makes reasonable decisions
- ✅ Name generator produces valid `{DATE}_{DOCTYPE}_{DESCRIPTION}` format
- ✅ Settings persist across sessions
- ✅ Main pipeline processes PDFs with LLM features

---

## 🎯 IMMEDIATE NEXT STEPS (Priority Order)

1. **Update main.py** (Phase 3.3) - ~30 minutes
   - Add `use_llm_refinement` and `use_llm_naming` parameters
   - Update Phase 3 (split detection) call
   - Update Phase 4 (naming) call
   - Add LLM cleanup phase

2. **Test locally** - ~30 minutes
   - Install llama-cpp-python
   - Download Phi-3 Mini model
   - Run test_llm_integration.py (create it first)
   - Process test PDF through full pipeline

3. **Frontend integration** (Phase 5) - ~1 hour
   - Create LLMStatus.svelte
   - Update ProcessingOptions.svelte
   - Add Rust command for LLM status (if needed)

4. **Documentation** (Phase 6) - ~1 hour
   - Create LLM_INTEGRATION.md
   - Update CLAUDE.md with LLM section
   - Update README.md

5. **Final testing** - ~30 minutes
   - Test with various PDF types
   - Verify GPU usage
   - Check memory cleanup
   - Validate generated names

**Total Remaining Effort**: ~4-5 hours

---

## 📊 CURRENT PROJECT STATUS

```
Phase 1: Dependencies & Model Management     ████████████████████ 100%
Phase 2: Core LLM Services                   ████████████████████ 100%
Phase 3: Integration with Existing Services  ████████████░░░░░░░░  60%
Phase 4: Resource Management & Optimization  ░░░░░░░░░░░░░░░░░░░░   0%
Phase 5: Frontend Integration                ░░░░░░░░░░░░░░░░░░░░   0%
Phase 6: Testing & Documentation             ░░░░░░░░░░░░░░░░░░░░   0%

Overall Progress: ████████████░░░░░░░░ 60%
```

---

## 🔑 KEY DESIGN DECISIONS

1. **Dual Integration**: llama-cpp-python (primary) + binary fallback (reliability)
2. **Sequential GPU Processing**: OCR → Embeddings → LLM (prevents VRAM conflicts)
3. **Lazy Loading**: LLM only initializes when needed (saves memory)
4. **Smart Fallbacks**: Heuristics if LLM unavailable (graceful degradation)
5. **Caching**: MD5-based caching for LLM decisions and names (performance)
6. **Settings-Driven**: All features configurable via settings.py (flexibility)

---

## 📁 FILE LOCATIONS REFERENCE

```
python-backend/
├── requirements.txt (UPDATED)
├── download_llm_models.py (NEW)
├── test_llm_integration.py (TO CREATE)
├── models/
│   └── llm/
│       ├── README.md (NEW)
│       └── (model files downloaded here)
├── bin/
│   └── llama-cpp/
│       └── README.md (NEW)
├── services/
│   ├── resource_path.py (UPDATED)
│   ├── split_detection.py (UPDATED)
│   ├── naming_service.py (UPDATED)
│   ├── main.py (TO UPDATE - PRIORITY)
│   └── llm/
│       ├── config.py (EXISTING - NO CHANGES)
│       ├── prompts.py (EXISTING - NO CHANGES)
│       ├── loader.py (NEW)
│       ├── manager.py (NEW)
│       ├── split_analyzer.py (NEW)
│       ├── name_generator.py (NEW)
│       └── settings.py (NEW)

src/lib/components/
├── LLMStatus.svelte (TO CREATE)
└── ProcessingOptions.svelte (TO UPDATE)

docs/
├── LLM_INTEGRATION.md (TO CREATE)
└── handoff/
    └── HANDOFF_LLAMA_CPP_INTEGRATION.md (THIS FILE)
```

---

## 💡 TROUBLESHOOTING TIPS

**If models won't download:**
- Check internet connection
- Try manual download from HuggingFace
- Verify `huggingface-hub` is installed

**If llama-cpp-python won't install:**
- May need Visual Studio Build Tools on Windows
- Try pre-built wheels: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121`

**If GPU not being used:**
- Check CUDA installation
- Verify VRAM >= 2GB
- Check logs for "GPU: enabled"

**If generation is slow:**
- Verify GPU is being used
- Check n_gpu_layers in config
- Try CPU mode to confirm it's faster

---

## 🎉 SUCCESS CRITERIA

You'll know it's working when:
1. ✅ Models download successfully (~2.3GB for Phi-3 Mini)
2. ✅ Logs show "GPU: enabled" and "✓ Model loaded"
3. ✅ Split analyzer returns YES/NO decisions with reasoning
4. ✅ Name generator produces `2024-01-15_Invoice_Acme Corp` format
5. ✅ Main pipeline processes PDFs with LLM phases in logs
6. ✅ Generated document names are intelligent and descriptive
7. ✅ Split points are refined with higher accuracy
8. ✅ Memory stays under 2.5GB VRAM peak
9. ✅ Processing time: +1-2 minutes overhead for LLM features

---

## 📋 IMPLEMENTATION CHECKLIST

Copy this to track progress:

### Phase 3.3: Main Pipeline Integration
- [ ] Add `use_llm_refinement` parameter extraction
- [ ] Add `use_llm_naming` parameter extraction
- [ ] Update Phase 3 split detection call
- [ ] Update Phase 4 naming service call (add second_page_text)
- [ ] Add Phase 5.5 LLM cleanup
- [ ] Test main.py changes

### Phase 4: Resource Management
- [ ] Verify sequential GPU processing (already implemented)
- [ ] Test memory cleanup between phases
- [ ] (Optional) Centralized LLM caching in cache_manager.py

### Phase 5: Frontend Integration
- [ ] Create LLMStatus.svelte component
- [ ] Update ProcessingOptions.svelte with LLM checkboxes
- [ ] Pass LLM options to backend invoke call
- [ ] (Optional) Add Rust command for LLM status query
- [ ] Test frontend UI

### Phase 6: Testing & Documentation
- [ ] Create test_llm_integration.py
- [ ] Run all unit tests (loader, manager, split_analyzer, name_generator, settings)
- [ ] Run integration test
- [ ] Test full pipeline with sample PDF
- [ ] Create LLM_INTEGRATION.md documentation
- [ ] Update CLAUDE.md with LLM section
- [ ] Update README.md with LLM features
- [ ] Verify all success criteria met

---

**Good luck with the continuation! The hard part is done - just integration and testing remaining.** 🚀
