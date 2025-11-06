# Language Pack Download Feature - Research Summary

**Date**: 2025-11-06
**Status**: Research Complete, Ready for Implementation

---

## Executive Summary

This document summarizes research into PaddleOCR's model download mechanisms to enable proactive language pack downloads with progress reporting in the Document De-Bundler application.

### Current State
- **UI**: Language selection dropdown with 20 languages exists
- **Backend**: No download management - relies on PaddleOCR's invisible auto-download
- **UX Issue**: Users experience 30-60 second "hangs" when first using non-English languages
- **Problem**: No progress feedback, no user control over downloads

### Target State
- Proactive language pack downloads before OCR processing
- Real-time progress reporting (download %, extraction status)
- Clear UI showing installed vs. available languages
- Ability to pre-download multiple languages
- Graceful error handling with retry capability

---

## PaddleOCR Model Architecture

### Model Types

PaddleOCR uses three types of models:

1. **Detection Model** (`det`): Locates text regions in images
   - Often shared across multiple languages (multilingual model)
   - Size: ~1-4 MB
   - Example: `ml_PP-OCRv3_det` (multilingual detection)

2. **Recognition Model** (`rec`): Recognizes text characters
   - Language-specific
   - Size: ~3-12 MB depending on character set complexity
   - Example: `en_PP-OCRv4_rec` (English recognition)

3. **Classification Model** (`cls`): Detects text angle/orientation
   - Shared across all languages
   - Size: ~1-2 MB
   - Optional but recommended

### Model Versions
- **PP-OCRv3**: Previous generation (2.x compatibility)
- **PP-OCRv4**: Latest English models
- **PP-OCRv5**: Newest version (3.x), server models by default

---

## Model Storage & Distribution

### Storage Locations

#### Bundled Models (Application)
```
python-backend/models/
├── det/                    # Detection model (shared)
│   ├── inference.pdmodel
│   ├── inference.pdiparams
│   └── inference.pdiparams.info
├── rec/                    # Recognition model (language-specific)
│   ├── inference.pdmodel
│   ├── inference.pdiparams
│   └── inference.pdiparams.info
└── cls/                    # Classification model (optional)
    ├── inference.pdmodel
    ├── inference.pdiparams
    └── inference.pdiparams.info
```

**For multi-language support**, use subdirectories:
```
python-backend/models/
├── det/                    # English detection (or use ml_det for multilingual)
├── rec/                    # English recognition
├── fr_det/                 # French detection (optional, can share ml_det)
├── fr_rec/                 # French recognition (required)
├── de_det/                 # German detection (optional)
├── de_rec/                 # German recognition (required)
└── ...
```

#### PaddleOCR Cache (Auto-Download Location)
- **Windows**: `%LOCALAPPDATA%\PaddleOCR\`
- **macOS**: `~/Library/Caches/PaddleOCR/`
- **Linux**: `~/.cache/PaddleOCR/`

PaddleOCR automatically downloads to this location on first use if bundled models not found.

### Download Sources

#### Primary (as of 2025)
**HuggingFace**: Default in PaddleOCR 3.0.2+
- Environment variable: `PADDLE_PDX_MODEL_SOURCE=HuggingFace`
- More reliable for international users

#### Secondary (Legacy)
**Baidu Cloud Storage (BOS)**: `https://paddleocr.bj.bcebos.com/`
- Environment variable: `PADDLE_PDX_MODEL_SOURCE=BOS`
- Faster in China
- URL pattern: `https://paddleocr.bj.bcebos.com/{version}/{language}/{model_name}.tar`

#### Examples
```
# English PP-OCRv3 Detection
https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar

# English PP-OCRv4 Recognition
https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar

# Japanese PP-OCRv3 Recognition
https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar

# Multilingual Detection (shared)
https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ml_PP-OCRv3_det_infer.tar

# Classification Model
https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
```

---

## PaddleOCR Download API

### Built-in Download Functions

PaddleOCR provides utilities in `paddleocr.ppocr.utils.network`:

#### 1. `download_with_progressbar`
```python
from paddleocr.ppocr.utils.network import download_with_progressbar

# Download with progress reporting
download_with_progressbar(url, save_path)
```

**Features:**
- Uses `requests` library with streaming
- Shows progress via tqdm or custom callback
- Supports proxies
- Handles network errors

#### 2. `maybe_download`
```python
from paddleocr.ppocr.utils.network import maybe_download

# Download only if not exists
maybe_download(url, save_path)
```

**Features:**
- Checks if file exists before downloading
- Skips download if file present
- Useful for preventing re-downloads

#### 3. `run_model_download`
```python
# Internal PaddleOCR function
def run_model_download(model_url: str, model_file_path: Path) -> None:
    # Downloads .tar archive
    # Extracts to model directory
    # Validates model files
```

### Model File Format

Models are distributed as `.tar` archives containing:
- `inference.pdmodel` - Model structure (serialized graph)
- `inference.pdiparams` - Model weights (parameters)
- `inference.pdiparams.info` - Parameter metadata
- `inference.yml` or `inference.json` - Configuration (PaddleOCR 3.x)

---

## Supported Languages

### Language List (20 Languages in UI)

| Language | Code | Detection Model | Recognition Model | Approx Size |
|----------|------|-----------------|-------------------|-------------|
| English | `en` | `ml_det` or `en_det` | `en_rec` | 3-9 MB |
| Chinese (Simplified) | `ch` | `ml_det` or `ch_det` | `ch_rec` | 8-12 MB |
| Chinese (Traditional) | `chinese_cht` | `ml_det` | `chinese_cht_rec` | 12 MB |
| French | `fr` | `ml_det` | `fr_rec` | 9-11 MB |
| German | `german` | `ml_det` | `german_rec` | 9-11 MB |
| Spanish | `es` | `ml_det` | `es_rec` | 9-11 MB |
| Portuguese | `pt` | `ml_det` | `pt_rec` | 9-11 MB |
| Russian | `ru` | `ml_det` (Cyrillic) | `cyrillic_rec` | 9-10 MB |
| Arabic | `ar` | `ml_det` | `arabic_rec` | 9-10 MB |
| Hindi | `hi` | `ml_det` (Devanagari) | `devanagari_rec` | 9-10 MB |
| Japanese | `japan` | `ml_det` | `japan_rec` | 11 MB |
| Korean | `korean` | `ml_det` | `korean_rec` | 11 MB |
| Italian | `it` | `ml_det` | `latin_rec` | 9-10 MB |
| Dutch | `nl` | `ml_det` | `latin_rec` | 9-10 MB |
| Vietnamese | `vi` | `ml_det` | `latin_rec` | 9-10 MB |
| Thai | `th` | `ml_det` | `th_rec` | 9-10 MB |
| Turkish | `tr` | `ml_det` | `latin_rec` | 9-10 MB |
| Polish | `pl` | `ml_det` | `latin_rec` | 9-10 MB |
| Swedish | `sv` | `ml_det` | `latin_rec` | 9-10 MB |
| Danish | `da` | `ml_det` | `latin_rec` | 9-10 MB |

**Total storage for all 20 languages**: ~150-200 MB

### Detection Model Sharing

Most languages can share the **multilingual detection model** (`ml_PP-OCRv3_det`):
- Detects text regions regardless of language
- Only download once, use for all languages
- Exception: Some languages may have optimized language-specific detection models

### Recognition Model Requirements

Each language **requires its own recognition model**:
- Character set varies by language
- Cannot be shared between languages
- Must be downloaded separately for each language

---

## Implementation Strategy

### Download Workflow

```
User selects language → Check if installed → If not:
  ├─→ Download detection model (if not shared/present)
  ├─→ Download recognition model (language-specific)
  ├─→ Download classification model (if enabled & not present)
  └─→ Extract .tar archives
      └─→ Validate model files (check .pdmodel and .pdiparams exist)
          └─→ Report success
```

### Progress Reporting

**Download Phases**:
1. **Checking**: Verify what's already installed
2. **Downloading**: Fetch .tar archive with progress (bytes/total)
3. **Extracting**: Unpack .tar file (may take 10-30s)
4. **Validating**: Ensure all required files present
5. **Complete**: Language ready for use

**Progress Events** (JSON over IPC):
```json
{
  "type": "language_download_progress",
  "data": {
    "language": "fr",
    "language_name": "French",
    "phase": "downloading",  // checking | downloading | extracting | validating | complete | error
    "model_type": "rec",     // det | rec | cls
    "progress_percent": 45,  // 0-100
    "bytes_downloaded": 4500000,
    "bytes_total": 10000000,
    "message": "Downloading French recognition model...",
    "speed_mbps": 2.5        // Optional: download speed
  }
}
```

### Storage Structure Decision

**Option A: Language-Prefixed Directories** (Recommended)
```
models/
├── ml_det/              # Multilingual detection (shared)
├── en_rec/              # English recognition
├── fr_rec/              # French recognition
├── de_rec/              # German recognition
├── cls/                 # Classification (shared)
└── ...
```

**Pros**:
- Clear separation per language
- Easy to identify installed languages
- Matches PaddleOCR's internal structure
- Easy to delete language packs

**Option B: Nested Language Directories**
```
models/
├── en/
│   ├── det/
│   └── rec/
├── fr/
│   ├── det/
│   └── rec/
└── ...
```

**Pros**:
- Logical grouping by language
- Easier to manage per-language installations

**Cons**:
- Detection models would be duplicated if language-specific
- More complex to share multilingual detection model

**Recommendation**: Use Option A (language-prefixed) as it aligns with PaddleOCR conventions.

---

## Technical Considerations

### Concurrency
- **Single download at a time**: Avoid overwhelming network/disk
- **Queue system**: Allow users to queue multiple languages
- **Background downloads**: Don't block UI during downloads

### Error Handling
- **Network errors**: Retry with exponential backoff
- **Disk space**: Check available space before download
- **Corrupted downloads**: Validate checksums if available
- **Partial downloads**: Support resume if possible

### Caching
- **Check cache first**: Look in both bundled and PaddleOCR cache
- **Reuse existing**: Copy from PaddleOCR cache if present
- **Symlinks**: Consider using symlinks to avoid duplication

### Memory Management
- **Stream downloads**: Don't load entire file into memory
- **Chunked extraction**: Extract .tar in chunks
- **Cleanup**: Remove .tar after successful extraction

---

## File References

### Existing Code
- `python-backend/download_models.py:14-34` - Current download script (English only)
- `python-backend/services/ocr/engines/paddleocr_engine.py:60-129` - Model initialization
- `python-backend/models/README.md:182-186` - Multi-language documentation
- `src/lib/components/AdvancedOCRSettings.svelte:34-55` - Language list in UI

### To Be Created
- `python-backend/services/language_pack_manager.py` - New service
- `src-tauri/src/commands/language_packs.rs` - New Tauri commands (or add to existing commands.rs)
- `src/lib/components/LanguagePackManager.svelte` - New UI component
- `src/lib/stores/languagePackStore.ts` - New state management

---

## Next Steps

1. ✅ **Research Complete** (this document)
2. ⏭️ **Create Language Pack Metadata** - Define language configs with URLs
3. ⏭️ **Implement language_pack_manager.py** - Core download service
4. ⏭️ **Integrate with IPC** - Add commands to main.py
5. ⏭️ **Create Tauri Commands** - Rust layer for language pack management
6. ⏭️ **Build UI Components** - LanguagePackManager.svelte
7. ⏭️ **Testing** - Unit, integration, and manual testing
8. ⏭️ **Documentation** - Update user and developer docs

---

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [Multi-Language Models](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/multi_languages_en.md)
- [Models List](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md)
- [PaddleOCR PyPI](https://pypi.org/project/paddleocr/)

---

**Document Status**: Complete and ready for handoff to implementation phase.
