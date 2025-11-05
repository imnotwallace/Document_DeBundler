# Embedding Models Directory

This directory contains bundled Nomic Embed v1.5 models for offline operation.

## Auto-Download (Default Behavior)

By default, the embedding service will automatically download models on first run:
- **Text Model**: ~550MB (for document analysis and splitting)
- **Vision Model**: ~600MB (for image/visual understanding)

Models are downloaded to the user's cache directory:
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`
- **macOS**: `~/Library/Caches/huggingface/hub/`
- **Linux**: `~/.cache/huggingface/hub/`

## Pre-Bundling Models (Recommended for Offline Use)

For offline installations or faster first-run experience, you can pre-bundle models locally.

### Using the Automated Download Script (Easiest)

Run the provided download script:

```bash
# From python-backend directory
python download_embedding_models.py
```

This will:
1. Check for existing models
2. Download both text and vision models from HuggingFace
3. Save them to the correct directories
4. Verify installation

**Expected Directory Structure After Download:**
```
embeddings/
├── text/                           # Text embedding model (~550MB)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── ...
└── vision/                         # Vision embedding model (~600MB)
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    └── ...
```

### Manual Download (Advanced)

If you need to download models manually (e.g., on a different machine):

**1. Install HuggingFace Hub CLI:**
```bash
pip install huggingface-hub
```

**2. Download Text Model:**
```bash
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 \
    --local-dir python-backend/models/embeddings/text \
    --local-dir-use-symlinks False
```

**3. Download Vision Model:**
```bash
huggingface-cli download nomic-ai/nomic-embed-vision-v1.5 \
    --local-dir python-backend/models/embeddings/vision \
    --local-dir-use-symlinks False
```

**Windows PowerShell:**
```powershell
# Text model
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 --local-dir "python-backend\models\embeddings\text" --local-dir-use-symlinks False

# Vision model
huggingface-cli download nomic-ai/nomic-embed-vision-v1.5 --local-dir "python-backend\models\embeddings\vision" --local-dir-use-symlinks False
```

### Verifying Installation

After downloading, verify the directory structure:

```bash
# Unix/macOS/Linux
ls -R python-backend/models/embeddings/

# Windows
tree /F python-backend\models\embeddings
```

Each model directory should contain at least:
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Model weights
- `tokenizer_config.json` - Tokenizer configuration (text model)
- `preprocessor_config.json` - Preprocessor config (vision model)

You can verify with Python:
```python
from services.resource_path import verify_embedding_models

models = verify_embedding_models()
print(f"Text model available: {models['text']}")
print(f"Vision model available: {models['vision']}")
```

## Model Information

### Text Embedding Model
- **Model**: nomic-ai/nomic-embed-text-v1.5
- **Size**: ~550MB
- **Dimensions**: 768
- **Use Case**: Document analysis, semantic search, text similarity
- **Features**:
  - Context length: 8192 tokens
  - Matryoshka Representation Learning (flexible dimensions)
  - Multimodal (aligned with vision model)

### Vision Embedding Model
- **Model**: nomic-ai/nomic-embed-vision-v1.5
- **Size**: ~600MB
- **Dimensions**: 768 (aligned with text model)
- **Use Case**: Image understanding, visual document analysis, multimodal retrieval
- **Features**:
  - Shares embedding space with text model
  - Can be used with text embeddings for cross-modal search

## Using Bundled Models

The embedding service automatically checks for models in this directory:

```python
from services.embedding_service import EmbeddingService

# Will use bundled model if available, otherwise auto-downloads
service = EmbeddingService(device='cpu', model_type='text')
service.initialize()

# Generate embeddings
embeddings = service.generate_embeddings(['Your text here'])
```

No code changes needed - the configuration in `services/resource_path.py` handles detection automatically.

## Model Sizes & Performance

### Disk Space Requirements
- **Text Model Only**: ~550MB
- **Vision Model Only**: ~600MB
- **Both Models**: ~1.15GB

### Inference Performance
- **CPU (text)**: ~10-50ms per batch (batch_size=32)
- **GPU (text)**: ~2-10ms per batch
- **CPU (vision)**: ~50-200ms per image
- **GPU (vision)**: ~10-50ms per image

### Memory Requirements
- **Text Model in Memory**: ~500MB
- **Vision Model in Memory**: ~650MB
- **Both Models**: Not recommended to load simultaneously unless you have 4GB+ RAM

## Size Considerations for Distribution

### Minimal Bundle (No Pre-bundled Models)
- Application size: Baseline (~550MB for dependencies)
- First-run download: ~550MB (text only) or ~1.15GB (both)
- **Recommended for**: Most users with internet access

### Full Bundle (Pre-bundled Models)
- Application size: +1.15GB additional
- No first-run download needed
- **Recommended for**:
  - Offline installations
  - Corporate environments with restricted internet
  - Air-gapped systems

### Partial Bundle (Text Only)
- Application size: +550MB additional
- **Recommended for**: Document-focused workflows without image analysis

## Production Deployment

For production installers (PyInstaller, cx_Freeze, or Tauri bundling):

1. **Include models** in the bundle by adding to package spec or Tauri config
2. **Configure path** - Already handled automatically via `get_base_path()` in `resource_path.py`
3. **Test both scenarios**:
   - With bundled models (offline use)
   - Without bundled models (verify auto-download fallback works)

The current code already handles path resolution for both development and production modes.

## .gitignore

Model files are gitignored to keep the repository lightweight. Only the directory structure (`.gitkeep`) is tracked.

Users either:
1. Auto-download on first run (default)
2. Run `python download_embedding_models.py` for pre-installation
3. CI/CD pipelines can download during build process

**To track specific models in git (not recommended):**
```bash
# In .gitignore, add exception:
!python-backend/models/embeddings/text/
!python-backend/models/embeddings/vision/
```

## Troubleshooting

### Models not loading
- Check directory structure matches above
- Verify `config.json` exists in model directories
- Check file permissions
- Run `python download_embedding_models.py` to reinstall

### Models still downloading despite bundling
- Ensure models are in correct subdirectories (`text/`, `vision/`)
- Check logs for "Found bundled model" messages
- Verify `config.json` is present
- Try deleting HuggingFace cache and rerunning

### Large application size
- Don't bundle vision model if you don't need image analysis
- Consider auto-download approach instead of bundling
- Models are already compressed (safetensors format)

### First-run download is slow
- Models are large (~550-600MB each)
- Download speed depends on internet connection
- Consider pre-bundling models for better UX
- Show progress indicators to users during download

### Out of disk space
- Each model requires 550-600MB
- HuggingFace cache can grow over time
- To clear cache: Delete `~/.cache/huggingface/` (or Windows equivalent)
- Free up space before running download script

## Additional Resources

- **Nomic Embed Documentation**: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- **Sentence Transformers**: https://www.sbert.net/
- **HuggingFace Hub**: https://huggingface.co/docs/huggingface_hub/
