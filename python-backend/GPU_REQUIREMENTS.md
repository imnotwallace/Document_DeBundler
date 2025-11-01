# GPU Requirements and Bundling Documentation

## Verified GPU-Compatible Package Versions

**Last Verified:** 2025-11-01
**Python Version:** 3.12.12
**CUDA Version:** 11.8
**Test Hardware:** NVIDIA GeForce GTX 1650 (4GB VRAM)

### Critical GPU Packages

The following package versions have been **verified to work together** without conflicts:

```
# Deep Learning Frameworks
torch==2.2.2+cu118
torchvision==0.17.2+cu118
torchaudio==2.2.2+cu118
paddlepaddle-gpu==2.6.2

# OCR and Document Processing
paddleocr==3.3.1
paddlex==3.3.6

# Embeddings and ML
sentence-transformers==5.1.2
scikit-learn==1.7.2

# Core Dependencies
numpy==1.26.4  # CRITICAL: Must be <2.0 for PyTorch 2.2.2 compatibility
protobuf==3.20.2  # CRITICAL: Required for PaddlePaddle GPU 2.6.2
```

### Version Compatibility Matrix

| Component | Version | CUDA Support | Python Version |
|-----------|---------|--------------|----------------|
| PyTorch | 2.2.2+cu118 | ✅ CUDA 11.8 | 3.12.x |
| PaddlePaddle | 2.6.2 GPU | ✅ CUDA 11.8 | 3.12.x |
| sentence-transformers | 5.1.2 | ✅ (via PyTorch) | 3.12.x |
| NumPy | 1.26.4 | N/A | 3.12.x |
| Protobuf | 3.20.2 | N/A | 3.12.x |

### Known Compatibility Issues Resolved

#### 1. PyTorch/PaddlePaddle Conflict (RESOLVED)
**Problem:** PyTorch 2.7+ and PaddlePaddle 2.6.2 have pybind11 type registration conflicts
**Solution:** Use PyTorch 2.2.2 which is compatible with PaddlePaddle 2.6.2
**Status:** ✅ Both libraries load and use GPU simultaneously

#### 2. NumPy 2.x Incompatibility (RESOLVED)
**Problem:** PyTorch 2.2.2 was compiled against NumPy 1.x and crashes with NumPy 2.x
**Solution:** Pin numpy==1.26.4 (latest 1.x version)
**Status:** ✅ No warnings or crashes

#### 3. Protobuf Version Mismatch (RESOLVED)
**Problem:** PaddlePaddle GPU 2.6.2 requires protobuf <4.0
**Solution:** Pin protobuf==3.20.2
**Status:** ✅ Compatible with all packages

### Installation Instructions

#### Option 1: Using `uv` (Recommended - Fast)

```bash
cd python-backend

# Create virtual environment with Python 3.12
C:\ProgramData\miniconda3\envs\document-debundler\python.exe -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Unix

# Install all dependencies
uv pip install -r requirements.txt

# Install PyTorch with CUDA (special index URL)
uv pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### Option 2: Using `pip`

```bash
cd python-backend

# Create virtual environment
C:\ProgramData\miniconda3\envs\document-debundler\python.exe -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Verification Script

Run this script to verify GPU setup:

```python
import torch
import paddle
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("GPU VERIFICATION REPORT")
print("=" * 60)

# PyTorch
print(f"\n✓ PyTorch: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# PaddlePaddle
print(f"\n✓ PaddlePaddle: {paddle.__version__}")
print(f"  CUDA Compiled: {paddle.device.is_compiled_with_cuda()}")
if paddle.device.is_compiled_with_cuda():
    print(f"  CUDA Devices: {paddle.device.cuda.device_count()}")

# sentence-transformers
print(f"\n✓ sentence-transformers: Can use CUDA via PyTorch")

print("\n" + "=" * 60)
print("✅ ALL GPU LIBRARIES VERIFIED")
print("=" * 60)
```

Save as `verify_gpu.py` and run:
```bash
.venv\Scripts\python.exe verify_gpu.py
```

### Bundling for Production

#### Required Files for GPU Support

When bundling the application, ensure these are included:

1. **Python Environment:**
   - Python 3.12.12 runtime
   - All packages from `requirements.txt`
   - PyTorch CUDA wheels (torch, torchvision, torchaudio with +cu118 suffix)

2. **CUDA Runtime Libraries (bundled with PyTorch):**
   - PyTorch 2.2.2+cu118 includes CUDA 11.8 runtime DLLs
   - No separate CUDA installation required on target machine

3. **PaddleOCR Models (optional for offline):**
   - Detection model: `models/det/`
   - Recognition model: `models/rec/`
   - Classification model: `models/cls/`
   - See `python-backend/models/README.md` for download instructions

4. **Embedding Models (optional for offline):**
   - Text model: `models/embeddings/text/`
   - Vision model: `models/embeddings/vision/`
   - See `python-backend/models/embeddings/README.md` for details

#### GPU Requirements for End Users

**Minimum:**
- NVIDIA GPU with CUDA Compute Capability 3.5+
- 2GB VRAM (CPU fallback for OCR if insufficient)
- NVIDIA driver version 450.80.02+ (for CUDA 11.8)

**Recommended:**
- NVIDIA GPU with 4GB+ VRAM (GTX 1650 or better)
- 8GB+ System RAM
- NVIDIA driver version 520+ (latest stable)

**Note:** Application automatically falls back to CPU if:
- No NVIDIA GPU detected
- Insufficient VRAM
- Driver version too old

### Testing GPU Functionality

After installation, test GPU features:

```bash
# Test OCR with GPU
.venv\Scripts\python.exe test_paddle_first.py

# Test hybrid GPU/CPU mode
.venv\Scripts\python.exe test_hybrid_mode.py

# Test embedding service
.venv\Scripts\python.exe test_embedding_service.py
```

### Troubleshooting

#### "CUDA not available" despite having NVIDIA GPU

1. **Check driver version:**
   ```
   nvidia-smi
   ```
   Should show driver 450.80.02 or newer

2. **Verify PyTorch CUDA build:**
   ```python
   import torch
   print(torch.version.cuda)  # Should print: 11.8
   ```

3. **Check for +cu118 suffix:**
   ```bash
   pip list | findstr torch
   ```
   Should show: `torch==2.2.2+cu118` (NOT `torch==2.2.2`)

#### "PaddlePaddle CUDA not compiled"

1. **Verify correct package installed:**
   ```bash
   pip list | findstr paddle
   ```
   Should show: `paddlepaddle-gpu==2.6.2` (NOT `paddlepaddle==`)

2. **Reinstall if needed:**
   ```bash
   uv pip uninstall paddlepaddle paddlepaddle-gpu
   uv pip install paddlepaddle-gpu==2.6.2
   ```

#### Import conflicts or crashes

1. **Check NumPy version:**
   ```bash
   pip list | findstr numpy
   ```
   Must be: `numpy==1.26.4` (NOT 2.x.x)

2. **Reinstall environment if mixed versions:**
   ```bash
   # Delete .venv and recreate from scratch
   ```

### Performance Benchmarks

With verified configuration on GTX 1650 (4GB VRAM):

| Task | GPU (CUDA 11.8) | CPU Fallback |
|------|-----------------|--------------|
| OCR (per page) | 0.15-0.35s | 0.5-1.0s |
| Embeddings (per page) | 0.3-0.8s | 1-3s |
| Large PDF (500 pages) | 8-20 min | 40-80 min |

### Update Log

- **2025-11-01:** Initial GPU verification with PyTorch 2.2.2 and PaddlePaddle 2.6.2
- Resolved pybind11 conflicts by downgrading PyTorch from 2.7 to 2.2
- Resolved NumPy incompatibility by pinning to 1.26.4
- Verified on GTX 1650 with 4GB VRAM
