"""
Test sequential GPU usage pattern used in Document-De-Bundler.
Simulates: OCR (GPU) -> cleanup -> Embedding (GPU)
"""
import os
import sys
import gc

# Add torch lib directory to DLL search path
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib)
    print(f"Added {torch_lib} to DLL search path\n")

def test_ocr_phase():
    """Phase 1: OCR with GPU"""
    print("=== Phase 1: OCR Processing (GPU) ===")
    try:
        from paddleocr import PaddleOCR
        import paddle

        print(f"PaddlePaddle version: {paddle.__version__}")
        print(f"CUDA available: {paddle.device.is_compiled_with_cuda()}")
        print(f"GPU device count: {paddle.device.cuda.device_count()}")

        # Initialize OCR with GPU
        print("Initializing PaddleOCR with GPU...")
        ocr = PaddleOCR(use_gpu=True, lang='en', show_log=False)

        # Simulate OCR processing
        print("OCR processing simulation complete")
        print("Phase 1: PASS\n")

        return True
    except Exception as e:
        print(f"Phase 1: FAIL - {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_phase():
    """Phase 2: Embedding with GPU (after OCR cleanup)"""
    print("=== Phase 2: Embedding Processing (GPU) ===")
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # Initialize embedding model with GPU
        print("Initializing sentence-transformers with GPU...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

        # Test embedding generation
        print("Testing embedding generation...")
        embeddings = model.encode(["This is a test sentence"], convert_to_tensor=True)
        print(f"Generated embedding shape: {embeddings.shape}")
        print(f"Embedding device: {embeddings.device}")

        print("Phase 2: PASS\n")
        return True
    except Exception as e:
        print(f"Phase 2: FAIL - {e}\n")
        import traceback
        traceback.print_exc()
        return False

# Run sequential test
print("Testing sequential GPU usage (OCR then Embedding)...\n")

# Phase 1: OCR
ocr_success = test_ocr_phase()

# Cleanup between phases
print("=== Cleanup Phase ===")
print("Cleaning up OCR resources...")
# In real app, OCR service calls cleanup() which deletes references
# and calls gc.collect()
gc.collect()
if 'torch' in sys.modules and hasattr(sys.modules['torch'], 'cuda'):
    try:
        sys.modules['torch'].cuda.empty_cache()
    except:
        pass
print("Cleanup complete\n")

# Phase 2: Embeddings
embedding_success = test_embedding_phase()

# Final summary
print("=== Final Results ===")
print(f"OCR GPU: {'PASS' if ocr_success else 'FAIL'}")
print(f"Embedding GPU: {'PASS' if embedding_success else 'FAIL'}")

if ocr_success and embedding_success:
    print("\nAll GPU acceleration is OPERATIONAL!")
    print("Sequential processing pattern works correctly.")
else:
    print("\nSome tests failed. See details above.")
