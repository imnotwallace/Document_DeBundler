"""
Test hybrid GPU/CPU usage: PaddleOCR (CPU) + sentence-transformers (GPU).
This is the practical solution to avoid PyTorch/PaddlePaddle conflicts.
"""
import os
import sys

print("=== Hybrid GPU/CPU Configuration Test ===\n")

# Test 1: PaddleOCR with CPU (no DLL path setup needed)
print("Phase 1: PaddleOCR (CPU mode)")
try:
    from paddleocr import PaddleOCR

    # Initialize PaddleOCR (defaults to CPU if CUDA unavailable)
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(lang='en')
    print("PaddleOCR (CPU): PASS\n")
    paddleocr_success = True
except Exception as e:
    print(f"PaddleOCR (CPU): FAIL - {e}\n")
    import traceback
    traceback.print_exc()
    paddleocr_success = False

# Test 2: sentence-transformers with GPU
print("Phase 2: sentence-transformers (GPU mode)")
try:
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")

        # Initialize with GPU
        print("Initializing sentence-transformers with GPU...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

        # Test embedding generation
        print("Testing embedding generation...")
        embeddings = model.encode(["Test sentence"], convert_to_tensor=True)
        print(f"Embedding device: {embeddings.device}")
        print("sentence-transformers (GPU): PASS\n")
        embedding_success = True
    else:
        print("CUDA not available for PyTorch")
        embedding_success = False

except Exception as e:
    print(f"sentence-transformers (GPU): FAIL - {e}\n")
    import traceback
    traceback.print_exc()
    embedding_success = False

# Summary
print("=== Final Configuration ===")
print(f"PaddleOCR (CPU): {'OPERATIONAL' if paddleocr_success else 'FAILED'}")
print(f"sentence-transformers (GPU): {'OPERATIONAL' if embedding_success else 'FAILED'}")

if paddleocr_success and embedding_success:
    print("\nHybrid configuration is OPERATIONAL!")
    print("Recommendation: Use CPU for OCR, GPU for embeddings.")
    print("This avoids PyTorch/PaddlePaddle conflicts and provides good performance.")
else:
    print("\nConfiguration test failed. See details above.")
