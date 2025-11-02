"""
Test PyTorch installation and CUDA support
"""

print("=" * 60)
print("PyTorch Installation Test")
print("=" * 60)

try:
    import torch
    print(f"✓ PyTorch imported successfully")
    print(f"  Version: {torch.__version__}")
    print()

    print("CUDA Support:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        print()

    print("Testing tensor creation:")
    x = torch.randn(5, 5)
    print(f"  CPU tensor: {x.shape}")

    if torch.cuda.is_available():
        x_gpu = x.cuda()
        print(f"  GPU tensor: {x_gpu.shape} (device: {x_gpu.device})")
    print()

    print("=" * 60)
    print("PyTorch test PASSED")
    print("=" * 60)

except Exception as e:
    print(f"✗ PyTorch test FAILED: {e}")
    import traceback
    traceback.print_exc()
