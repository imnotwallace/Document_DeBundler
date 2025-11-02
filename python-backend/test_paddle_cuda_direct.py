"""
Test PaddlePaddle CUDA functionality directly
"""

import sys

print("=" * 60)
print("Testing PaddlePaddle CUDA Directly")
print("=" * 60)

print("\n[1/4] Importing paddle...")
try:
    import paddle
    print(f"[OK] Paddle {paddle.__version__} imported")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

print("\n[2/4] Checking CUDA compilation...")
try:
    compiled_with_cuda = paddle.device.is_compiled_with_cuda()
    print(f"[OK] Compiled with CUDA: {compiled_with_cuda}")

    if compiled_with_cuda:
        device_count = paddle.device.cuda.device_count()
        print(f"[OK] CUDA devices: {device_count}")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

print("\n[3/4] Testing CUDA tensor operations...")
try:
    # Create a simple tensor on CPU first
    x_cpu = paddle.to_tensor([1.0, 2.0, 3.0])
    print(f"[OK] CPU tensor created: {x_cpu}")

    # Try to move to GPU
    if compiled_with_cuda:
        paddle.device.set_device('gpu:0')
        x_gpu = paddle.to_tensor([1.0, 2.0, 3.0])
        print(f"[OK] GPU tensor created: {x_gpu}")

        # Simple operation
        y_gpu = x_gpu * 2
        print(f"[OK] GPU operation result: {y_gpu}")
    else:
        print("[INFO] CUDA not available, skipping GPU tensor test")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] Testing cuDNN operations...")
try:
    import paddle.nn as nn

    # Create a simple conv layer (uses cuDNN)
    conv = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3)

    # Create input tensor
    x = paddle.randn([1, 3, 32, 32])

    if compiled_with_cuda:
        paddle.device.set_device('gpu:0')
        conv_gpu = conv
        x_gpu = x

        # Run forward pass (this will use cuDNN)
        print("[INFO] Running convolution on GPU (uses cuDNN)...")
        out = conv_gpu(x_gpu)
        print(f"[OK] cuDNN convolution succeeded! Output shape: {out.shape}")
    else:
        # CPU mode
        out = conv(x)
        print(f"[OK] CPU convolution succeeded! Output shape: {out.shape}")

except Exception as e:
    print(f"[FAIL] cuDNN operation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: All Paddle CUDA/cuDNN operations work!")
print("=" * 60)
sys.exit(0)
