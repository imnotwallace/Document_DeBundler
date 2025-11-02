"""
Debug script to check what paths are added by cuda_path_fix
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("CUDA Path Fix Debug")
print("=" * 60)

print("\n[1] PATH before cuda_path_fix:")
print(os.environ.get('PATH', '')[:500] + "...")

print("\n[2] Importing cuda_path_fix...")
from services.ocr.cuda_path_fix import add_cuda_dlls_to_path
add_cuda_dlls_to_path()

print("\n[3] PATH after cuda_path_fix:")
path_entries = os.environ.get('PATH', '').split(';')
print(f"Total PATH entries: {len(path_entries)}")
print("\nFirst 20 entries:")
for i, entry in enumerate(path_entries[:20]):
    print(f"  {i+1}. {entry}")

print("\n[4] Checking for critical directories in PATH:")
critical_dirs = [
    'nvidia\\cudnn\\bin',
    'torch\\lib',
]

for critical in critical_dirs:
    found = any(critical.lower() in entry.lower() for entry in path_entries)
    status = "✓ FOUND" if found else "✗ MISSING"
    print(f"  {status}: {critical}")
    if found:
        matches = [e for e in path_entries if critical.lower() in e.lower()]
        for match in matches:
            print(f"      → {match}")

print("\n[5] Checking for zlibwapi.dll in directories:")
torch_lib = Path(sys.prefix) / 'Lib' / 'site-packages' / 'torch' / 'lib'
nvidia_cudnn = Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia' / 'cudnn' / 'bin'

for dir_path, name in [(torch_lib, 'torch/lib'), (nvidia_cudnn, 'nvidia/cudnn/bin')]:
    zlib_path = dir_path / 'zlibwapi.dll'
    if zlib_path.exists():
        print(f"  ✓ {name}: {zlib_path}")
    else:
        print(f"  ✗ {name}: NOT FOUND")

print("\n" + "=" * 60)
