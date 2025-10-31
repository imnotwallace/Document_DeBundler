"""
Test script for resource_path module
"""
import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly without going through services/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("resource_path", Path(__file__).parent / "services" / "resource_path.py")
resource_path = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource_path)

def test_resource_paths():
    """Test resource path resolution"""
    print("=" * 60)
    print("Resource Path Resolution Test")
    print("=" * 60)

    # Test base path
    base_path = resource_path.get_base_path()
    print(f"\nBase path: {base_path}")
    print(f"Base path exists: {base_path.exists()}")

    # Test bin path
    bin_path = resource_path.get_bin_path()
    print(f"\nBin path: {bin_path}")
    print(f"Bin path exists: {bin_path.exists()}")

    # Test tesseract path
    tesseract_path = resource_path.get_tesseract_path()
    print(f"\nTesseract path: {tesseract_path}")
    if tesseract_path:
        print(f"Tesseract exists: {tesseract_path.exists()}")
    else:
        print("Tesseract not found (expected - binaries not yet installed)")

    # Test tessdata path
    tessdata_path = resource_path.get_tessdata_path()
    print(f"\nTessdata path: {tessdata_path}")
    if tessdata_path:
        print(f"Tessdata exists: {tessdata_path.exists()}")
    else:
        print("Tessdata not found (expected - binaries not yet installed)")

    # Test production mode detection
    is_prod = resource_path.is_production_mode()
    print(f"\nProduction mode: {is_prod}")
    print(f"Expected: False (running in development)")

    # Test environment setup
    print("\n" + "=" * 60)
    print("Tesseract Environment Setup")
    print("=" * 60)

    config = resource_path.setup_tesseract_environment()
    print(f"\nMode: {config['mode']}")
    print(f"Tesseract command: {config['tesseract_cmd']}")
    print(f"Tessdata prefix: {config['tessdata_prefix']}")

    # Test verification
    print("\n" + "=" * 60)
    print("Tesseract Verification")
    print("=" * 60)

    success, message = resource_path.verify_tesseract_setup()
    print(f"\nSuccess: {success}")
    print(f"Message: {message}")

    # Test generic resource path function
    print("\n" + "=" * 60)
    print("Generic Resource Path")
    print("=" * 60)

    test_resource = resource_path.get_resource_path("bin/tesseract/tesseract.exe")
    print(f"\nResource path (bin/tesseract/tesseract.exe): {test_resource}")
    print(f"Resource exists: {test_resource.exists()}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_resource_paths()
