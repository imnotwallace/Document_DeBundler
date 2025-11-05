"""
CUDA Path Fix for PyTorch cuDNN DLL Loading

This module ensures PyTorch can find the NVIDIA cuDNN DLLs without modifying
the system PATH. It should be imported before any PyTorch or PaddleOCR imports.

Issue: PyTorch looks for cuDNN DLLs (cudnn_adv_train64_8.dll, etc.) but can't
find them even though they're installed via nvidia-cudnn-cu11 package.

Solution: Add the nvidia-cudnn bin directory to the process PATH temporarily.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def add_cuda_dlls_to_path():
    """
    Add ALL NVIDIA CUDA DLL directories to the PATH for this process only.
    Handles cuDNN, cuBLAS, cuSolver, cuSPARSE, cuFFT, cuRAND, and CUDA Runtime.
    This does NOT modify the system PATH permanently.

    Returns:
        bool: True if paths were added successfully, False otherwise
    """
    try:
        # List of all NVIDIA CUDA packages to check
        nvidia_packages = [
            'nvidia.cudnn',      # cuDNN
            'nvidia.cublas',     # cuBLAS
            'nvidia.cusolver',   # cuSOLVER
            'nvidia.cusparse',   # cuSPARSE
            'nvidia.cufft',      # cuFFT
            'nvidia.curand',     # cuRAND
            'nvidia.cuda_runtime',  # CUDA Runtime
            'nvidia.cuda_nvrtc',    # NVRTC
        ]

        dll_dirs = []

        # Find DLL directories for each NVIDIA package
        for package_name in nvidia_packages:
            try:
                # Dynamically import the package
                package = __import__(package_name, fromlist=[''])
                
                # Skip if package has no __file__ (namespace packages)
                if package.__file__ is None:
                    logger.debug(f"{package_name} has no __file__ (namespace package), skipping")
                    continue
                
                package_base = Path(package.__file__).parent

                # Check common locations for DLLs
                possible_dirs = [
                    package_base / 'lib',
                    package_base / 'bin',
                    package_base,  # Sometimes DLLs are directly in the package
                ]

                for dll_dir in possible_dirs:
                    if dll_dir.exists():
                        # Check if directory contains DLLs
                        has_dlls = any(dll_dir.glob('*.dll'))
                        if has_dlls and dll_dir not in dll_dirs:
                            dll_dirs.append(dll_dir)
                            logger.debug(f"Found DLLs in {package_name}: {dll_dir}")

            except ImportError:
                logger.debug(f"{package_name} not installed, skipping")
                continue

        # Also add PyTorch's lib directory (without importing torch to avoid conflicts)
        try:
            # Find torch installation by looking in site-packages
            import site
            for site_dir in site.getsitepackages():
                torch_lib = Path(site_dir) / 'torch' / 'lib'
                if torch_lib.exists() and any(torch_lib.glob('*.dll')):
                    if torch_lib not in dll_dirs:
                        dll_dirs.append(torch_lib)
                        logger.debug(f"Found DLLs in torch: {torch_lib}")
                    break
        except Exception as e:
            logger.debug(f"Could not find PyTorch lib directory: {e}")

        if not dll_dirs:
            logger.warning("No NVIDIA CUDA DLL directories found")
            return False

        # Add directories to PATH (only for this process)
        original_path = os.environ.get('PATH', '')
        new_paths = [str(d) for d in dll_dirs]

        # Check if already in PATH
        paths_to_add = []
        for new_path in new_paths:
            if new_path not in original_path:
                paths_to_add.append(new_path)

        if paths_to_add:
            # Add to PATH (temporary, only for this process)
            os.environ['PATH'] = ';'.join(paths_to_add) + ';' + original_path
            logger.info(f"Added {len(paths_to_add)} CUDA DLL path(s) to process PATH:")
            for path in paths_to_add:
                logger.info(f"  - {path}")
            return True
        else:
            logger.debug("CUDA DLL paths already in PATH")
            return True

    except Exception as e:
        logger.warning(f"Failed to add CUDA DLL paths: {e}")
        return False


def verify_cuda_dlls():
    """
    Verify that cuDNN DLLs can be found.

    Returns:
        dict: Status information about DLL availability
    """
    result = {
        'nvidia_cudnn_installed': False,
        'dlls_found': False,
        'dll_paths': [],
        'missing_dlls': []
    }

    try:
        import nvidia.cudnn
        result['nvidia_cudnn_installed'] = True

        cudnn_base = Path(nvidia.cudnn.__file__).parent

        # Expected DLLs
        expected_dlls = [
            'cudnn64_8.dll',
            'cudnn_adv_infer64_8.dll',
            'cudnn_adv_train64_8.dll',
            'cudnn_cnn_infer64_8.dll',
            'cudnn_cnn_train64_8.dll',
            'cudnn_ops_infer64_8.dll',
            'cudnn_ops_train64_8.dll',
        ]

        # Search for DLLs
        for dll_name in expected_dlls:
            dll_path = None
            for subdir in ['lib', 'bin', '.']:
                check_path = cudnn_base / subdir / dll_name
                if check_path.exists():
                    dll_path = check_path
                    break

            if dll_path:
                result['dll_paths'].append(str(dll_path))
            else:
                result['missing_dlls'].append(dll_name)

        result['dlls_found'] = len(result['missing_dlls']) == 0

    except ImportError:
        pass

    return result


# Automatically apply the fix when this module is imported
# This ensures it's applied before any PyTorch/PaddleOCR imports
add_cuda_dlls_to_path()
