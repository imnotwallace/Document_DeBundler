"""
Advanced Preprocessing Techniques for OCR

Provides sophisticated preprocessing methods beyond basic operations:
- Richardson-Lucy deblurring
- Adaptive binarization (Sauvola, Wolf)
- Morphological operations

These techniques are more computationally intensive but provide
better results for challenging document images.
"""

import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


def richardson_lucy_deblur(
    image: np.ndarray,
    iterations: int = 5,
    psf_size: int = 5
) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution for advanced deblurring.

    More sophisticated than simple sharpening, can recover from
    motion blur and defocus blur using iterative deconvolution.

    Args:
        image: Input image (grayscale or RGB)
        iterations: Number of iterations (3-10 typical, default 5 balanced for OCR)
        psf_size: Point spread function kernel size (3-7 typical)

    Returns:
        Deblurred image

    Example:
        deblurred = richardson_lucy_deblur(blurry_image, iterations=5)
    """
    try:
        import cv2
        from scipy import signal

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            was_color = True
        else:
            gray = image.copy()
            was_color = False

        # Create PSF (point spread function) - assume Gaussian blur
        psf = _create_gaussian_psf(psf_size)

        # Convert to float
        img_float = gray.astype(np.float64) / 255.0

        # Richardson-Lucy deconvolution
        deblurred = img_float.copy()
        for i in range(iterations):
            # Convolve current estimate with PSF
            conv = signal.convolve2d(deblurred, psf, mode='same', boundary='symm')

            # Avoid division by zero
            conv = np.maximum(conv, 1e-10)

            # Calculate relative blur
            relative_blur = img_float / conv

            # Convolve with flipped PSF
            psf_flipped = np.flip(psf)
            correction = signal.convolve2d(
                relative_blur, psf_flipped, mode='same', boundary='symm'
            )

            # Update estimate
            deblurred = deblurred * correction

            # Clip to valid range
            deblurred = np.clip(deblurred, 0, 1)

        # Convert back to uint8
        result = (deblurred * 255).astype(np.uint8)

        # Convert back to RGB if input was color
        if was_color:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        logger.debug(f"Applied Richardson-Lucy deblurring ({iterations} iterations)")
        return result

    except Exception as e:
        logger.warning(f"Richardson-Lucy deblurring failed: {e}")
        return image


def _create_gaussian_psf(size: int) -> np.ndarray:
    """
    Create Gaussian point spread function (PSF) kernel.

    Args:
        size: Kernel size (will be made odd if even)

    Returns:
        Normalized PSF kernel
    """
    # Ensure odd size
    if size % 2 == 0:
        size += 1

    # Create 1D Gaussian
    sigma = size / 6.0
    x = np.linspace(-size // 2, size // 2, size)
    gaussian_1d = np.exp(-x**2 / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()

    # Create 2D Gaussian (outer product)
    psf = np.outer(gaussian_1d, gaussian_1d)

    # Normalize
    psf /= psf.sum()

    return psf


def adaptive_binarize_sauvola(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2
) -> np.ndarray:
    """
    Apply Sauvola's adaptive binarization.

    Better than Otsu's method for documents with uneven illumination
    and low contrast. Uses local statistics to compute adaptive threshold.

    Args:
        image: Input image (grayscale or RGB)
        window_size: Local window size for adaptive threshold (must be odd)
        k: Sensitivity parameter (0.2-0.5 typical, higher = more aggressive)

    Returns:
        Binarized image (0 or 255)

    Reference:
        Sauvola, J. and Pietikainen, M., "Adaptive document image binarization"

    Example:
        binary = adaptive_binarize_sauvola(low_contrast_image, k=0.3)
    """
    try:
        import cv2

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1

        # Compute local mean and std dev
        mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
        mean_sq = cv2.blur(
            (gray.astype(np.float64))**2,
            (window_size, window_size)
        )
        std_dev = np.sqrt(mean_sq - mean**2)

        # Sauvola threshold: T(x,y) = mean(x,y) * (1 + k * (std_dev(x,y)/R - 1))
        R = 128.0  # Dynamic range of std dev
        threshold = mean * (1 + k * ((std_dev / R) - 1))

        # Binarize
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)

        logger.debug(f"Applied Sauvola binarization (window={window_size}, k={k})")
        return binary

    except Exception as e:
        logger.warning(f"Sauvola binarization failed: {e}")
        return image


def adaptive_binarize_wolf(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.5
) -> np.ndarray:
    """
    Apply Wolf's adaptive binarization.

    Variant of Sauvola with better performance on very low contrast
    documents. Uses global statistics in addition to local.

    Args:
        image: Input image (grayscale or RGB)
        window_size: Local window size for adaptive threshold (must be odd)
        k: Sensitivity parameter (0.2-0.5 typical)

    Returns:
        Binarized image (0 or 255)

    Reference:
        Wolf, C. and Jolion, J., "Extraction and Recognition of Artificial
        Text in Multimedia Documents"

    Example:
        binary = adaptive_binarize_wolf(faded_document, k=0.5)
    """
    try:
        import cv2

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1

        # Compute local mean and std dev
        mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
        mean_sq = cv2.blur(
            (gray.astype(np.float64))**2,
            (window_size, window_size)
        )
        std_dev = np.sqrt(mean_sq - mean**2)

        # Compute global mean and std dev
        global_mean = np.mean(gray)
        global_std = np.std(gray)

        # Wolf threshold (modified Sauvola)
        R = 128.0
        threshold = mean - k * (
            (mean - global_mean) * (1 - std_dev / R)
        )

        # Binarize
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)

        logger.debug(f"Applied Wolf binarization (window={window_size}, k={k})")
        return binary

    except Exception as e:
        logger.warning(f"Wolf binarization failed: {e}")
        return image


def morph_open(
    image: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: str = 'rect'
) -> np.ndarray:
    """
    Apply morphological opening (erosion then dilation).

    Removes small noise/artifacts while preserving structure.
    Best used on binary images.

    Args:
        image: Input image (binary recommended)
        kernel_size: Structuring element size
        kernel_shape: 'rect', 'ellipse', or 'cross'

    Returns:
        Morphologically opened image

    Example:
        clean = morph_open(binary_noisy, kernel_size=3)
    """
    try:
        import cv2

        # Select kernel shape
        if kernel_shape == 'ellipse':
            shape = cv2.MORPH_ELLIPSE
        elif kernel_shape == 'cross':
            shape = cv2.MORPH_CROSS
        else:
            shape = cv2.MORPH_RECT

        kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        logger.debug(f"Applied morphological opening (kernel={kernel_size}, shape={kernel_shape})")
        return opened

    except Exception as e:
        logger.warning(f"Morphological opening failed: {e}")
        return image


def morph_close(
    image: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: str = 'rect'
) -> np.ndarray:
    """
    Apply morphological closing (dilation then erosion).

    Fills small gaps in text while preserving structure.
    Best used on binary images.

    Args:
        image: Input image (binary recommended)
        kernel_size: Structuring element size
        kernel_shape: 'rect', 'ellipse', or 'cross'

    Returns:
        Morphologically closed image

    Example:
        filled = morph_close(binary_with_gaps, kernel_size=3)
    """
    try:
        import cv2

        # Select kernel shape
        if kernel_shape == 'ellipse':
            shape = cv2.MORPH_ELLIPSE
        elif kernel_shape == 'cross':
            shape = cv2.MORPH_CROSS
        else:
            shape = cv2.MORPH_RECT

        kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        logger.debug(f"Applied morphological closing (kernel={kernel_size}, shape={kernel_shape})")
        return closed

    except Exception as e:
        logger.warning(f"Morphological closing failed: {e}")
        return image


def apply_technique(
    image: np.ndarray,
    technique_name: str,
    **kwargs
) -> np.ndarray:
    """
    Apply a named preprocessing technique.

    Convenience function for applying techniques by name.

    Args:
        image: Input image
        technique_name: Name of technique to apply
        **kwargs: Technique-specific parameters

    Returns:
        Processed image

    Supported techniques:
        - 'richardson_lucy': Richardson-Lucy deblurring
        - 'sauvola': Sauvola adaptive binarization
        - 'wolf': Wolf adaptive binarization
        - 'morph_open': Morphological opening
        - 'morph_close': Morphological closing

    Example:
        deblurred = apply_technique(image, 'richardson_lucy', iterations=10)
    """
    technique_map = {
        'richardson_lucy': richardson_lucy_deblur,
        'sauvola': adaptive_binarize_sauvola,
        'wolf': adaptive_binarize_wolf,
        'morph_open': morph_open,
        'morph_close': morph_close,
    }

    if technique_name not in technique_map:
        logger.warning(f"Unknown technique: {technique_name}")
        return image

    technique_func = technique_map[technique_name]
    return technique_func(image, **kwargs)
