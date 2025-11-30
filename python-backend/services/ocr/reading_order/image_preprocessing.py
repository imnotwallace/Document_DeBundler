"""
Image preprocessing for reading order detection.

Includes deskewing and other image-level corrections that should
happen before OCR to improve text alignment.
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - image deskewing disabled")


def detect_skew_angle(image: np.ndarray) -> float:
    """
    Detect the skew angle of a document image using Hough transform.

    Args:
        image: Input image as numpy array (grayscale or color)

    Returns:
        Skew angle in degrees (positive = clockwise rotation needed)
    """
    if not CV2_AVAILABLE:
        return 0.0

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        logger.debug("No lines detected for skew estimation")
        return 0.0

    # Calculate angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > 10:  # Avoid vertical lines
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Only consider near-horizontal lines (within 45 degrees)
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return 0.0

    # Use median angle to be robust against outliers
    median_angle = np.median(angles)

    logger.debug(f"Detected skew angle: {median_angle:.2f} degrees from {len(angles)} lines")

    return float(median_angle)


def deskew_image(image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Deskew a document image by rotating to correct for tilt.

    Args:
        image: Input image as numpy array
        angle: Skew angle in degrees. If None, will be detected automatically.

    Returns:
        Tuple of (deskewed image, angle used)
    """
    if not CV2_AVAILABLE:
        logger.warning("OpenCV not available - cannot deskew image")
        return image, 0.0

    if angle is None:
        angle = detect_skew_angle(image)

    if abs(angle) < 0.1:  # Less than 0.1 degrees - not worth rotating
        return image, angle

    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image dimensions to avoid cropping
    cos_angle = abs(np.cos(np.radians(angle)))
    sin_angle = abs(np.sin(np.radians(angle)))
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)

    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Apply rotation
    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )

    logger.info(f"Deskewed image by {angle:.2f} degrees")

    return deskewed, angle


def preprocess_for_ocr(image: np.ndarray, deskew: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Apply preprocessing to improve OCR accuracy.

    Args:
        image: Input image as numpy array
        deskew: Whether to apply deskewing

    Returns:
        Tuple of (processed image, metadata dict with applied transformations)
    """
    metadata = {
        'deskew_angle': 0.0,
        'original_size': image.shape[:2]
    }

    processed = image.copy()

    if deskew:
        processed, angle = deskew_image(processed)
        metadata['deskew_angle'] = angle
        metadata['deskewed_size'] = processed.shape[:2]

    return processed, metadata


def is_deskewing_available() -> bool:
    """Check if deskewing is available (OpenCV installed)."""
    return CV2_AVAILABLE
