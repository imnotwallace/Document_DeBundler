"""
Utility functions for reading order detection.

Helper functions used across multiple layers.
"""

import numpy as np
import re
from typing import List
from .data_structures import WordBox, Line


def calculate_vertical_overlap(word1: WordBox, word2: WordBox) -> float:
    """
    Calculate vertical overlap between two words.

    Args:
        word1, word2: WordBox instances

    Returns:
        Float between 0.0 and 1.0, representing overlap ratio
        relative to the smaller word's height.
        1.0 means complete overlap, 0.0 means no overlap.
    """
    overlap_top = max(word1.y0, word2.y0)
    overlap_bottom = min(word1.y1, word2.y1)
    overlap_height = max(0.0, overlap_bottom - overlap_top)

    min_height = min(word1.height, word2.height)

    if min_height == 0:
        return 0.0

    return overlap_height / min_height


def calculate_horizontal_overlap(line1: Line, line2: Line) -> float:
    """
    Calculate horizontal overlap between two lines.

    Args:
        line1, line2: Line instances

    Returns:
        Float between 0.0 and 1.0, representing overlap ratio
        relative to the smaller line's width.
    """
    overlap_left = max(line1.x0, line2.x0)
    overlap_right = min(line1.x1, line2.x1)
    overlap_width = max(0.0, overlap_right - overlap_left)

    min_width = min(line1.width, line2.width)

    if min_width == 0:
        return 0.0

    return overlap_width / min_width


def gaussian_smooth(array: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to an array.

    Used for smoothing horizontal projection histograms
    to reduce noise before column detection.

    Args:
        array: 1D numpy array to smooth
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Smoothed array of same length
    """
    kernel_size = int(sigma * 6)  # ±3 sigma covers 99.7%
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size

    # Create Gaussian kernel
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)  # Normalize

    # Apply convolution with padding
    padded = np.pad(array, pad_width=kernel_size // 2, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='same')

    # Remove padding
    return smoothed[kernel_size // 2: -kernel_size // 2]


def normalize_whitespace(text: str, max_consecutive_newlines: int = 2) -> str:
    """
    Normalize whitespace in text.

    - Removes excessive newlines
    - Removes trailing whitespace on each line
    - Removes leading/trailing whitespace from entire text

    Args:
        text: Input text
        max_consecutive_newlines: Maximum consecutive newlines allowed

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove excessive newlines
    pattern = "\n{" + str(max_consecutive_newlines + 1) + ",}"
    replacement = "\n" * max_consecutive_newlines
    text = re.sub(pattern, replacement, text)

    # Remove trailing whitespace on each line
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace from entire text
    text = text.strip()

    return text


def get_average_word_height(words: List[WordBox]) -> float:
    """
    Calculate average word height.

    Args:
        words: List of WordBox instances

    Returns:
        Average height in same units as word coordinates
    """
    if not words:
        return 0.0

    heights = [word.height for word in words if word.height > 0]

    if not heights:
        return 0.0

    return sum(heights) / len(heights)


def get_average_line_height(lines: List[Line]) -> float:
    """
    Calculate average line height.

    Args:
        lines: List of Line instances

    Returns:
        Average height in same units as line coordinates
    """
    if not lines:
        return 0.0

    heights = [line.height for line in lines if line.height > 0]

    if not heights:
        return 0.0

    return sum(heights) / len(heights)


def group_by_vertical_position(items: List, tolerance: float = 10.0) -> List[List]:
    """
    Group items (words or lines) by vertical position into horizontal bands.

    Used for table detection and row-based processing.

    Args:
        items: List of items with y0 attribute
        tolerance: Maximum Y-distance to group into same band (pixels)

    Returns:
        List of lists, each inner list is a horizontal band
    """
    if not items:
        return []

    # Sort by vertical position
    sorted_items = sorted(items, key=lambda item: item.y0)

    bands = []
    current_band = [sorted_items[0]]

    for item in sorted_items[1:]:
        # Check if item is close to current band
        band_y = sum(i.y0 for i in current_band) / len(current_band)
        distance = abs(item.y0 - band_y)

        if distance < tolerance:
            current_band.append(item)
        else:
            bands.append(current_band)
            current_band = [item]

    bands.append(current_band)

    return bands


def calculate_median_height(items: List) -> float:
    """
    Calculate median height of items (words or lines).

    Median is more robust to outliers than mean.

    Args:
        items: List of items with height attribute

    Returns:
        Median height
    """
    if not items:
        return 0.0

    heights = sorted([item.height for item in items if item.height > 0])

    if not heights:
        return 0.0

    n = len(heights)
    if n % 2 == 0:
        return (heights[n // 2 - 1] + heights[n // 2]) / 2.0
    else:
        return heights[n // 2]


def merge_adjacent_bboxes(items: List) -> tuple:
    """
    Calculate merged bounding box from list of items.

    Args:
        items: List of items with x0, y0, x1, y1 attributes

    Returns:
        Tuple (x0, y0, x1, y1) representing merged bounding box
    """
    if not items:
        return (0.0, 0.0, 0.0, 0.0)

    x0 = min(item.x0 for item in items)
    y0 = min(item.y0 for item in items)
    x1 = max(item.x1 for item in items)
    y1 = max(item.y1 for item in items)

    return (x0, y0, x1, y1)
