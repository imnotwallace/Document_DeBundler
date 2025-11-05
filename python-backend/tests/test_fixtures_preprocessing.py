"""
Test Fixtures for Intelligent Preprocessing Tests

Provides synthetic test images with known characteristics:
- Sharp images
- Blurry images
- Low contrast images
- Noisy images
- High quality scans
- etc.
"""

import numpy as np
import pytest


def create_synthetic_image(
    width: int = 800,
    height: int = 600,
    noise_level: float = 0.0,
    blur_kernel: int = 0,
    contrast_factor: float = 1.0,
    brightness: int = 128
) -> np.ndarray:
    """
    Create synthetic test image with controlled properties.

    Args:
        width: Image width
        height: Image height
        noise_level: Gaussian noise std dev (0-50 typical)
        blur_kernel: Gaussian blur kernel size (0=no blur)
        contrast_factor: Contrast multiplier (0.5=low, 1.0=normal, 2.0=high)
        brightness: Base brightness (0-255)

    Returns:
        Grayscale image as uint8 numpy array
    """
    # Create base image with text-like patterns
    image = np.ones((height, width), dtype=np.float32) * brightness

    # Add horizontal lines (simulating text)
    for y in range(50, height - 50, 40):
        # Add variation to line positions and widths
        line_height = np.random.randint(15, 25)
        image[y:y+line_height, 100:width-100] = brightness - 100

    # Add some vertical variation (simulating characters)
    for x in range(100, width - 100, 20):
        char_width = np.random.randint(8, 15)
        for y in range(50, height - 50, 40):
            line_height = np.random.randint(15, 25)
            if np.random.random() > 0.3:  # Not all positions have chars
                image[y:y+line_height, x:x+char_width] = brightness - 100

    # Apply contrast
    image = (image - 128) * contrast_factor + 128
    image = np.clip(image, 0, 255)

    # Apply blur if requested
    if blur_kernel > 0:
        import cv2
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape)
        image = image + noise
        image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


@pytest.fixture
def sharp_high_contrast_image():
    """High quality scan - sharp and good contrast"""
    return create_synthetic_image(
        noise_level=5.0,
        blur_kernel=0,
        contrast_factor=1.2,
        brightness=200
    )


@pytest.fixture
def blurry_image():
    """Blurry image needing deblurring"""
    return create_synthetic_image(
        noise_level=10.0,
        blur_kernel=21,  # Very heavy blur (increased from 9)
        contrast_factor=0.8,
        brightness=180
    )


@pytest.fixture
def low_contrast_image():
    """Low contrast image needing enhancement"""
    return create_synthetic_image(
        noise_level=15.0,
        blur_kernel=0,
        contrast_factor=0.3,  # Very low contrast
        brightness=150
    )


@pytest.fixture
def noisy_image():
    """Noisy image needing denoising"""
    return create_synthetic_image(
        noise_level=55.0,  # Very heavy noise (increased from 40)
        blur_kernel=0,
        contrast_factor=1.0,
        brightness=180
    )


@pytest.fixture
def perfect_image():
    """Perfect quality image - no preprocessing needed"""
    return create_synthetic_image(
        noise_level=2.0,
        blur_kernel=0,
        contrast_factor=1.5,
        brightness=200
    )


@pytest.fixture
def low_quality_image():
    """Very poor quality - needs multiple techniques"""
    return create_synthetic_image(
        noise_level=55.0,  # Very heavy noise (increased from 35)
        blur_kernel=15,  # Heavy blur (increased from 7)
        contrast_factor=0.3,  # Lower contrast (decreased from 0.4)
        brightness=130
    )


@pytest.fixture
def receipt_image():
    """Simulated thermal receipt - faded, low contrast"""
    return create_synthetic_image(
        noise_level=20.0,
        blur_kernel=0,
        contrast_factor=0.25,  # Very faded
        brightness=220  # High brightness (faded background)
    )


@pytest.fixture
def photo_good_lighting():
    """Camera photo with good lighting"""
    return create_synthetic_image(
        noise_level=15.0,
        blur_kernel=3,  # Slight blur
        contrast_factor=0.9,
        brightness=170
    )


@pytest.fixture
def photo_poor_lighting():
    """Camera photo with poor lighting"""
    return create_synthetic_image(
        noise_level=25.0,
        blur_kernel=5,
        contrast_factor=0.5,
        brightness=100  # Dark
    )


def create_test_image_with_text_boxes(
    num_boxes: int = 5,
    image_size: tuple = (800, 600)
) -> tuple:
    """
    Create test image with known text box positions.

    Returns:
        Tuple of (image, bounding_boxes)
        where bounding_boxes is list of [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    """
    width, height = image_size
    image = np.ones((height, width), dtype=np.uint8) * 255

    bounding_boxes = []

    for i in range(num_boxes):
        # Random position and size
        x0 = np.random.randint(50, width - 200)
        y0 = np.random.randint(50, height - 100)
        box_width = np.random.randint(100, 180)
        box_height = np.random.randint(30, 60)

        # Draw black rectangle (text box)
        image[y0:y0+box_height, x0:x0+box_width] = 50

        # Store bounding box in format [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        bbox = [
            [x0, y0],
            [x0 + box_width, y0],
            [x0 + box_width, y0 + box_height],
            [x0, y0 + box_height]
        ]
        bounding_boxes.append(bbox)

    return image, bounding_boxes


@pytest.fixture
def image_with_known_boxes():
    """Image with known bounding box positions for coordinate tests"""
    return create_test_image_with_text_boxes(num_boxes=5)


def add_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Add Gaussian blur to image"""
    import cv2
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_noise(image: np.ndarray, noise_level: float = 30.0) -> np.ndarray:
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def reduce_contrast(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Reduce image contrast"""
    mean = np.mean(image)
    low_contrast = (image - mean) * factor + mean
    return np.clip(low_contrast, 0, 255).astype(np.uint8)


@pytest.fixture
def image_transformation_functions():
    """Provide functions for image transformations in tests"""
    return {
        'blur': add_gaussian_blur,
        'noise': add_noise,
        'reduce_contrast': reduce_contrast
    }
