"""
PP-DocLayout Engine for Document Layout Detection.

Replaces Florence-2 with PaddlePaddle's PP-DocLayout model which is specifically
designed for document layout analysis. Detects 23 document-specific categories
including: title, text, image, table, header, footer, etc.

Benefits over Florence-2:
- 22MB model vs 2.5GB (100x smaller)
- 13ms inference vs 10+ seconds (750x faster)
- Document-specific categories vs generic image regions
- Same framework as PaddleOCR (no additional dependencies)
"""

import gc
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# PP-DocLayout category mapping (23 categories)
DOCLAYOUT_CATEGORIES = {
    0: "paragraph_title",
    1: "image",
    2: "text",
    3: "number",
    4: "abstract",
    5: "content",
    6: "figure_title",
    7: "formula",
    8: "table",
    9: "table_title",
    10: "reference",
    11: "doc_title",
    12: "footnote",
    13: "header",
    14: "algorithm",
    15: "footer",
    16: "seal",
    17: "chart_title",
    18: "chart",
    19: "formula_number",
    20: "header_image",
    21: "footer_image",
    22: "sidebar_text",
}

# Categories that contain text (for reading order)
TEXT_CATEGORIES = {
    "paragraph_title", "text", "abstract", "content", "doc_title",
    "footnote", "reference", "table_title", "figure_title", "chart_title",
    "sidebar_text", "header", "footer", "algorithm", "formula",
}

# Categories that are structural (headers/footers)
STRUCTURAL_CATEGORIES = {"header", "footer", "header_image", "footer_image"}

# Reading order priority (lower = read first)
READING_ORDER_PRIORITY = {
    "doc_title": 0,
    "header": 1,
    "header_image": 1,
    "paragraph_title": 2,
    "abstract": 3,
    "text": 4,
    "content": 4,
    "sidebar_text": 4,
    "image": 5,
    "figure_title": 6,
    "chart": 5,
    "chart_title": 6,
    "table": 7,
    "table_title": 8,
    "formula": 9,
    "formula_number": 10,
    "algorithm": 11,
    "reference": 12,
    "footnote": 13,
    "footer": 14,
    "footer_image": 14,
    "seal": 15,
    "number": 16,
}


class DocLayoutEngine:
    """
    PP-DocLayout engine for document layout detection.

    Uses PaddleOCR's LayoutDetection module with PP-DocLayout-M model.
    Much faster and more accurate for documents than Florence-2.
    """

    # Model variants and their characteristics
    MODELS = {
        "PP-DocLayout-L": {"mAP": 90.4, "size_mb": 124, "inference_ms": 34},
        "PP-DocLayout-M": {"mAP": 75.2, "size_mb": 22, "inference_ms": 13},
        "PP-DocLayout-S": {"mAP": 70.9, "size_mb": 5, "inference_ms": 12},
    }

    DEFAULT_MODEL = "PP-DocLayout-M"  # Best balance of speed/accuracy
    VRAM_REQUIREMENT_MB = 200  # Much smaller than Florence-2's 2500MB

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "gpu",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize PP-DocLayout engine.

        Args:
            model_name: Model variant (L, M, or S)
            device: Device to use ("gpu" or "cpu")
            confidence_threshold: Minimum confidence for detections
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold

        self._model = None
        self._initialized = False

        logger.info(
            f"DocLayoutEngine created: model={model_name}, device={device}, "
            f"threshold={confidence_threshold}"
        )

    def initialize(self) -> None:
        """Load the PP-DocLayout model."""
        if self._initialized:
            return

        logger.info(f"Loading PP-DocLayout model: {self.model_name}")
        start_time = time.time()

        try:
            from paddleocr import LayoutDetection

            self._model = LayoutDetection(model_name=self.model_name)
            self._initialized = True

            load_time = time.time() - start_time
            logger.info(f"PP-DocLayout loaded in {load_time:.2f}s")

        except ImportError as e:
            raise ImportError(
                f"paddleocr not installed. Install with: pip install paddleocr\n{e}"
            )
        except Exception as e:
            logger.error(f"Failed to load PP-DocLayout: {e}")
            raise

    def analyze_layout(
        self,
        image: np.ndarray,
        page_num: int = 0,
    ) -> Dict[str, Any]:
        """
        Analyze document layout of a single image.

        Args:
            image: Image as numpy array (H, W, C) in RGB
            page_num: Page number for metadata

        Returns:
            Dictionary with detected regions and layout info
        """
        if not self._initialized:
            raise RuntimeError("DocLayoutEngine not initialized. Call initialize() first.")

        start_time = time.time()

        # Ensure RGB format
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        height, width = image.shape[:2]

        # Run PP-DocLayout
        output = self._model.predict(image, batch_size=1)

        # Parse results
        regions = []
        for res in output:
            boxes = res.get('boxes', []) if hasattr(res, 'get') else []

            for i, box in enumerate(boxes):
                score = box.get('score', 0)

                # Filter by confidence
                if score < self.confidence_threshold:
                    continue

                cls_id = box.get('cls_id', -1)
                label = box.get('label', DOCLAYOUT_CATEGORIES.get(cls_id, 'unknown'))
                coords = box.get('coordinate', [0, 0, 0, 0])

                # Convert coordinates to standard format
                x0, y0, x1, y1 = coords[:4] if len(coords) >= 4 else [0, 0, 0, 0]

                regions.append({
                    'region_id': i,
                    'label': label,
                    'cls_id': cls_id,
                    'confidence': float(score),
                    'bbox': {
                        'x0': float(x0),
                        'y0': float(y0),
                        'x1': float(x1),
                        'y1': float(y1),
                    },
                    'is_text': label in TEXT_CATEGORIES,
                    'is_structural': label in STRUCTURAL_CATEGORIES,
                    'reading_priority': READING_ORDER_PRIORITY.get(label, 99),
                })

        # Sort by reading priority and position
        regions = self._sort_regions_for_reading(regions, width, height)

        # Infer layout type
        layout_type = self._infer_layout_type(regions, width)

        processing_time = time.time() - start_time

        return {
            'page_number': page_num,
            'regions': regions,
            'layout_type': layout_type,
            'image_width': width,
            'image_height': height,
            'processing_time': processing_time,
            'model': self.model_name,
            'num_regions': len(regions),
            'num_text_regions': sum(1 for r in regions if r['is_text']),
        }

    def analyze_batch(
        self,
        images: List[np.ndarray],
        page_nums: Optional[List[int]] = None,
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Analyze layout of multiple images.

        Args:
            images: List of images as numpy arrays
            page_nums: Optional list of page numbers
            batch_size: Batch size for processing

        Returns:
            List of layout results
        """
        if page_nums is None:
            page_nums = list(range(len(images)))

        results = []
        for img, page_num in zip(images, page_nums):
            result = self.analyze_layout(img, page_num)
            results.append(result)

        return results

    def _sort_regions_for_reading(
        self,
        regions: List[Dict],
        page_width: int,
        page_height: int,
    ) -> List[Dict]:
        """Sort regions in reading order."""
        if not regions:
            return regions

        # First pass: sort by priority and vertical position
        def sort_key(r):
            bbox = r['bbox']
            y_center = (bbox['y0'] + bbox['y1']) / 2
            x_center = (bbox['x0'] + bbox['x1']) / 2
            priority = r['reading_priority']

            # Normalize positions
            y_norm = y_center / page_height
            x_norm = x_center / page_width

            # Primary: priority, Secondary: top-to-bottom, Tertiary: left-to-right
            return (priority, y_norm, x_norm)

        sorted_regions = sorted(regions, key=sort_key)

        # Assign reading order
        for i, region in enumerate(sorted_regions):
            region['reading_order'] = i

        return sorted_regions

    def _infer_layout_type(
        self,
        regions: List[Dict],
        page_width: int,
    ) -> str:
        """Infer the overall layout type from detected regions."""
        text_regions = [r for r in regions if r['is_text'] and r['label'] == 'text']

        if not text_regions:
            return "unknown"

        # Check for multi-column layout by analyzing x-positions
        if len(text_regions) >= 2:
            x_centers = [(r['bbox']['x0'] + r['bbox']['x1']) / 2 for r in text_regions]

            # Group by horizontal position (left third, middle, right third)
            left = sum(1 for x in x_centers if x < page_width * 0.4)
            right = sum(1 for x in x_centers if x > page_width * 0.6)

            if left > 0 and right > 0:
                # Check if regions overlap vertically (indicating columns)
                left_regions = [r for r in text_regions
                              if (r['bbox']['x0'] + r['bbox']['x1']) / 2 < page_width * 0.4]
                right_regions = [r for r in text_regions
                               if (r['bbox']['x0'] + r['bbox']['x1']) / 2 > page_width * 0.6]

                if left_regions and right_regions:
                    left_y = (min(r['bbox']['y0'] for r in left_regions),
                             max(r['bbox']['y1'] for r in left_regions))
                    right_y = (min(r['bbox']['y0'] for r in right_regions),
                              max(r['bbox']['y1'] for r in right_regions))

                    # Check vertical overlap
                    overlap = min(left_y[1], right_y[1]) - max(left_y[0], right_y[0])
                    if overlap > 0:
                        return "two_column"

        # Check for table-heavy layout
        table_regions = [r for r in regions if r['label'] in ('table', 'chart')]
        if table_regions:
            table_area = sum(
                (r['bbox']['x1'] - r['bbox']['x0']) * (r['bbox']['y1'] - r['bbox']['y0'])
                for r in table_regions
            )
            total_area = sum(
                (r['bbox']['x1'] - r['bbox']['x0']) * (r['bbox']['y1'] - r['bbox']['y0'])
                for r in regions
            )
            if total_area > 0 and table_area / total_area > 0.3:
                return "table"

        return "single_column"

    def cleanup(self) -> None:
        """Release model resources."""
        logger.info("Cleaning up DocLayoutEngine...")

        if self._model is not None:
            del self._model
            self._model = None

        self._initialized = False

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

        logger.info("DocLayoutEngine cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized

    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        model_info = self.MODELS.get(self.model_name, {})
        return {
            'engine': 'PP-DocLayout',
            'model': self.model_name,
            'initialized': self._initialized,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'expected_mAP': model_info.get('mAP'),
            'model_size_mb': model_info.get('size_mb'),
            'expected_inference_ms': model_info.get('inference_ms'),
            'vram_requirement_mb': self.VRAM_REQUIREMENT_MB,
            'num_categories': len(DOCLAYOUT_CATEGORIES),
        }
