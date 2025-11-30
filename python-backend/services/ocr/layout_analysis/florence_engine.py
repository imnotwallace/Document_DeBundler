"""
Florence-2 Layout Analysis Engine.

Provides semantic layout understanding using Microsoft's Florence-2 model.
Optimized for 4GB VRAM operation with FP16 and optimistic batching.
"""

import gc
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FlorenceEngine:
    """
    Florence-2 layout analysis engine.

    Runs two tasks:
    1. DENSE_REGION_CAPTION - Detect regions with semantic labels
    2. DETAILED_CAPTION - Get overall page description

    Memory usage (FP16):
    - Base model: ~2GB
    - Per image: ~0.3GB
    - Batch size 1: ~2.3GB
    - Batch size 2: ~2.6GB
    """

    # Task prompts
    TASK_DENSE_REGION = "<DENSE_REGION_CAPTION>"
    TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
    TASK_OCR_WITH_REGION = "<OCR_WITH_REGION>"  # Text detection with bounding boxes

    # VRAM requirements (MB)
    VRAM_BASE_MB = 2000
    VRAM_PER_IMAGE_MB = 300

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        """
        Initialize Florence-2 engine.

        Args:
            model_id: Hugging Face model ID
            device: "cuda" or "cpu"
            use_fp16: Use FP16 for lower memory (recommended for 4GB VRAM)
        """
        self.model_id = model_id
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"

        self.model = None
        self.processor = None
        self._initialized = False

    def initialize(self) -> None:
        """Load model and processor."""
        if self._initialized:
            return

        logger.info(f"Loading Florence-2: {self.model_id}")
        start_time = time.time()

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers and torch required for Florence-2. "
                "Install with: pip install transformers torch"
            ) from e

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Load model with appropriate dtype
        dtype = torch.float16 if self.use_fp16 else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # Avoid SDPA compatibility issues
        ).to(self.device)

        # Set to eval mode
        self.model.eval()

        self._initialized = True
        load_time = time.time() - start_time

        logger.info(
            f"Florence-2 loaded in {load_time:.2f}s "
            f"(device={self.device}, fp16={self.use_fp16})"
        )

    def cleanup(self) -> None:
        """Unload model and free memory."""
        if not self._initialized:
            return

        logger.info("Unloading Florence-2...")

        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._initialized = False

        # Aggressive cleanup
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        logger.info("Florence-2 unloaded")

    def analyze_layout(
        self,
        image: np.ndarray,
        page_num: int = 0,
    ) -> Dict[str, Any]:
        """
        Analyze layout of a single image.

        Args:
            image: Image as numpy array (H, W, C) in RGB
            page_num: Page number for metadata

        Returns:
            Dictionary with regions, layout_type, and overall_caption
        """
        if not self._initialized:
            raise RuntimeError("Florence-2 not initialized. Call initialize() first.")

        start_time = time.time()

        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Ensure RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        import torch

        # Run OCR_WITH_REGION for text detection with bounding boxes
        ocr_result = self._run_task(pil_image, self.TASK_OCR_WITH_REGION)

        # Run DENSE_REGION_CAPTION for semantic regions
        dense_result = self._run_task(pil_image, self.TASK_DENSE_REGION)

        # Run DETAILED_CAPTION for overall description
        caption_result = self._run_task(pil_image, self.TASK_DETAILED_CAPTION)

        processing_time = time.time() - start_time

        return {
            "page_number": page_num,
            "ocr_with_region": ocr_result,  # Text boxes with bounding boxes
            "dense_regions": dense_result,
            "detailed_caption": caption_result,
            "image_size": (pil_image.width, pil_image.height),
            "processing_time": processing_time,
        }

    def analyze_batch(
        self,
        images: List[np.ndarray],
        page_nums: Optional[List[int]] = None,
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Analyze layout of multiple images with batching.

        Args:
            images: List of images as numpy arrays
            page_nums: Optional page numbers for metadata
            batch_size: Batch size for processing

        Returns:
            List of analysis results
        """
        if not self._initialized:
            raise RuntimeError("Florence-2 not initialized. Call initialize() first.")

        if page_nums is None:
            page_nums = list(range(len(images)))

        results = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_page_nums = page_nums[i:i + batch_size]

            batch_results = self._process_batch(batch_images, batch_page_nums)
            results.extend(batch_results)

        return results

    def _run_task(self, image: Image.Image, task: str) -> Dict[str, Any]:
        """Run a single task on an image."""
        import torch

        # Prepare inputs
        inputs = self.processor(
            text=task,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Convert to appropriate dtype
        if self.use_fp16:
            inputs = {
                k: v.to(torch.float16) if v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                use_cache=False,  # Disable KV cache to avoid compatibility issues
            )

        # Decode
        output_text = self.processor.batch_decode(
            outputs, skip_special_tokens=False
        )[0]

        # Post-process
        result = self.processor.post_process_generation(
            output_text,
            task=task,
            image_size=(image.width, image.height),
        )

        return result

    def _process_batch(
        self,
        images: List[np.ndarray],
        page_nums: List[int],
    ) -> List[Dict[str, Any]]:
        """Process a batch of images."""
        import torch

        results = []
        batch_start = time.time()

        # Convert all to PIL
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img)
            else:
                pil_img = img

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            pil_images.append(pil_img)

        # Process DENSE_REGION_CAPTION for batch
        dense_inputs = self.processor(
            text=[self.TASK_DENSE_REGION] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if self.use_fp16:
            dense_inputs = {
                k: v.to(torch.float16) if v.dtype == torch.float32 else v
                for k, v in dense_inputs.items()
            }

        with torch.no_grad():
            dense_outputs = self.model.generate(
                input_ids=dense_inputs["input_ids"],
                pixel_values=dense_inputs.get("pixel_values"),
                max_new_tokens=1024,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )

        dense_texts = self.processor.batch_decode(
            dense_outputs, skip_special_tokens=False
        )

        # Process DETAILED_CAPTION for batch
        caption_inputs = self.processor(
            text=[self.TASK_DETAILED_CAPTION] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if self.use_fp16:
            caption_inputs = {
                k: v.to(torch.float16) if v.dtype == torch.float32 else v
                for k, v in caption_inputs.items()
            }

        with torch.no_grad():
            caption_outputs = self.model.generate(
                input_ids=caption_inputs["input_ids"],
                pixel_values=caption_inputs.get("pixel_values"),
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )

        caption_texts = self.processor.batch_decode(
            caption_outputs, skip_special_tokens=False
        )

        batch_time = time.time() - batch_start
        per_page_time = batch_time / len(images)

        # Post-process results
        for i, (pil_img, page_num, dense_text, caption_text) in enumerate(
            zip(pil_images, page_nums, dense_texts, caption_texts)
        ):
            dense_result = self.processor.post_process_generation(
                dense_text,
                task=self.TASK_DENSE_REGION,
                image_size=(pil_img.width, pil_img.height),
            )

            caption_result = self.processor.post_process_generation(
                caption_text,
                task=self.TASK_DETAILED_CAPTION,
                image_size=(pil_img.width, pil_img.height),
            )

            results.append({
                "page_number": page_num,
                "dense_regions": dense_result,
                "detailed_caption": caption_result,
                "image_size": (pil_img.width, pil_img.height),
                "processing_time": per_page_time,
            })

        return results

    def get_vram_estimate(self, batch_size: int = 1) -> int:
        """
        Estimate VRAM usage for given batch size.

        Args:
            batch_size: Number of images in batch

        Returns:
            Estimated VRAM usage in MB
        """
        return self.VRAM_BASE_MB + (batch_size * self.VRAM_PER_IMAGE_MB)

    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized
