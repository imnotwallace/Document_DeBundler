"""
Embedding Service for Document De-bundling
Generates semantic embeddings using Nomic Embed v1.5
Supports both text and vision embeddings with local model bundling.
"""

import logging
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages embedding generation using Nomic Embed v1.5.

    Features:
    - Text embeddings (768-dim) or Vision embeddings
    - Pre-bundled model support with auto-download fallback
    - Batch processing on CPU or GPU
    - Progress callbacks
    - Memory-efficient processing
    """

    def __init__(self, device: str = 'cuda', model_type: str = 'multimodal'):
        """
        Initialize embedding service.

        Args:
            device: 'cpu' or 'cuda' (default: cuda for GPU acceleration)
            model_type: 'text', 'vision', or 'multimodal' (default: multimodal for both)

        Note: OCR and embedding don't run simultaneously - OCR is unloaded before
        embedding models are loaded, so GPU memory conflict is not an issue.
        """
        self.device = device
        self.model_type = model_type
        self.text_model = None
        self.vision_model = None
        self.vision_processor = None  # For vision model image preprocessing
        self._initialized = False

    def _get_bundled_text_model_path(self) -> Optional[Path]:
        """Get path to bundled text model if available."""
        try:
            from .resource_path import get_text_embedding_path
            model_path = get_text_embedding_path()

            if model_path and model_path.exists() and (model_path / "config.json").exists():
                logger.info(f"Found bundled text model at: {model_path}")
                return model_path

            logger.debug("Bundled text model not found")
            return None
        except Exception as e:
            logger.debug(f"Error checking for bundled text model: {e}")
            return None

    def _get_bundled_vision_model_path(self) -> Optional[Path]:
        """
        Get path to bundled vision model if available and complete.

        Vision models need additional sentence-transformers metadata files
        (modules.json, config_sentence_transformers.json) to load properly.
        If these are missing, fall back to HuggingFace download.
        """
        try:
            from .resource_path import get_vision_embedding_path
            model_path = get_vision_embedding_path()

            if not model_path or not model_path.exists():
                logger.debug("Bundled vision model path not found")
                return None

            # Check for required files
            required_files = [
                "config.json",
                "modules.json",  # Required for sentence-transformers
                "config_sentence_transformers.json"  # Version metadata
            ]

            missing_files = [f for f in required_files if not (model_path / f).exists()]

            if missing_files:
                logger.warning(
                    f"Bundled vision model incomplete (missing: {', '.join(missing_files)}). "
                    f"Will download from HuggingFace instead."
                )
                return None

            logger.info(f"Found complete bundled vision model at: {model_path}")
            return model_path

        except Exception as e:
            logger.debug(f"Error checking for bundled vision model: {e}")
            return None

    def initialize(self):
        """
        Load Nomic Embed models.

        Checks for bundled models first, then falls back to auto-download from HuggingFace.
        - Text model: ~550MB (uses sentence-transformers)
        - Vision model: ~600MB (uses transformers Auto classes)
        - Multimodal: ~1.15GB (both models)

        Note: Vision models use AutoModel + AutoImageProcessor, not sentence-transformers,
        because they process images not text.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModel, AutoImageProcessor
            import torch

            if self.model_type == 'multimodal':
                logger.info(f"Loading Nomic Embed multimodal (text + vision) v1.5 on {self.device}...")

                # Load text model (sentence-transformers)
                text_bundled = self._get_bundled_text_model_path()
                if text_bundled:
                    logger.info(f"Using bundled text model from: {text_bundled}")
                    self.text_model = SentenceTransformer(str(text_bundled), trust_remote_code=True, device=self.device)
                else:
                    logger.info("Downloading text model from HuggingFace (~550MB)...")
                    self.text_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=self.device)

                logger.info("Text model loaded successfully")

                # Load vision model (transformers Auto classes - vision models don't use tokenizers)
                logger.info("Loading vision model using transformers Auto classes...")
                self.vision_processor = AutoImageProcessor.from_pretrained('nomic-ai/nomic-embed-vision-v1.5')
                self.vision_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-vision-v1.5', trust_remote_code=True)

                # Move to device
                if self.device == 'cuda':
                    self.vision_model = self.vision_model.cuda()

                self.vision_model.eval()  # Set to evaluation mode
                logger.info("Vision model loaded successfully")
                logger.info(f"Multimodal embedding ready on {self.device}")

            elif self.model_type == 'text':
                logger.info(f"Loading Nomic Embed text v1.5 on {self.device}...")
                text_bundled = self._get_bundled_text_model_path()

                if text_bundled:
                    self.text_model = SentenceTransformer(str(text_bundled), trust_remote_code=True, device=self.device)
                    logger.info("Bundled text model loaded successfully")
                else:
                    logger.info("Downloading text model from HuggingFace (~550MB)...")
                    self.text_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=self.device)
                    logger.info("Text model downloaded and loaded successfully")

            elif self.model_type == 'vision':
                logger.info(f"Loading Nomic Embed vision v1.5 on {self.device}...")
                # Vision models use transformers Auto classes (no tokenizer)
                self.vision_processor = AutoImageProcessor.from_pretrained('nomic-ai/nomic-embed-vision-v1.5')
                self.vision_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-vision-v1.5', trust_remote_code=True)

                # Move to device
                if self.device == 'cuda':
                    self.vision_model = self.vision_model.cuda()

                self.vision_model.eval()
                logger.info("Vision model loaded successfully")
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self._initialized = True
            logger.info(f"Nomic Embed {self.model_type} v1.5 ready on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize embedding service: {e}")

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate text embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default: 32)
            show_progress: Show progress bar during encoding

        Returns:
            numpy array of shape (len(texts), 768) with normalized embeddings

        Raises:
            RuntimeError: If text model is not initialized
        """
        if not self._initialized:
            self.initialize()

        if self.text_model is None:
            raise RuntimeError("Text model not loaded. Use model_type='text' or 'multimodal'")

        if not texts:
            logger.warning("Empty text list provided for embedding generation")
            return np.array([]).reshape(0, 768)

        try:
            # Encode texts with normalization for cosine similarity
            embeddings = self.text_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # For cosine similarity
            )

            logger.info(f"Generated {len(embeddings)} text embeddings with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}", exc_info=True)
            raise

    def generate_vision_embeddings(
        self,
        images: Union[List[np.ndarray], List[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate vision embeddings for a list of images.

        Args:
            images: List of images as numpy arrays or file paths
            batch_size: Batch size for processing (default: 32)
            show_progress: Show progress bar during encoding

        Returns:
            numpy array of shape (len(images), 768) with normalized embeddings

        Raises:
            RuntimeError: If vision model is not initialized
        """
        if not self._initialized:
            self.initialize()

        if self.vision_model is None:
            raise RuntimeError("Vision model not loaded. Use model_type='vision' or 'multimodal'")

        if not images:
            logger.warning("Empty image list provided for embedding generation")
            return np.array([]).reshape(0, 768)

        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            from tqdm import tqdm

            # Prepare images
            if isinstance(images[0], str):
                # Load images from file paths
                pil_images = [Image.open(img_path) for img_path in images]
            else:
                # Convert numpy arrays to PIL Images
                pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]

            all_embeddings = []

            # Process in batches
            num_batches = (len(pil_images) + batch_size - 1) // batch_size
            iterator = range(num_batches)

            if show_progress:
                iterator = tqdm(iterator, desc="Generating vision embeddings")

            with torch.no_grad():
                for batch_idx in iterator:
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(pil_images))
                    batch_images = pil_images[start_idx:end_idx]

                    # Preprocess images using vision processor
                    inputs = self.vision_processor(batch_images, return_tensors="pt")

                    # Move to device
                    if self.device == 'cuda':
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    # Get embeddings from model
                    outputs = self.vision_model(**inputs)

                    # Extract CLS token embeddings (first token)
                    img_emb = outputs.last_hidden_state[:, 0]

                    # Normalize for cosine similarity
                    img_embeddings = F.normalize(img_emb, p=2, dim=1)

                    # Convert to numpy
                    batch_embeddings = img_embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings)

            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)

            logger.info(f"Generated {len(embeddings)} vision embeddings with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating vision embeddings: {e}", exc_info=True)
            raise

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector (768-dim)
            embedding2: Second embedding vector (768-dim)

        Returns:
            Cosine similarity score between -1 and 1 (higher = more similar)
            For normalized embeddings, this is just the dot product.
        """
        # For normalized embeddings, cosine similarity = dot product
        return float(np.dot(embedding1, embedding2))

    def compute_similarity_matrix(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Array of shape (n, 768)

        Returns:
            Similarity matrix of shape (n, n) where element [i,j] is
            the cosine similarity between embeddings[i] and embeddings[j]
        """
        # Normalize if not already (defensive)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)  # Add epsilon to avoid division by zero

        # Compute dot product (cosine similarity for normalized vectors)
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix

    def cleanup(self):
        """
        Release model resources and free GPU/CPU memory.

        Important: Call this when done with embedding to free GPU memory
        before loading other models (e.g., OCR).
        """
        if self.text_model is not None:
            del self.text_model
            self.text_model = None

        if self.vision_model is not None:
            del self.vision_model
            self.vision_model = None

        import gc
        gc.collect()

        # If using CUDA, clear cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
        except ImportError:
            pass

        self._initialized = False
        logger.info("Embedding service cleaned up - GPU memory freed")


def generate_embeddings_for_document(
    doc_id: str,
    progress_callback=None
) -> bool:
    """
    Generate embeddings for all pages in a document.

    This function:
    1. Checks if embeddings already exist (skip if present)
    2. Loads all page texts from cache
    3. Generates embeddings in batches
    4. Saves embeddings to SQLite database

    Args:
        doc_id: Document ID (UUID string)
        progress_callback: Optional callback function(current, total, message)
            Called during processing to report progress

    Returns:
        True if successful, False otherwise

    Raises:
        RuntimeError: If embedding generation fails
    """
    from .cache_manager import get_cache_manager

    cache = get_cache_manager()

    # Check if already generated (or mostly complete)
    # Count existing embeddings to allow recovery from partial failures
    with cache.get_connection() as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM pages
            WHERE doc_id = ? AND text_embedding IS NOT NULL
        """, (doc_id,))
        existing_count = cursor.fetchone()[0]
        
        cursor = conn.execute(
            "SELECT total_pages FROM documents WHERE doc_id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        total_pages = result[0] if result else 0
    
    # If 80%+ embeddings exist, consider it complete (avoid reprocessing on partial failure)
    if total_pages > 0 and existing_count >= (total_pages * 0.8):
        completion_rate = (existing_count / total_pages) * 100
        logger.info(
            f"Embeddings {completion_rate:.1f}% complete for document {doc_id[:8]}... "
            f"({existing_count}/{total_pages}), considering complete"
        )
        if progress_callback:
            progress_callback(100, 100, f"Embeddings {completion_rate:.1f}% complete")
        return True

    try:
        # Load all page texts
        pages = cache.get_all_pages(doc_id)

        if not pages:
            logger.warning(f"No pages found for document {doc_id[:8]}...")
            return False

        texts = [page['text'] or '' for page in pages]
        total = len(texts)

        logger.info(f"Generating embeddings for {total} pages of document {doc_id[:8]}...")

        if progress_callback:
            progress_callback(0, total, "Loading embedding model...")

        # Initialize service on GPU (OCR is unloaded before this runs)
        service = EmbeddingService(device='cuda', model_type='text')
        service.initialize()

        if progress_callback:
            progress_callback(0, total, f"Generating embeddings for {total} pages...")

        # Generate embeddings in batches
        embeddings = service.generate_embeddings(
            texts,
            batch_size=32,
            show_progress=False  # We'll use our own progress callback
        )

        # Save to database
        for i, (page, embedding) in enumerate(zip(pages, embeddings)):
            cache.save_page_embedding(doc_id, page['page_num'], embedding)

            # Report progress every 50 pages
            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, total, f"Saved {i+1}/{total} embeddings")

        # Final progress update
        if progress_callback:
            progress_callback(total, total, "Embedding generation complete")

        # Cleanup
        service.cleanup()

        logger.info(f"Successfully generated {total} embeddings for document {doc_id[:8]}...")
        return True

    except Exception as e:
        logger.error(f"Failed to generate embeddings for document {doc_id[:8]}...: {e}", exc_info=True)
        raise RuntimeError(f"Embedding generation failed: {e}")
