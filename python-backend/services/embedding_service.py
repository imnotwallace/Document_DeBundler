"""
Embedding Service for Document De-bundling
Generates semantic embeddings using Nomic Embed v1.5
"""

import logging
import numpy as np
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages embedding generation using Nomic Embed v1.5.

    Features:
    - Text embeddings (768-dim)
    - Batch processing on CPU
    - Progress callbacks
    - Memory-efficient processing
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize embedding service.

        Args:
            device: 'cpu' or 'cuda' (default: cpu to avoid GPU conflict with OCR)
        """
        self.device = device
        self.model = None
        self._initialized = False

    def initialize(self):
        """
        Load Nomic Embed model.

        The model is auto-downloaded on first run (~768MB).
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading Nomic Embed v1.5 on {self.device}...")

            self.model = SentenceTransformer(
                'nomic-ai/nomic-embed-text-v1.5',
                trust_remote_code=True,
                device=self.device
            )
            self._initialized = True
            logger.info(f"Nomic Embed v1.5 loaded successfully on {self.device}")

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
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default: 32)
            show_progress: Show progress bar during encoding

        Returns:
            numpy array of shape (len(texts), 768) with normalized embeddings

        Raises:
            RuntimeError: If model is not initialized
        """
        if not self._initialized:
            self.initialize()

        if not texts:
            logger.warning("Empty text list provided for embedding generation")
            return np.array([]).reshape(0, 768)

        try:
            # Encode texts with normalization for cosine similarity
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # For cosine similarity
            )

            logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
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
        Release model resources and free memory.

        Call this when done with the embedding service to free GPU/CPU memory.
        """
        if self.model is not None:
            del self.model
            self.model = None

        import gc
        gc.collect()

        # If using CUDA, clear cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._initialized = False
        logger.info("Embedding service cleaned up")


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

        # Initialize service on CPU to avoid GPU conflict with OCR
        service = EmbeddingService(device='cpu')
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
