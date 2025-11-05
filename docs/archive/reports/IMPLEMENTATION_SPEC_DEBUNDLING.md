# Document De-bundling Implementation Specification

## Overview
This document provides detailed specifications for implementing the AI-powered document de-bundling feature. Components should be implemented independently following these specs.

## Architecture Context

**Completed:**
- ✅ `cache_manager.py` - SQLite database schema and utilities
- ✅ OCR optimization (4GB VRAM)
- ✅ VRAM monitoring system

**To Implement:**
1. `embedding_service.py` - Nomic Embed v1.5 integration
2. `split_detection.py` - Heuristic + clustering split detection
3. `llm/` module - LLM configuration, loading, and generation
4. `phase_coordinator.py` - Phase orchestration with checkpoints

---

## Component 1: embedding_service.py

**Location:** `python-backend/services/embedding_service.py`

**Dependencies:**
- `sentence-transformers` (Nomic Embed v1.5)
- `cache_manager.py` (already created)
- `numpy`

**Purpose:** Generate and manage page embeddings for semantic analysis

**Implementation Details:**

```python
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
            device: 'cpu' or 'cuda' (default: cpu during OCR)
        """
        self.device = device
        self.model = None
        self._initialized = False

    def initialize(self):
        """Load Nomic Embed model"""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5',
            trust_remote_code=True,
            device=self.device
        )
        self._initialized = True
        logger.info(f"Nomic Embed v1.5 loaded on {self.device}")

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(texts), 768)
        """
        if not self._initialized:
            self.initialize()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )

        return embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings"""
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
            Similarity matrix of shape (n, n)
        """
        # Normalize if not already
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Compute dot product (cosine similarity for normalized vectors)
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix

    def cleanup(self):
        """Release model resources"""
        if self.model is not None:
            del self.model
            self.model = None

        import gc
        gc.collect()

        self._initialized = False
        logger.info("Embedding service cleaned up")


def generate_embeddings_for_document(
    doc_id: str,
    progress_callback=None
) -> bool:
    """
    Generate embeddings for all pages in a document.

    Args:
        doc_id: Document ID
        progress_callback: Optional callback(current, total, message)

    Returns:
        True if successful
    """
    from .cache_manager import get_cache_manager

    cache = get_cache_manager()

    # Check if already generated
    if cache.has_embeddings(doc_id):
        logger.info(f"Embeddings already exist for {doc_id[:8]}...")
        return True

    # Load all page texts
    pages = cache.get_all_pages(doc_id)
    texts = [page['text'] or '' for page in pages]
    total = len(texts)

    if progress_callback:
        progress_callback(0, total, "Loading embedding model...")

    # Initialize service
    service = EmbeddingService(device='cpu')  # CPU to avoid GPU conflict
    service.initialize()

    if progress_callback:
        progress_callback(0, total, f"Generating embeddings for {total} pages...")

    # Generate embeddings in batches
    embeddings = service.generate_embeddings(
        texts,
        batch_size=32,
        show_progress=False
    )

    # Save to database
    for i, (page, embedding) in enumerate(zip(pages, embeddings)):
        cache.save_page_embedding(doc_id, page['page_num'], embedding)

        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, total, f"Saved {i+1}/{total} embeddings")

    service.cleanup()

    if progress_callback:
        progress_callback(total, total, "Embedding generation complete")

    logger.info(f"Generated {total} embeddings for document {doc_id[:8]}...")
    return True
```

**Testing:**
- Test embedding generation for sample document
- Verify similarity computation
- Ensure memory cleanup

---

## Component 2: split_detection.py

**Location:** `python-backend/services/split_detection.py`

**Dependencies:**
- `scikit-learn` (clustering)
- `numpy`
- `cache_manager.py`
- `embedding_service.py`

**Purpose:** Detect document boundaries using heuristics and clustering

**Implementation Details:**

```python
"""
Split Detection Service
Combines heuristic rules and clustering to detect document boundaries
"""

import logging
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import DBSCAN
from collections import Counter

logger = logging.getLogger(__name__)


class SplitDetector:
    """
    Detects document split points using multiple signals:
    - Heuristic rules (page numbers, headers, blank pages)
    - Semantic clustering (embeddings)
    - Layout analysis
    """

    # Confidence thresholds
    THRESHOLD_HIGH = 0.8  # Auto-accept
    THRESHOLD_MEDIUM = 0.5  # Mark for LLM review
    THRESHOLD_LOW = 0.3  # Ignore

    def __init__(self):
        self.signals = []

    def detect_page_number_reset(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        """
        Detect page number resets (1,2,3 → 1,2,3).

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        for i in range(1, len(pages)):
            curr_page = pages[i]
            prev_page = pages[i - 1]

            # Extract page numbers from text
            curr_num = self._extract_page_number(curr_page['text'])
            prev_num = self._extract_page_number(prev_page['text'])

            if curr_num and prev_num:
                # Check for reset (previous high, current low)
                if prev_num >= 2 and curr_num == 1:
                    splits.append((
                        i,
                        0.9,
                        f"Page number reset: {prev_num} → {curr_num}"
                    ))
                elif curr_num < prev_num - 5:  # Large backward jump
                    splits.append((
                        i,
                        0.7,
                        f"Page number jump backward: {prev_num} → {curr_num}"
                    ))

        return splits

    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from text"""
        if not text:
            return None

        # Look for common page number patterns
        patterns = [
            r'Page\s+(\d+)',
            r'p\.\s*(\d+)',
            r'^(\d+)$',  # Standalone number
            r'[-−]\s*(\d+)\s*[-−]',  # - 5 -
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue

        return None

    def detect_header_footer_changes(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        """
        Detect changes in header/footer patterns.

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        # Extract headers/footers for all pages
        headers = []
        footers = []

        for page in pages:
            text = page['text'] or ''
            lines = text.split('\n')

            if len(lines) >= 2:
                headers.append(lines[0].strip())
                footers.append(lines[-1].strip())
            else:
                headers.append('')
                footers.append('')

        # Detect pattern changes
        for i in range(1, len(pages)):
            header_changed = headers[i] != headers[i-1]
            footer_changed = footers[i] != footers[i-1]

            if header_changed and footer_changed:
                splits.append((
                    i,
                    0.6,
                    "Both header and footer changed"
                ))
            elif header_changed or footer_changed:
                splits.append((
                    i,
                    0.4,
                    "Header or footer changed"
                ))

        return splits

    def detect_blank_pages(self, pages: List[Dict]) -> List[Tuple[int, float, str]]:
        """
        Detect blank separator pages.

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        for i, page in enumerate(pages):
            text = page['text'] or ''
            text_length = len(text.strip())

            if text_length < 50:  # Very little text
                # Check if next page has content
                if i + 1 < len(pages):
                    next_text = pages[i + 1]['text'] or ''
                    if len(next_text.strip()) > 100:
                        splits.append((
                            i + 1,  # Split AFTER blank page
                            0.85,
                            f"Blank separator page (only {text_length} chars)"
                        ))

        return splits

    def detect_semantic_discontinuity(
        self,
        doc_id: str,
        embeddings: np.ndarray
    ) -> List[Tuple[int, float, str]]:
        """
        Detect semantic discontinuity using embedding similarity.

        Args:
            doc_id: Document ID
            embeddings: Array of page embeddings (n, 768)

        Returns:
            List of (page_num, confidence, reason)
        """
        splits = []

        # Compute consecutive page similarities
        for i in range(1, len(embeddings)):
            similarity = float(np.dot(embeddings[i-1], embeddings[i]))

            # Low similarity = likely different documents
            if similarity < 0.3:
                splits.append((
                    i,
                    0.7,
                    f"Low semantic similarity ({similarity:.2f})"
                ))
            elif similarity < 0.5:
                splits.append((
                    i,
                    0.5,
                    f"Moderate semantic discontinuity ({similarity:.2f})"
                ))

        return splits

    def detect_with_clustering(
        self,
        embeddings: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 3
    ) -> List[Tuple[int, float, str]]:
        """
        Use DBSCAN clustering to detect document boundaries.

        Args:
            embeddings: Page embeddings
            eps: DBSCAN epsilon parameter
            min_samples: Minimum cluster size

        Returns:
            List of (page_num, confidence, reason)
        """
        if len(embeddings) < min_samples:
            return []

        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        splits = []

        # Detect cluster boundaries
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                # Cluster changed = potential split
                splits.append((
                    i,
                    0.6,
                    f"Cluster boundary (cluster {labels[i-1]} → {labels[i]})"
                ))

        return splits

    def combine_signals(
        self,
        all_splits: List[List[Tuple[int, float, str]]]
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple detection signals into final split candidates.

        Args:
            all_splits: List of split lists from different detectors

        Returns:
            List of split candidates with aggregated confidence
        """
        # Aggregate by page number
        page_scores = {}
        page_reasons = {}

        for split_list in all_splits:
            for page_num, confidence, reason in split_list:
                if page_num not in page_scores:
                    page_scores[page_num] = []
                    page_reasons[page_num] = []

                page_scores[page_num].append(confidence)
                page_reasons[page_num].append(reason)

        # Compute final candidates
        candidates = []

        for page_num in sorted(page_scores.keys()):
            scores = page_scores[page_num]
            reasons = page_reasons[page_num]

            # Aggregate confidence (max + bonus for multiple signals)
            max_conf = max(scores)
            bonus = min(0.2, (len(scores) - 1) * 0.1)  # Up to 0.2 bonus
            final_conf = min(1.0, max_conf + bonus)

            if final_conf >= self.THRESHOLD_LOW:
                candidates.append({
                    'page': page_num,
                    'confidence': final_conf,
                    'signals': reasons,
                    'signal_count': len(reasons)
                })

        return candidates


def detect_splits_for_document(
    doc_id: str,
    progress_callback=None
) -> int:
    """
    Detect split points for a document.

    Args:
        doc_id: Document ID
        progress_callback: Optional callback(current, total, message)

    Returns:
        Number of splits detected
    """
    from .cache_manager import get_cache_manager

    cache = get_cache_manager()
    detector = SplitDetector()

    if progress_callback:
        progress_callback(0, 5, "Loading document data...")

    # Load pages and embeddings
    pages = cache.get_all_pages(doc_id)

    if not pages:
        logger.warning(f"No pages found for document {doc_id[:8]}...")
        return 0

    # Get embeddings
    embeddings = []
    for page in pages:
        emb = cache.get_page_embedding(doc_id, page['page_num'])
        if emb is not None:
            embeddings.append(emb)
        else:
            logger.warning(f"Missing embedding for page {page['page_num']}")
            embeddings.append(np.zeros(768))  # Placeholder

    embeddings = np.array(embeddings)

    if progress_callback:
        progress_callback(1, 5, "Running heuristic detection...")

    # Run all detectors
    all_splits = []

    # Heuristic detectors
    all_splits.append(detector.detect_page_number_reset(pages))
    all_splits.append(detector.detect_header_footer_changes(pages))
    all_splits.append(detector.detect_blank_pages(pages))

    if progress_callback:
        progress_callback(2, 5, "Analyzing semantic similarity...")

    # Embedding-based detectors
    all_splits.append(detector.detect_semantic_discontinuity(doc_id, embeddings))

    if progress_callback:
        progress_callback(3, 5, "Running clustering analysis...")

    all_splits.append(detector.detect_with_clustering(embeddings))

    if progress_callback:
        progress_callback(4, 5, "Combining signals...")

    # Combine signals
    candidates = detector.combine_signals(all_splits)

    # Save to database
    for candidate in candidates:
        method = 'heuristic' if candidate['confidence'] >= 0.8 else 'clustering'

        cache.save_split_candidate(
            doc_id=doc_id,
            split_page=candidate['page'],
            confidence=candidate['confidence'],
            detection_method=method,
            reasoning={
                'signals': candidate['signals'],
                'signal_count': candidate['signal_count']
            }
        )

    if progress_callback:
        progress_callback(5, 5, f"Detected {len(candidates)} potential splits")

    logger.info(f"Detected {len(candidates)} splits for document {doc_id[:8]}...")
    return len(candidates)
```

**Testing:**
- Test page number reset detection
- Test clustering on sample embeddings
- Verify confidence scoring

---

## Component 3: llm/ Module

**Location:** `python-backend/services/llm/`

**Files to Create:**
1. `__init__.py`
2. `config.py` - VRAM detection and model selection
3. `loader.py` - LLM loading with memory monitoring
4. `prompts.py` - Prompt templates
5. `split_analyzer.py` - Split refinement
6. `name_generator.py` - Filename generation

### 3.1: llm/config.py

```python
"""
LLM Configuration with VRAM-based Model Selection
Optimized for 4GB VRAM target hardware
"""

import logging
from typing import Dict, Any, Optional
from services.ocr.config import detect_hardware_capabilities

logger = logging.getLogger(__name__)


def select_optimal_llm_config(gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
    """
    Automatically select best LLM configuration for available VRAM.

    Args:
        gpu_memory_gb: Manual override for GPU memory (auto-detected if None)

    Returns:
        LLM configuration dictionary
    """
    if gpu_memory_gb is None:
        capabilities = detect_hardware_capabilities()
        gpu_memory_gb = capabilities['gpu_memory_gb']

    logger.info(f"Detecting optimal LLM for {gpu_memory_gb:.1f}GB VRAM")

    if gpu_memory_gb >= 8:
        # High-end GPU: Better quality
        return {
            'model_name': 'Phi-3 Mini (High Quality)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q5_k_m.gguf',
            'quantization': 'Q5_K_M',
            'n_gpu_layers': 32,
            'n_batch': 512,
            'n_ctx': 4096,
            'expected_vram_gb': 3.2,
            'offload_strategy': 'gpu_only'
        }
    elif gpu_memory_gb >= 6:
        # Mid-range GPU
        return {
            'model_name': 'Phi-3 Mini (Balanced)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 32,
            'n_batch': 512,
            'n_ctx': 4096,
            'expected_vram_gb': 2.5,
            'offload_strategy': 'gpu_only'
        }
    elif gpu_memory_gb >= 4:
        # TARGET: 4GB VRAM
        return {
            'model_name': 'Phi-3 Mini (4GB Optimized)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 28,  # 28 GPU + 4 CPU layers
            'n_batch': 256,
            'n_ctx': 4096,
            'expected_vram_gb': 2.3,
            'offload_strategy': 'hybrid',
            'cpu_layers': 4
        }
    elif gpu_memory_gb >= 2:
        # Low VRAM: Lighter model
        return {
            'model_name': 'Gemma 2 2B',
            'model_id': 'google/gemma-2-2b-it-gguf',
            'model_file': 'gemma-2-2b-it-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 20,
            'n_batch': 256,
            'n_ctx': 4096,
            'expected_vram_gb': 1.5,
            'offload_strategy': 'hybrid',
            'cpu_layers': 8
        }
    else:
        # CPU-only fallback
        return {
            'model_name': 'Phi-3 Mini (CPU)',
            'model_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'model_file': 'Phi-3-mini-4k-instruct-q4_k_m.gguf',
            'quantization': 'Q4_K_M',
            'n_gpu_layers': 0,
            'n_batch': 128,
            'n_ctx': 4096,
            'expected_vram_gb': 0,
            'offload_strategy': 'cpu_only',
            'warning': 'LLM running on CPU - will be slow (5-10x slower)'
        }


def get_model_download_info(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get model download information.

    Returns:
        Dictionary with download URLs and paths
    """
    # HuggingFace download URLs
    base_url = f"https://huggingface.co/{config['model_id']}/resolve/main"

    return {
        'model_url': f"{base_url}/{config['model_file']}",
        'model_file': config['model_file'],
        'expected_size_gb': config['expected_vram_gb']
    }
```

### 3.2: llm/prompts.py

```python
"""
Prompt Templates for LLM-based De-bundling
"""


SPLIT_REFINEMENT_PROMPT = """You are analyzing a potential document boundary in a bundled PDF.

Context: Pages {start_page}-{end_page} are being considered for a split at page {split_page}.

Pages BEFORE split point (pages {before_start}-{before_end}):
{before_text}

Pages AFTER split point (pages {after_start}-{after_end}):
{after_text}

Heuristic signals detected:
{heuristic_signals}

Question: Should there be a document boundary at page {split_page}?

Consider:
- Page numbering patterns
- Header/footer consistency
- Content topic/subject
- Document structure
- Semantic continuity

Answer with ONLY "YES" or "NO", followed by a brief reason (one sentence).

Example: "YES - Page numbering resets and header changes indicate new document"
Example: "NO - Content continues same topic with consistent formatting"

Answer:"""


DOCUMENT_NAMING_PROMPT = """You are generating a filename for a document extracted from a bundled PDF.

Document Segment: Pages {start_page} to {end_page}

First page content:
{first_page_text}

Second page content (if available):
{second_page_text}

Task: Generate a filename in this EXACT format:
{DATE}_{DOCTYPE}_{DESCRIPTION}

Instructions:

1. DATE (YYYY-MM-DD):
   - Find the DOCUMENT DATE (when written/issued), NOT today's date
   - Look for: "Date:", "Dated:", "Issued on:", "Effective Date:", "As of:"
   - If multiple dates, use the earliest/primary date
   - If no date found, use: "UNDATED"

2. DOCTYPE (one word):
   - Choose ONE type: Invoice, Contract, Agreement, Letter, Report, Receipt,
     Form, Certificate, Statement, Memo, Proposal, Notice, Policy, Other
   - Be specific (e.g., "Agreement" not "Document")

3. DESCRIPTION (2-5 words, no special characters):
   - Include: parties, subject matter, identifiers
   - Examples: "Acme Corp Service Agreement", "Q4 Financial Report", "Employee John Smith"
   - Avoid: generic words like "document", "pdf", articles like "the", "a"
   - Keep concise but descriptive

Output ONLY the filename (no quotes, no .pdf extension, no explanation):

Examples:
2024-12-01_Letter_Employment Offer John Smith
2023-06-15_Invoice_Acme Corp June Services
UNDATED_Report_Annual Financial Summary
2024-03-20_Contract_Software License Agreement

Filename:"""


SPLIT_REASONING_SYSTEM_PROMPT = """You are an expert document analyst specializing in identifying document boundaries in bundled PDFs. You analyze page content, structure, and metadata to make accurate split decisions."""


NAMING_SYSTEM_PROMPT = """You are a document classification expert. You extract metadata from documents and generate descriptive, standardized filenames following specific naming conventions."""


def format_split_prompt(
    split_page: int,
    before_pages: list,
    after_pages: list,
    heuristic_signals: list
) -> str:
    """Format split refinement prompt with context"""

    before_text = "\n---\n".join([
        f"Page {p['page_num']}: {p['text'][:300]}..."
        for p in before_pages[-3:]  # Last 3 pages before split
    ])

    after_text = "\n---\n".join([
        f"Page {p['page_num']}: {p['text'][:300]}..."
        for p in after_pages[:3]  # First 3 pages after split
    ])

    signals_text = "\n- ".join(heuristic_signals)

    return SPLIT_REFINEMENT_PROMPT.format(
        start_page=before_pages[0]['page_num'] if before_pages else 0,
        end_page=after_pages[-1]['page_num'] if after_pages else 0,
        split_page=split_page,
        before_start=before_pages[-3]['page_num'] if len(before_pages) >= 3 else before_pages[0]['page_num'],
        before_end=before_pages[-1]['page_num'],
        after_start=after_pages[0]['page_num'],
        after_end=after_pages[2]['page_num'] if len(after_pages) >= 3 else after_pages[-1]['page_num'],
        before_text=before_text,
        after_text=after_text,
        heuristic_signals=signals_text
    )


def format_naming_prompt(
    start_page: int,
    end_page: int,
    first_page_text: str,
    second_page_text: str = ""
) -> str:
    """Format document naming prompt"""

    return DOCUMENT_NAMING_PROMPT.format(
        start_page=start_page,
        end_page=end_page,
        first_page_text=first_page_text[:2000],  # Limit context
        second_page_text=second_page_text[:1000] if second_page_text else "(Not available)"
    )
```

---

## Component 4: phase_coordinator.py

**Location:** `python-backend/services/phase_coordinator.py`

**Purpose:** Orchestrate the de-bundling pipeline with checkpoint recovery

**Key Features:**
- Phase sequencing
- Checkpoint management
- Error recovery
- Progress reporting
- Memory cleanup between phases

**Basic Structure:**

```python
"""
Phase Coordinator for Document De-bundling Pipeline
Orchestrates the multi-phase de-bundling process with checkpoints
"""

import logging
import gc
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class Phase(Enum):
    """De-bundling pipeline phases"""
    OCR = "ocr"
    EMBEDDING = "embedding"
    DETECTION = "detection"
    LLM_REFINE = "llm_refine"
    LLM_NAMING = "llm_naming"
    REVIEW = "review"
    EXECUTE = "execute"


class PhaseCoordinator:
    """
    Coordinates the de-bundling pipeline.

    Features:
    - Sequential phase execution
    - Checkpoint recovery
    - Memory management
    - Progress callbacks
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.current_phase = None

    def run_pipeline(
        self,
        skip_llm_refine: bool = False,
        progress_callback: Optional[Callable] = None
    ):
        """
        Run complete de-bundling pipeline.

        Args:
            skip_llm_refine: Skip LLM refinement (use heuristics only)
            progress_callback: Optional callback(phase, status, message)
        """
        from .cache_manager import get_cache_manager
        cache = get_cache_manager()

        # Check for existing progress
        last_phase = cache.get_last_completed_phase(self.doc_id)

        # Define pipeline
        phases = [
            (Phase.OCR, self._phase_ocr),
            (Phase.EMBEDDING, self._phase_embedding),
            (Phase.DETECTION, self._phase_detection),
        ]

        if not skip_llm_refine:
            phases.append((Phase.LLM_REFINE, self._phase_llm_refine))

        phases.extend([
            (Phase.LLM_NAMING, self._phase_llm_naming),
            # REVIEW and EXECUTE handled by UI
        ])

        # Execute phases
        for phase_enum, phase_func in phases:
            if last_phase and self._phase_completed(last_phase, phase_enum):
                logger.info(f"Skipping {phase_enum.value} (already completed)")
                continue

            try:
                self.current_phase = phase_enum
                cache.log_phase(self.doc_id, phase_enum.value, 'started')

                if progress_callback:
                    progress_callback(phase_enum.value, 'started', f"Starting {phase_enum.value}")

                # Execute phase
                phase_func(progress_callback)

                # Cleanup memory
                self._cleanup_memory()

                cache.log_phase(self.doc_id, phase_enum.value, 'completed')

                if progress_callback:
                    progress_callback(phase_enum.value, 'completed', f"Completed {phase_enum.value}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Phase {phase_enum.value} failed: {error_msg}", exc_info=True)
                cache.log_phase(self.doc_id, phase_enum.value, 'failed', error_details=error_msg)

                if progress_callback:
                    progress_callback(phase_enum.value, 'failed', error_msg)

                raise

    def _phase_ocr(self, progress_callback):
        """Phase 0: OCR (if needed)"""
        from .cache_manager import get_cache_manager
        cache = get_cache_manager()

        # Check if OCR already done
        pages = cache.get_all_pages(self.doc_id)
        if pages:
            logger.info("OCR already completed")
            return

        # Run OCR (integrate with existing OCR service)
        # TODO: Call existing OCR functionality
        pass

    def _phase_embedding(self, progress_callback):
        """Phase 1: Generate embeddings"""
        from .embedding_service import generate_embeddings_for_document
        generate_embeddings_for_document(self.doc_id, progress_callback)

    def _phase_detection(self, progress_callback):
        """Phase 2: Detect splits"""
        from .split_detection import detect_splits_for_document
        detect_splits_for_document(self.doc_id, progress_callback)

    def _phase_llm_refine(self, progress_callback):
        """Phase 3: LLM refinement (optional)"""
        # TODO: Implement LLM split refinement
        pass

    def _phase_llm_naming(self, progress_callback):
        """Phase 4: Generate names"""
        # TODO: Implement LLM naming
        pass

    def _cleanup_memory(self):
        """Clean memory between phases"""
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _phase_completed(self, last_phase: str, current_phase: Phase) -> bool:
        """Check if current phase was already completed"""
        phase_order = [p.value for p in Phase]

        try:
            last_idx = phase_order.index(last_phase)
            current_idx = phase_order.index(current_phase.value)
            return current_idx <= last_idx
        except ValueError:
            return False
```

---

## Dependencies to Add

**File:** `python-backend/requirements.txt`

Add:
```
# Document De-bundling
sentence-transformers>=2.2.2  # Nomic Embed
scikit-learn>=1.3.0  # Clustering
llama-cpp-python>=0.2.0  # LLM inference
huggingface-hub>=0.19.0  # Model downloads
```

---

## Testing Strategy

Create `python-backend/tests/test_debundling.py`:

```python
"""
Integration tests for de-bundling pipeline
"""

import pytest
from services.cache_manager import CacheManager
from services.embedding_service import EmbeddingService
from services.split_detection import SplitDetector


def test_cache_creation(tmp_path):
    """Test cache database creation"""
    cache = CacheManager(db_path=tmp_path / "test.db")

    # Create test document
    doc_id = cache.create_document(
        file_path="/test/sample.pdf",
        total_pages=10,
        file_size_bytes=1024000
    )

    # Verify creation
    doc = cache.get_document(doc_id)
    assert doc['total_pages'] == 10


def test_embedding_generation():
    """Test embedding service"""
    service = EmbeddingService(device='cpu')
    service.initialize()

    texts = ["This is page 1", "This is page 2", "This is page 3"]
    embeddings = service.generate_embeddings(texts, show_progress=False)

    assert embeddings.shape == (3, 768)

    # Test similarity
    sim = service.compute_similarity(embeddings[0], embeddings[1])
    assert 0 <= sim <= 1

    service.cleanup()


def test_split_detection():
    """Test split detection"""
    detector = SplitDetector()

    # Mock pages with page number reset
    pages = [
        {'page_num': 0, 'text': 'Page 1\nSome content'},
        {'page_num': 1, 'text': 'Page 2\nMore content'},
        {'page_num': 2, 'text': 'Page 1\nNew document'},  # Reset!
    ]

    splits = detector.detect_page_number_reset(pages)

    assert len(splits) == 1
    assert splits[0][0] == 2  # Split at page 2
    assert splits[0][1] > 0.8  # High confidence
```

---

## Integration Points

**With Existing Code:**

1. **OCR Integration** - `phase_coordinator.py` should call existing OCR service
2. **UI Components** - Frontend needs to display splits and names
3. **File Operations** - Extend existing `bundler.py` for splitting

**Rust Commands to Add:**

```rust
// src-tauri/src/commands.rs

#[tauri::command]
pub fn start_debundling(
    file_path: String,
    skip_llm: bool
) -> Result<String, String> {
    // Call Python de-bundling pipeline
    // Return doc_id for tracking
}

#[tauri::command]
pub fn get_split_candidates(doc_id: String) -> Result<Vec<SplitCandidate>, String> {
    // Query SQLite for split candidates
}

#[tauri::command]
pub fn confirm_splits(
    doc_id: String,
    confirmed_splits: Vec<ConfirmedSplit>
) -> Result<(), String> {
    // Update database with user confirmations
    // Execute split operation
}
```

---

## Summary

This specification provides complete implementation details for:
1. ✅ Embedding generation (Nomic Embed v1.5)
2. ✅ Split detection (Heuristics + Clustering)
3. ✅ LLM configuration (VRAM-optimized)
4. ✅ Phase coordination (Checkpoint recovery)

Each component can be implemented independently following these specs.
