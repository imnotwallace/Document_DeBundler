"""
Simple test script for embedding_service.py
Tests basic functionality before full integration
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_embedding_service():
    """Test basic embedding service functionality"""
    from services.embedding_service import EmbeddingService
    import numpy as np

    print("\n" + "="*60)
    print("Testing EmbeddingService")
    print("="*60)

    # Test 1: Initialization
    print("\nTest 1: Initializing service...")
    service = EmbeddingService(device='cpu')

    try:
        service.initialize()
        print("✓ Service initialized successfully")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

    # Test 2: Generate embeddings
    print("\nTest 2: Generating embeddings...")
    test_texts = [
        "This is the first page of a document about contracts.",
        "This page continues the contract discussion.",
        "This is a completely different document about invoices.",
        "Invoice number 12345 dated January 1, 2024."
    ]

    try:
        embeddings = service.generate_embeddings(
            test_texts,
            batch_size=2,
            show_progress=False
        )

        print(f"✓ Generated embeddings with shape: {embeddings.shape}")

        # Verify shape
        expected_shape = (len(test_texts), 768)
        if embeddings.shape == expected_shape:
            print(f"✓ Shape is correct: {expected_shape}")
        else:
            print(f"✗ Shape mismatch: expected {expected_shape}, got {embeddings.shape}")
            return False

    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False

    # Test 3: Compute similarity
    print("\nTest 3: Computing similarity...")
    try:
        # Similar texts (both about contracts)
        sim_similar = service.compute_similarity(embeddings[0], embeddings[1])
        print(f"  Similarity (contract pages): {sim_similar:.4f}")

        # Dissimilar texts (contract vs invoice)
        sim_different = service.compute_similarity(embeddings[0], embeddings[2])
        print(f"  Similarity (contract vs invoice): {sim_different:.4f}")

        # Invoice pages should be similar to each other
        sim_invoice = service.compute_similarity(embeddings[2], embeddings[3])
        print(f"  Similarity (invoice pages): {sim_invoice:.4f}")

        # Validate
        if sim_similar > sim_different:
            print("✓ Similar texts have higher similarity (expected)")
        else:
            print("✗ Similarity ordering incorrect")
            return False

    except Exception as e:
        print(f"✗ Similarity computation failed: {e}")
        return False

    # Test 4: Compute similarity matrix
    print("\nTest 4: Computing similarity matrix...")
    try:
        sim_matrix = service.compute_similarity_matrix(embeddings)

        print(f"✓ Similarity matrix shape: {sim_matrix.shape}")

        # Verify shape
        expected_matrix_shape = (len(test_texts), len(test_texts))
        if sim_matrix.shape == expected_matrix_shape:
            print(f"✓ Matrix shape is correct: {expected_matrix_shape}")
        else:
            print(f"✗ Matrix shape mismatch: expected {expected_matrix_shape}, got {sim_matrix.shape}")
            return False

        # Diagonal should be 1.0 (self-similarity)
        diagonal_values = np.diag(sim_matrix)
        if np.allclose(diagonal_values, 1.0, atol=0.01):
            print("✓ Diagonal values are ~1.0 (self-similarity)")
        else:
            print(f"✗ Diagonal values incorrect: {diagonal_values}")
            return False

    except Exception as e:
        print(f"✗ Similarity matrix computation failed: {e}")
        return False

    # Test 5: Cleanup
    print("\nTest 5: Cleanup...")
    try:
        service.cleanup()

        if service.model is None and not service._initialized:
            print("✓ Service cleaned up successfully")
        else:
            print("✗ Cleanup incomplete")
            return False

    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        return False

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60 + "\n")
    return True


def test_with_cache_manager():
    """Test embedding generation with cache manager integration"""
    from services.embedding_service import generate_embeddings_for_document
    from services.cache_manager import CacheManager
    import tempfile
    import os

    print("\n" + "="*60)
    print("Testing Cache Manager Integration")
    print("="*60)

    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        db_path = Path(tmp.name)

    try:
        print(f"\nCreating test database at: {db_path}")
        cache = CacheManager(db_path=db_path)

        # Create test document
        print("\nCreating test document...")
        doc_id = cache.create_document(
            file_path="/test/sample.pdf",
            total_pages=3,
            file_size_bytes=1024000
        )
        print(f"✓ Created document: {doc_id[:8]}...")

        # Save some test pages
        print("\nSaving test pages...")
        test_pages = [
            "This is page 1 about employment contracts.",
            "This is page 2 continuing the employment contract.",
            "This is page 3 with invoice information."
        ]

        for i, text in enumerate(test_pages):
            cache.save_page_text(
                doc_id=doc_id,
                page_num=i,
                text=text,
                has_text_layer=True,
                ocr_method='direct',
                ocr_confidence=1.0
            )
        print(f"✓ Saved {len(test_pages)} pages")

        # Progress callback
        def progress(current, total, message):
            print(f"  Progress: {current}/{total} - {message}")

        # Generate embeddings
        print("\nGenerating embeddings...")
        success = generate_embeddings_for_document(
            doc_id=doc_id,
            progress_callback=progress
        )

        if success:
            print("✓ Embedding generation successful")
        else:
            print("✗ Embedding generation failed")
            return False

        # Verify embeddings were saved
        print("\nVerifying saved embeddings...")
        for i in range(len(test_pages)):
            embedding = cache.get_page_embedding(doc_id, i)
            if embedding is not None:
                print(f"  Page {i}: embedding shape = {embedding.shape}")
                if embedding.shape != (768,):
                    print(f"✗ Incorrect shape: expected (768,), got {embedding.shape}")
                    return False
            else:
                print(f"✗ No embedding found for page {i}")
                return False

        print("✓ All embeddings verified")

        # Test idempotency (should skip if already exists)
        print("\nTesting idempotency (re-run should skip)...")
        success = generate_embeddings_for_document(
            doc_id=doc_id,
            progress_callback=progress
        )

        if success:
            print("✓ Idempotency test passed")
        else:
            print("✗ Idempotency test failed")
            return False

        print("\n" + "="*60)
        print("CACHE INTEGRATION TEST PASSED")
        print("="*60 + "\n")
        return True

    finally:
        # Cleanup
        if db_path.exists():
            os.unlink(db_path)
            print(f"Cleaned up test database: {db_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMBEDDING SERVICE TEST SUITE")
    print("="*60)
    print("\nNOTE: This will download Nomic Embed v1.5 model (~768MB)")
    print("      on first run. Subsequent runs will use cached model.")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")

    try:
        import time
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(0)

    # Run tests
    test1_passed = test_embedding_service()

    if test1_passed:
        test2_passed = test_with_cache_manager()

        if test2_passed:
            print("\n" + "="*60)
            print("ALL TESTS PASSED SUCCESSFULLY")
            print("="*60 + "\n")
            sys.exit(0)
        else:
            print("\n✗ Cache integration test failed")
            sys.exit(1)
    else:
        print("\n✗ Basic service test failed")
        sys.exit(1)
