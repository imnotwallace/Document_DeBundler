"""
Cache Manager for Document De-bundling
Manages persistent SQLite database for OCR results, embeddings, and split recommendations
"""

import sqlite3
import logging
import json
import uuid
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager
import numpy as np
import platform

logger = logging.getLogger(__name__)


def get_cache_db_path() -> Path:
    """
    Get the path to the cache database based on platform.

    Returns:
        Path to cache.db in user's application data directory
    """
    system = platform.system()

    if system == "Windows":
        base_path = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    elif system == "Darwin":  # macOS
        base_path = Path.home() / 'Library' / 'Application Support'
    else:  # Linux and others
        base_path = Path.home() / '.local' / 'share'

    cache_dir = base_path / 'DocumentDeBundler'
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / 'cache.db'


class CacheManager:
    """
    Manages the persistent SQLite cache for document processing.

    Responsibilities:
    - Database schema creation and versioning
    - CRUD operations for all tables
    - Cache size monitoring and cleanup
    - Checkpoint management for phase recovery
    """

    DB_VERSION = 1

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize cache manager.

        Args:
            db_path: Optional custom path to database (defaults to system app data)
        """
        self.db_path = db_path or get_cache_db_path()
        self._initialize_database()
        logger.info(f"Cache manager initialized: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def _initialize_database(self):
        """Create database schema if not exists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS _metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Check version
            cursor.execute("SELECT value FROM _metadata WHERE key = 'version'")
            result = cursor.fetchone()

            if not result:
                # First time setup
                self._create_schema(cursor)
                cursor.execute(
                    "INSERT INTO _metadata (key, value) VALUES ('version', ?)",
                    (str(self.DB_VERSION),)
                )
                logger.info(f"Database schema created (version {self.DB_VERSION})")
            else:
                version = int(result[0])
                if version < self.DB_VERSION:
                    self._upgrade_schema(cursor, version, self.DB_VERSION)

    def _create_schema(self, cursor):
        """Create all database tables"""

        # Main document tracking
        cursor.execute("""
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                total_pages INTEGER NOT NULL,
                file_size_bytes INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status TEXT DEFAULT 'ocr_pending',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # OCR results and embeddings
        cursor.execute("""
            CREATE TABLE pages (
                doc_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                text TEXT,
                text_length INTEGER,
                has_text_layer BOOLEAN,
                ocr_method TEXT,
                ocr_confidence FLOAT,
                text_embedding BLOB,
                features TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (doc_id, page_num),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)

        # Split candidates
        cursor.execute("""
            CREATE TABLE split_candidates (
                split_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                split_page INTEGER NOT NULL,
                confidence FLOAT,
                detection_method TEXT,
                reasoning TEXT,
                status TEXT DEFAULT 'pending',
                user_modified_page INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)

        # Document naming suggestions
        cursor.execute("""
            CREATE TABLE document_names (
                name_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                start_page INTEGER NOT NULL,
                end_page INTEGER NOT NULL,
                suggested_name TEXT,
                reasoning TEXT,
                confidence FLOAT,
                status TEXT DEFAULT 'pending',
                user_final_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)

        # Split execution results
        cursor.execute("""
            CREATE TABLE split_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                output_filename TEXT NOT NULL,
                output_path TEXT NOT NULL,
                start_page INTEGER,
                end_page INTEGER,
                page_count INTEGER,
                file_size_bytes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)

        # Processing log for checkpoints and debugging
        cursor.execute("""
            CREATE TABLE processing_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                phase TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                error_details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)

        # Create indices for performance
        cursor.execute("CREATE INDEX idx_pages_doc ON pages(doc_id)")
        cursor.execute("CREATE INDEX idx_splits_doc ON split_candidates(doc_id)")
        cursor.execute("CREATE INDEX idx_names_doc ON document_names(doc_id)")
        cursor.execute("CREATE INDEX idx_log_doc_phase ON processing_log(doc_id, phase)")

    def _upgrade_schema(self, cursor, from_version: int, to_version: int):
        """Upgrade database schema between versions"""
        logger.info(f"Upgrading database from v{from_version} to v{to_version}")
        # Future: Add migration logic here
        pass

    # ===== Document Management =====

    def create_document(
        self,
        file_path: str,
        total_pages: int,
        file_size_bytes: Optional[int] = None
    ) -> str:
        """
        Create a new document entry.

        Args:
            file_path: Path to PDF file
            total_pages: Number of pages
            file_size_bytes: File size in bytes

        Returns:
            Document ID (UUID)
        """
        doc_id = str(uuid.uuid4())
        filename = Path(file_path).name

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (doc_id, original_filename, file_path, total_pages, file_size_bytes)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, filename, file_path, total_pages, file_size_bytes))

        logger.info(f"Created document entry: {doc_id} ({filename})")
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_document_status(self, doc_id: str, status: str):
        """Update document processing status"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE documents
                SET processing_status = ?, last_updated = CURRENT_TIMESTAMP
                WHERE doc_id = ?
            """, (status, doc_id))

        logger.info(f"Document {doc_id[:8]}... status: {status}")

    def find_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Find existing document by file path"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE file_path = ? ORDER BY uploaded_at DESC LIMIT 1",
                (file_path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # ===== Page Management =====

    def save_page_text(
        self,
        doc_id: str,
        page_num: int,
        text: str,
        has_text_layer: bool,
        ocr_method: str,
        ocr_confidence: Optional[float] = None,
        features: Optional[Dict] = None
    ):
        """Save OCR text for a page"""
        features_json = json.dumps(features) if features else None

        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pages
                (doc_id, page_num, text, text_length, has_text_layer, ocr_method, ocr_confidence, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, page_num, text, len(text), has_text_layer,
                ocr_method, ocr_confidence, features_json
            ))

    def save_page_embedding(self, doc_id: str, page_num: int, embedding: np.ndarray):
        """Save page embedding vector"""
        # Convert numpy array to bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()

        with self.get_connection() as conn:
            conn.execute("""
                UPDATE pages
                SET text_embedding = ?
                WHERE doc_id = ? AND page_num = ?
            """, (embedding_bytes, doc_id, page_num))

    def get_page_text(self, doc_id: str, page_num: int) -> Optional[str]:
        """Get OCR text for a page"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT text FROM pages WHERE doc_id = ? AND page_num = ?",
                (doc_id, page_num)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_page_embedding(self, doc_id: str, page_num: int) -> Optional[np.ndarray]:
        """Get embedding for a page"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT text_embedding FROM pages WHERE doc_id = ? AND page_num = ?",
                (doc_id, page_num)
            )
            row = cursor.fetchone()

            if row and row[0]:
                # Convert bytes back to numpy array
                return np.frombuffer(row[0], dtype=np.float32)
            return None

    def get_all_pages(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all pages for a document"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM pages WHERE doc_id = ? ORDER BY page_num",
                (doc_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def has_embeddings(self, doc_id: str) -> bool:
        """Check if embeddings exist for document"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM pages
                WHERE doc_id = ? AND text_embedding IS NOT NULL
            """, (doc_id,))
            count = cursor.fetchone()[0]

            # Check if all pages have embeddings
            cursor = conn.execute(
                "SELECT total_pages FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            total = cursor.fetchone()[0]

            return count == total

    # ===== Split Management =====

    def save_split_candidate(
        self,
        doc_id: str,
        split_page: int,
        confidence: float,
        detection_method: str,
        reasoning: Dict[str, Any]
    ) -> int:
        """Save a split candidate"""
        reasoning_json = json.dumps(reasoning)

        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO split_candidates
                (doc_id, split_page, confidence, detection_method, reasoning)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, split_page, confidence, detection_method, reasoning_json))

            return cursor.lastrowid

    def get_split_candidates(self, doc_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get split candidates for a document"""
        with self.get_connection() as conn:
            if status:
                cursor = conn.execute("""
                    SELECT * FROM split_candidates
                    WHERE doc_id = ? AND status = ?
                    ORDER BY split_page
                """, (doc_id, status))
            else:
                cursor = conn.execute("""
                    SELECT * FROM split_candidates
                    WHERE doc_id = ?
                    ORDER BY split_page
                """, (doc_id,))

            results = []
            for row in cursor.fetchall():
                data = dict(row)
                # Parse JSON reasoning
                if data['reasoning']:
                    data['reasoning'] = json.loads(data['reasoning'])
                results.append(data)

            return results

    def update_split_status(self, split_id: int, status: str, user_modified_page: Optional[int] = None):
        """Update split candidate status"""
        with self.get_connection() as conn:
            if user_modified_page is not None:
                conn.execute("""
                    UPDATE split_candidates
                    SET status = ?, user_modified_page = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE split_id = ?
                """, (status, user_modified_page, split_id))
            else:
                conn.execute("""
                    UPDATE split_candidates
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE split_id = ?
                """, (status, split_id))

    # ===== Document Naming =====

    def save_document_name(
        self,
        doc_id: str,
        start_page: int,
        end_page: int,
        suggested_name: str,
        reasoning: str,
        confidence: float
    ) -> int:
        """Save a document naming suggestion"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO document_names
                (doc_id, start_page, end_page, suggested_name, reasoning, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, start_page, end_page, suggested_name, reasoning, confidence))

            return cursor.lastrowid

    def get_document_names(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get document naming suggestions"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM document_names
                WHERE doc_id = ?
                ORDER BY start_page
            """, (doc_id,))

            return [dict(row) for row in cursor.fetchall()]

    def update_document_name(self, name_id: int, user_final_name: str, status: str = 'confirmed'):
        """Update document name with user's choice"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE document_names
                SET user_final_name = ?, status = ?
                WHERE name_id = ?
            """, (user_final_name, status, name_id))

    # ===== Processing Log =====

    def log_phase(
        self,
        doc_id: str,
        phase: str,
        status: str,
        message: Optional[str] = None,
        error_details: Optional[str] = None
    ):
        """Log a processing phase"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO processing_log (doc_id, phase, status, message, error_details)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, phase, status, message, error_details))

    def get_last_completed_phase(self, doc_id: str) -> Optional[str]:
        """Get the last successfully completed phase for checkpoint recovery"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT phase FROM processing_log
                WHERE doc_id = ? AND status = 'completed'
                ORDER BY timestamp DESC
                LIMIT 1
            """, (doc_id,))

            row = cursor.fetchone()
            return row[0] if row else None

    # ===== Cache Management =====

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        # File size
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        size_mb = size_bytes / (1024 * 1024)

        with self.get_connection() as conn:
            # Document count
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]

            # Oldest entry
            cursor = conn.execute("SELECT MIN(uploaded_at) FROM documents")
            oldest = cursor.fetchone()[0]

            # Embedding storage
            cursor = conn.execute("""
                SELECT SUM(LENGTH(text_embedding)) FROM pages
                WHERE text_embedding IS NOT NULL
            """)
            embedding_bytes = cursor.fetchone()[0] or 0
            embedding_mb = embedding_bytes / (1024 * 1024)

            # Text storage
            cursor = conn.execute("SELECT SUM(LENGTH(text)) FROM pages")
            text_bytes = cursor.fetchone()[0] or 0
            text_mb = text_bytes / (1024 * 1024)

        return {
            'db_path': str(self.db_path),
            'total_size_mb': round(size_mb, 2),
            'embeddings_mb': round(embedding_mb, 2),
            'text_mb': round(text_mb, 2),
            'metadata_mb': round(size_mb - embedding_mb - text_mb, 2),
            'document_count': doc_count,
            'oldest_date': oldest
        }

    def cleanup_old_documents(self, days_old: int = 30) -> int:
        """
        Remove documents older than specified days.

        Args:
            days_old: Delete documents older than this many days

        Returns:
            Number of documents deleted
        """
        cutoff = datetime.now() - timedelta(days=days_old)

        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM documents
                WHERE uploaded_at < ?
            """, (cutoff,))

            deleted_count = cursor.rowcount

            # Vacuum to reclaim space
            conn.execute("VACUUM")

        logger.info(f"Cleaned up {deleted_count} documents older than {days_old} days")
        return deleted_count

    def clear_all_cache(self):
        """Clear entire cache database"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]

            conn.execute("DELETE FROM documents")
            conn.execute("VACUUM")

        logger.info(f"Cleared all cache ({count} documents)")
        return count

    def delete_document(self, doc_id: str):
        """Delete a specific document and all related data (CASCADE)"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

        logger.info(f"Deleted document: {doc_id[:8]}...")


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
