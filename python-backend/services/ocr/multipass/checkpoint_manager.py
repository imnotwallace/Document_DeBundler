"""
Checkpoint Manager for Multi-Pass Pipeline.

Provides thread-safe per-page JSON checkpointing for resume capability.
Stores checkpoint state in temp/checkpoint.json and per-page results
in temp/{ocr_results,layout_results,reading_order,pdf_pages}/ directories.
"""

import json
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from .data_structures import PassStatus, PageCheckpoint

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Thread-safe checkpoint manager for multi-pass pipeline recovery.

    Maintains:
    - Global checkpoint.json with pass-level status and page lists
    - Per-page JSON files for each pass output

    Resume Strategy:
    - On restart, load checkpoint.json
    - For each pass, skip already-completed pages
    - On OOM, record fallback and continue
    """

    VERSION = "1.0"

    def __init__(self, workspace_dir: Path, doc_id: str):
        """
        Initialize checkpoint manager.

        Args:
            workspace_dir: Working directory for temp files
            doc_id: Unique document identifier
        """
        self.workspace_dir = Path(workspace_dir)
        self.doc_id = doc_id
        self.lock = threading.Lock()

        # Ensure temp directory exists
        self.temp_dir = self.workspace_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.temp_dir / "checkpoint.json"

        # Load or create checkpoint
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing checkpoint or create new one."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")

                # Validate version
                if self.data.get("version") != self.VERSION:
                    logger.warning(
                        f"Checkpoint version mismatch: {self.data.get('version')} != {self.VERSION}"
                    )
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}, creating new")
                self.data = self._create_new()
                self._save()
        else:
            self.data = self._create_new()
            self._save()
            logger.info("Created new checkpoint")

    def _create_new(self) -> Dict[str, Any]:
        """Create new checkpoint structure."""
        return {
            "version": self.VERSION,
            "doc_id": self.doc_id,
            "workspace": str(self.workspace_dir),
            "total_pages": 0,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "passes": {
                "pass1_paddleocr": {
                    "status": PassStatus.NOT_STARTED.value,
                    "completed_pages": [],
                    "failed_pages": [],
                },
                "pass2_layout": {
                    "status": PassStatus.NOT_STARTED.value,
                    "completed_pages": [],
                    "failed_pages": [],
                    "current_batch_size": 4,
                    "fallback_triggered": False,
                },
                "pass3_reading_order": {
                    "status": PassStatus.NOT_STARTED.value,
                    "completed_pages": [],
                    "failed_pages": [],
                },
                "pass4_pdf_creation": {
                    "status": PassStatus.NOT_STARTED.value,
                    "completed_pages": [],
                    "failed_pages": [],
                },
                "pass5_pdf_merge": {
                    "status": PassStatus.NOT_STARTED.value,
                },
            },
            "errors": [],
        }

    def _save(self) -> None:
        """Save checkpoint to disk (must hold lock)."""
        self.data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def set_total_pages(self, total_pages: int) -> None:
        """Set total page count."""
        with self.lock:
            self.data["total_pages"] = total_pages
            self._save()

    def mark_page_completed(self, pass_name: str, page_num: int) -> None:
        """
        Mark a page as successfully processed for a pass.

        Args:
            pass_name: Pass identifier (e.g., "pass1_paddleocr")
            page_num: 0-indexed page number
        """
        with self.lock:
            pass_data = self.data["passes"].get(pass_name)
            if not pass_data:
                logger.warning(f"Unknown pass: {pass_name}")
                return

            if page_num not in pass_data["completed_pages"]:
                pass_data["completed_pages"].append(page_num)
                pass_data["completed_pages"].sort()

            # Remove from failed if it was there
            if page_num in pass_data.get("failed_pages", []):
                pass_data["failed_pages"].remove(page_num)

            pass_data["status"] = PassStatus.IN_PROGRESS.value
            self._save()

    def mark_page_failed(self, pass_name: str, page_num: int, error: str) -> None:
        """
        Mark a page as failed for a pass.

        Args:
            pass_name: Pass identifier
            page_num: 0-indexed page number
            error: Error message
        """
        with self.lock:
            pass_data = self.data["passes"].get(pass_name)
            if not pass_data:
                logger.warning(f"Unknown pass: {pass_name}")
                return

            if page_num not in pass_data.get("failed_pages", []):
                if "failed_pages" not in pass_data:
                    pass_data["failed_pages"] = []
                pass_data["failed_pages"].append(page_num)

            # Log error
            self.data["errors"].append({
                "pass": pass_name,
                "page": page_num,
                "error": error,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

            self._save()

    def mark_pass_completed(self, pass_name: str) -> None:
        """Mark entire pass as completed."""
        with self.lock:
            pass_data = self.data["passes"].get(pass_name)
            if pass_data:
                pass_data["status"] = PassStatus.COMPLETED.value
                pass_data["completed_at"] = datetime.utcnow().isoformat() + "Z"
                self._save()

    def mark_pass_failed(self, pass_name: str, error: str) -> None:
        """Mark entire pass as failed."""
        with self.lock:
            pass_data = self.data["passes"].get(pass_name)
            if pass_data:
                pass_data["status"] = PassStatus.FAILED.value
                pass_data["error"] = error
                self._save()

    def mark_pass_in_progress(self, pass_name: str) -> None:
        """Mark pass as in progress."""
        with self.lock:
            pass_data = self.data["passes"].get(pass_name)
            if pass_data:
                pass_data["status"] = PassStatus.IN_PROGRESS.value
                self._save()

    def get_remaining_pages(self, pass_name: str, total_pages: int) -> List[int]:
        """
        Get list of pages that still need processing for a pass.

        Args:
            pass_name: Pass identifier
            total_pages: Total number of pages in document

        Returns:
            List of 0-indexed page numbers to process
        """
        with self.lock:
            pass_data = self.data["passes"].get(pass_name, {})
            completed = set(pass_data.get("completed_pages", []))
            all_pages = set(range(total_pages))
            return sorted(all_pages - completed)

    def get_pass_status(self, pass_name: str) -> PassStatus:
        """Get current status of a pass."""
        with self.lock:
            pass_data = self.data["passes"].get(pass_name, {})
            status_str = pass_data.get("status", PassStatus.NOT_STARTED.value)
            return PassStatus(status_str)

    def is_pass_completed(self, pass_name: str) -> bool:
        """Check if a pass is fully completed."""
        return self.get_pass_status(pass_name) == PassStatus.COMPLETED

    def update_batch_size(self, pass_name: str, new_batch_size: int, reason: str) -> None:
        """
        Update batch size for a pass (for OOM fallback tracking).

        Args:
            pass_name: Pass identifier
            new_batch_size: New batch size
            reason: Reason for change
        """
        with self.lock:
            pass_data = self.data["passes"].get(pass_name, {})
            old_batch_size = pass_data.get("current_batch_size", 2)
            pass_data["current_batch_size"] = new_batch_size
            pass_data["fallback_triggered"] = True

            self.data["errors"].append({
                "pass": pass_name,
                "action": f"Reduced batch size from {old_batch_size} to {new_batch_size}",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

            self._save()

    def get_batch_size(self, pass_name: str, default: int = 2) -> int:
        """Get current batch size for a pass."""
        with self.lock:
            pass_data = self.data["passes"].get(pass_name, {})
            return pass_data.get("current_batch_size", default)

    def get_resume_point(self) -> tuple:
        """
        Determine where to resume processing.

        Returns:
            Tuple of (pass_number, first_incomplete_page)
            Returns (0, 0) if starting fresh
        """
        with self.lock:
            pass_order = [
                ("pass1_paddleocr", 1),
                ("pass2_layout", 2),
                ("pass3_reading_order", 3),
                ("pass4_pdf_creation", 4),
                ("pass5_pdf_merge", 5),
            ]

            for pass_name, pass_num in pass_order:
                pass_data = self.data["passes"].get(pass_name, {})
                status = pass_data.get("status", PassStatus.NOT_STARTED.value)

                if status == PassStatus.COMPLETED.value:
                    continue
                elif status in (PassStatus.IN_PROGRESS.value, PassStatus.NOT_STARTED.value):
                    # Find first incomplete page
                    completed = set(pass_data.get("completed_pages", []))
                    total = self.data.get("total_pages", 0)

                    for page in range(total):
                        if page not in completed:
                            return (pass_num, page)

                    # All pages done but pass not marked complete
                    return (pass_num, 0)

            # All passes complete
            return (6, 0)

    def cleanup_on_success(self) -> None:
        """Remove checkpoint and temp files on successful completion."""
        logger.info("Cleaning up checkpoint files...")

        # Remove checkpoint file
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

        # Remove temp directories
        for subdir in ["ocr_results", "layout_results", "reading_order", "pdf_pages"]:
            temp_subdir = self.temp_dir / subdir
            if temp_subdir.exists():
                import shutil
                shutil.rmtree(temp_subdir)

        logger.info("Checkpoint cleanup complete")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of checkpoint state."""
        with self.lock:
            return {
                "doc_id": self.data.get("doc_id"),
                "total_pages": self.data.get("total_pages"),
                "started_at": self.data.get("started_at"),
                "last_updated": self.data.get("last_updated"),
                "passes": {
                    name: {
                        "status": data.get("status"),
                        "completed": len(data.get("completed_pages", [])),
                        "failed": len(data.get("failed_pages", [])),
                    }
                    for name, data in self.data.get("passes", {}).items()
                },
                "error_count": len(self.data.get("errors", [])),
            }
