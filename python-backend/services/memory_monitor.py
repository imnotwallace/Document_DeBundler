"""
Memory monitoring and leak detection utility.

Provides tools for tracking memory usage, detecting leaks, and logging resource lifecycle.
"""

import psutil
import gc
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import tracemalloc

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: datetime
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage as percentage of total
    available_mb: float  # Available system memory in MB
    gc_stats: Dict[str, int] = field(default_factory=dict)  # Garbage collection stats
    tracemalloc_top: Optional[list] = None  # Top memory allocations if tracemalloc enabled


class MemoryMonitor:
    """
    Monitor memory usage and detect potential leaks.

    Usage:
        monitor = MemoryMonitor(enable_tracemalloc=True)
        monitor.start_tracking("batch_processing")
        # ... do work ...
        snapshot = monitor.checkpoint("after_batch")
        monitor.stop_tracking()
        monitor.report()
    """

    def __init__(self, enable_tracemalloc: bool = False, tracemalloc_top_lines: int = 10):
        """
        Initialize memory monitor.

        Args:
            enable_tracemalloc: If True, enable detailed memory allocation tracking (adds overhead)
            tracemalloc_top_lines: Number of top memory allocations to track
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.tracemalloc_top_lines = tracemalloc_top_lines
        self.process = psutil.Process()
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.tracking_label: Optional[str] = None
        self.start_snapshot: Optional[MemorySnapshot] = None

        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("tracemalloc enabled for detailed memory tracking")

    def _get_snapshot(self, label: str) -> MemorySnapshot:
        """Capture current memory state."""
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        vm = psutil.virtual_memory()

        # Collect garbage collection stats
        gc_stats = {
            f"gen{i}_collections": gc.get_count()[i]
            for i in range(len(gc.get_count()))
        }
        gc_stats["uncollectable"] = len(gc.garbage)

        # Get top memory allocations if tracemalloc enabled
        tracemalloc_top = None
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            tracemalloc_top = [
                {
                    "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count
                }
                for stat in top_stats[:self.tracemalloc_top_lines]
            ]

        return MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=mem_percent,
            available_mb=vm.available / (1024 * 1024),
            gc_stats=gc_stats,
            tracemalloc_top=tracemalloc_top
        )

    def start_tracking(self, label: str) -> MemorySnapshot:
        """
        Start tracking memory for a specific operation.

        Args:
            label: Label for this tracking session

        Returns:
            Initial memory snapshot
        """
        self.tracking_label = label
        self.start_snapshot = self._get_snapshot(f"{label}_start")
        self.snapshots[f"{label}_start"] = self.start_snapshot

        logger.info(
            f"Memory tracking started for '{label}': "
            f"RSS={self.start_snapshot.rss_mb:.2f}MB, "
            f"Available={self.start_snapshot.available_mb:.2f}MB"
        )

        return self.start_snapshot

    def checkpoint(self, label: str) -> MemorySnapshot:
        """
        Create a memory checkpoint during tracking.

        Args:
            label: Label for this checkpoint

        Returns:
            Memory snapshot at checkpoint
        """
        snapshot = self._get_snapshot(label)
        self.snapshots[label] = snapshot

        if self.start_snapshot:
            delta_mb = snapshot.rss_mb - self.start_snapshot.rss_mb
            logger.info(
                f"Memory checkpoint '{label}': "
                f"RSS={snapshot.rss_mb:.2f}MB ({delta_mb:+.2f}MB from start), "
                f"Available={snapshot.available_mb:.2f}MB"
            )

        return snapshot

    def stop_tracking(self) -> Optional[MemorySnapshot]:
        """
        Stop tracking and create final snapshot.

        Returns:
            Final memory snapshot if tracking was active
        """
        if not self.tracking_label or not self.start_snapshot:
            return None

        final_snapshot = self._get_snapshot(f"{self.tracking_label}_end")
        self.snapshots[f"{self.tracking_label}_end"] = final_snapshot

        delta_mb = final_snapshot.rss_mb - self.start_snapshot.rss_mb
        delta_percent = ((final_snapshot.rss_mb / self.start_snapshot.rss_mb) - 1) * 100

        logger.info(
            f"Memory tracking stopped for '{self.tracking_label}': "
            f"RSS={final_snapshot.rss_mb:.2f}MB ({delta_mb:+.2f}MB, {delta_percent:+.1f}%), "
            f"Available={final_snapshot.available_mb:.2f}MB"
        )

        # Check for potential leak (memory increased by more than 10%)
        if delta_mb > 0 and delta_percent > 10:
            logger.warning(
                f"POTENTIAL MEMORY LEAK DETECTED in '{self.tracking_label}': "
                f"Memory increased by {delta_mb:.2f}MB ({delta_percent:.1f}%)"
            )

        return final_snapshot

    def get_current_memory(self) -> Dict[str, Any]:
        """
        Get current memory usage stats.

        Returns:
            Dictionary with current memory stats
        """
        snapshot = self._get_snapshot("current")
        return {
            "rss_mb": snapshot.rss_mb,
            "vms_mb": snapshot.vms_mb,
            "percent": snapshot.percent,
            "available_mb": snapshot.available_mb,
            "gc_stats": snapshot.gc_stats
        }

    def force_gc(self) -> int:
        """
        Force garbage collection and return number of objects collected.

        Returns:
            Number of objects collected
        """
        before = self.get_current_memory()
        collected = gc.collect()
        after = self.get_current_memory()

        freed_mb = before["rss_mb"] - after["rss_mb"]
        logger.info(
            f"Forced GC: collected {collected} objects, "
            f"freed {freed_mb:.2f}MB"
        )

        return collected

    def report(self) -> Dict[str, Any]:
        """
        Generate a report of all snapshots.

        Returns:
            Dictionary with complete memory tracking report
        """
        report = {
            "tracking_label": self.tracking_label,
            "snapshots": {},
            "summary": {}
        }

        for label, snapshot in self.snapshots.items():
            report["snapshots"][label] = {
                "timestamp": snapshot.timestamp.isoformat(),
                "rss_mb": round(snapshot.rss_mb, 2),
                "vms_mb": round(snapshot.vms_mb, 2),
                "percent": round(snapshot.percent, 2),
                "available_mb": round(snapshot.available_mb, 2),
                "gc_stats": snapshot.gc_stats
            }

            if snapshot.tracemalloc_top:
                report["snapshots"][label]["top_allocations"] = snapshot.tracemalloc_top

        # Generate summary
        if self.start_snapshot and self.tracking_label:
            end_label = f"{self.tracking_label}_end"
            if end_label in self.snapshots:
                end_snapshot = self.snapshots[end_label]
                report["summary"] = {
                    "start_rss_mb": round(self.start_snapshot.rss_mb, 2),
                    "end_rss_mb": round(end_snapshot.rss_mb, 2),
                    "delta_mb": round(end_snapshot.rss_mb - self.start_snapshot.rss_mb, 2),
                    "delta_percent": round(
                        ((end_snapshot.rss_mb / self.start_snapshot.rss_mb) - 1) * 100, 2
                    )
                }

        return report

    def cleanup(self):
        """Clean up monitoring resources."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("tracemalloc stopped")

        self.snapshots.clear()
        self.tracking_label = None
        self.start_snapshot = None
