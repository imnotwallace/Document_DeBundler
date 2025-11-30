"""
Pipeline Orchestrator for Multi-Pass OCR Processing.

Coordinates all passes in sequence:
Pass 0: Initialization
Pass 1: PaddleOCR text extraction
Pass 2: PP-DocLayout layout analysis (replaced Florence-2)
Pass 3: Reading order detection
Pass 4: PDF creation
Pass 5: PDF merging

Guarantees 4GB VRAM operation by loading only ONE model at a time.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

from .checkpoint_manager import CheckpointManager
from .model_lifecycle import ModelLifecycleManager
from .pass_config import MultiPassConfig
from .data_structures import PipelineState, PassStatus

logger = logging.getLogger(__name__)


class MultiPassPipelineOrchestrator:
    """
    Main orchestrator for the multi-pass OCR pipeline.

    Usage:
        orchestrator = MultiPassPipelineOrchestrator()
        result = orchestrator.process_document(
            pdf_path="input.pdf",
            output_path="output.pdf",
            config=MultiPassConfig()
        )
    """

    def __init__(
        self,
        vram_monitor: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            vram_monitor: Optional existing VRAMMonitor to reuse
            progress_callback: Optional callback(pass_name, current, total, message)
        """
        self.model_lifecycle = ModelLifecycleManager(vram_monitor=vram_monitor)
        self.progress_callback = progress_callback
        self.checkpoint: Optional[CheckpointManager] = None
        self.state: Optional[PipelineState] = None

    def process_document(
        self,
        pdf_path: Path,
        output_path: Path,
        config: Optional[MultiPassConfig] = None,
        workspace_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Process a document through the multi-pass pipeline.

        Args:
            pdf_path: Input PDF path
            output_path: Output searchable PDF path
            config: Pipeline configuration
            workspace_dir: Working directory (auto-generated if None)

        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()

        pdf_path = Path(pdf_path)
        output_path = Path(output_path)
        config = config or MultiPassConfig()

        # Generate workspace if not provided
        if workspace_dir is None:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            workspace_dir = pdf_path.parent / f".multipass_{doc_id}"
        else:
            workspace_dir = Path(workspace_dir)
            doc_id = workspace_dir.name

        workspace_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== Multi-Pass Pipeline Starting ===")
        logger.info(f"Input: {pdf_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Workspace: {workspace_dir}")

        # Initialize checkpoint manager
        self.checkpoint = CheckpointManager(workspace_dir, doc_id)

        try:
            # === PASS 0: Initialization ===
            self._report_progress("pass0_init", 0, 1, "Initializing pipeline...")
            self.state = self._run_pass_0(pdf_path, output_path, workspace_dir, config)
            self._report_progress("pass0_init", 1, 1, "Initialization complete")

            # === PASS 1: PaddleOCR ===
            self._report_progress("pass1_paddleocr", 0, self.state.total_pages, "Starting OCR...")
            self._run_pass_1()
            self._report_progress("pass1_paddleocr", self.state.total_pages, self.state.total_pages, "OCR complete")

            # === PASS 2: PP-DocLayout ===
            if config.enable_layout:
                self._report_progress("pass2_layout", 0, self.state.total_pages, "Starting layout analysis...")
                self._run_pass_2()
                self._report_progress("pass2_layout", self.state.total_pages, self.state.total_pages, "Layout analysis complete")
            else:
                logger.info("PP-DocLayout disabled, skipping Pass 2")
                self.state.pass_statuses["pass2_layout"] = PassStatus.SKIPPED

            # === PASS 3: Reading Order ===
            self._report_progress("pass3_reading_order", 0, self.state.total_pages, "Detecting reading order...")
            self._run_pass_3()
            self._report_progress("pass3_reading_order", self.state.total_pages, self.state.total_pages, "Reading order complete")

            # === PASS 4: PDF Creation ===
            self._report_progress("pass4_pdf_creation", 0, self.state.total_pages, "Creating PDF pages...")
            self._run_pass_4()
            self._report_progress("pass4_pdf_creation", self.state.total_pages, self.state.total_pages, "PDF pages created")

            # === PASS 5: PDF Merge ===
            self._report_progress("pass5_pdf_merge", 0, 1, "Merging final PDF...")
            self._run_pass_5()
            self._report_progress("pass5_pdf_merge", 1, 1, "PDF merged")

            # Cleanup on success
            if config.cleanup_on_success:
                self.checkpoint.cleanup_on_success()

            total_time = time.time() - start_time

            result = {
                "status": "success",
                "output_path": str(output_path),
                "total_pages": self.state.total_pages,
                "total_time_seconds": total_time,
                "pages_per_second": self.state.total_pages / total_time if total_time > 0 else 0,
                "passes_completed": {
                    name: status.value
                    for name, status in self.state.pass_statuses.items()
                },
                "errors": self.state.errors,
            }

            logger.info(f"=== Pipeline Complete ===")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Pages/sec: {result['pages_per_second']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)

            # Don't cleanup on failure
            return {
                "status": "failed",
                "error": str(e),
                "checkpoint_path": str(self.checkpoint.checkpoint_path) if self.checkpoint else None,
                "resume_command": f"Resume from: {workspace_dir}",
            }

        finally:
            # Always unload models
            self.model_lifecycle.unload_current()

    def resume_from_checkpoint(
        self,
        workspace_dir: Path,
        output_path: Optional[Path] = None,
        config: Optional[MultiPassConfig] = None,
    ) -> Dict[str, Any]:
        """
        Resume processing from a checkpoint.

        Args:
            workspace_dir: Workspace directory containing checkpoint
            output_path: Output path (reads from checkpoint if None)
            config: Configuration (reads from checkpoint if None)

        Returns:
            Processing result dictionary
        """
        workspace_dir = Path(workspace_dir)
        checkpoint_path = workspace_dir / "temp" / "checkpoint.json"

        if not checkpoint_path.exists():
            raise ValueError(f"No checkpoint found at {checkpoint_path}")

        import json
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        doc_id = checkpoint_data.get("doc_id", "unknown")
        logger.info(f"Resuming from checkpoint: {doc_id}")

        # TODO: Extract original paths from checkpoint
        # For now, require explicit paths
        if output_path is None:
            raise ValueError("output_path required for resume")

        # Find input PDF from image paths
        # This is a simplified approach - full implementation would store this
        raise NotImplementedError("Resume from checkpoint not yet fully implemented")

    def _run_pass_0(
        self,
        pdf_path: Path,
        output_path: Path,
        workspace_dir: Path,
        config: MultiPassConfig,
    ) -> PipelineState:
        """
        Pass 0: Initialization.

        - Validate inputs
        - Count pages
        - Render pages to images
        - Create workspace structure
        - Check for resume point
        """
        logger.info("=== PASS 0: Initialization ===")

        from .passes.pass_0_init import run_pass_0

        state = run_pass_0(
            pdf_path=pdf_path,
            output_path=output_path,
            workspace_dir=workspace_dir,
            config=config,
            checkpoint=self.checkpoint,
            model_lifecycle=self.model_lifecycle,
        )

        state.pass_statuses["pass0_init"] = PassStatus.COMPLETED
        return state

    def _run_pass_1(self) -> None:
        """
        Pass 1: PaddleOCR text extraction.

        - Load PaddleOCR
        - Process all pages
        - Save per-page JSON
        - Unload + cleanup
        """
        logger.info("=== PASS 1: PaddleOCR ===")

        from .passes.pass_1_paddleocr import run_pass_1

        run_pass_1(
            state=self.state,
            checkpoint=self.checkpoint,
            model_lifecycle=self.model_lifecycle,
            progress_callback=lambda curr, total, msg: self._report_progress(
                "pass1_paddleocr", curr, total, msg
            ),
        )

        self.state.pass_statuses["pass1_paddleocr"] = PassStatus.COMPLETED
        self.checkpoint.mark_pass_completed("pass1_paddleocr")

    def _run_pass_2(self) -> None:
        """
        Pass 2: PP-DocLayout layout analysis.

        - Load PP-DocLayout (22MB, much smaller than Florence-2's 2.5GB)
        - Process all pages with optimistic batching
        - Save per-page JSON
        - Unload + cleanup
        """
        logger.info("=== PASS 2: PP-DocLayout ===")

        from .passes.pass_2_florence import run_pass_2

        run_pass_2(
            state=self.state,
            checkpoint=self.checkpoint,
            model_lifecycle=self.model_lifecycle,
            progress_callback=lambda curr, total, msg: self._report_progress(
                "pass2_layout", curr, total, msg
            ),
        )

        self.state.pass_statuses["pass2_layout"] = PassStatus.COMPLETED
        self.checkpoint.mark_pass_completed("pass2_layout")

    def _run_pass_3(self) -> None:
        """
        Pass 3: Reading order detection.

        - Load OCR + Layout results
        - Run fusion strategy
        - Detect reading order
        - Save per-page JSON
        """
        logger.info("=== PASS 3: Reading Order ===")

        from .passes.pass_3_reading_order import run_pass_3

        run_pass_3(
            state=self.state,
            checkpoint=self.checkpoint,
            progress_callback=lambda curr, total, msg: self._report_progress(
                "pass3_reading_order", curr, total, msg
            ),
        )

        self.state.pass_statuses["pass3_reading_order"] = PassStatus.COMPLETED
        self.checkpoint.mark_pass_completed("pass3_reading_order")

    def _run_pass_4(self) -> None:
        """
        Pass 4: PDF creation.

        - Create per-page PDFs with invisible text overlay
        - Save to pdf_pages directory
        """
        logger.info("=== PASS 4: PDF Creation ===")

        from .passes.pass_4_pdf_creation import run_pass_4

        run_pass_4(
            state=self.state,
            checkpoint=self.checkpoint,
            progress_callback=lambda curr, total, msg: self._report_progress(
                "pass4_pdf_creation", curr, total, msg
            ),
        )

        self.state.pass_statuses["pass4_pdf_creation"] = PassStatus.COMPLETED
        self.checkpoint.mark_pass_completed("pass4_pdf_creation")

    def _run_pass_5(self) -> None:
        """
        Pass 5: PDF merging.

        - Merge all page PDFs
        - Apply compression
        - Save final output
        """
        logger.info("=== PASS 5: PDF Merge ===")

        from .passes.pass_5_pdf_merge import run_pass_5

        run_pass_5(
            state=self.state,
            checkpoint=self.checkpoint,
        )

        self.state.pass_statuses["pass5_pdf_merge"] = PassStatus.COMPLETED
        self.checkpoint.mark_pass_completed("pass5_pdf_merge")

    def _report_progress(
        self,
        pass_name: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        """Report progress through callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(pass_name, current, total, message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")
