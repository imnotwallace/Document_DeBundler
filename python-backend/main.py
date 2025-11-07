"""
Document De-Bundler - Python Backend
Main entry point for PDF processing via stdin/stdout IPC with Tauri frontend
"""

import sys
import json
import logging
from typing import Dict, Any
from pathlib import Path

# Configure logging to stderr (stdout is used for IPC)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class IPCHandler:
    """Handles JSON-based IPC communication via stdin/stdout"""

    def __init__(self):
        self.running = True
        self.cancelled = False
        self.current_request_id = None

    def send_event(self, event_type: str, data: Any, request_id: str = None):
        """Send an event to the frontend via stdout"""
        event = {
            "type": event_type,
            "data": data
        }
        # Include request_id if provided
        if request_id:
            event["request_id"] = request_id
            logger.debug(f"Sending {event_type} event with request_id: {request_id}")
        else:
            logger.debug(f"Sending {event_type} event without request_id")
        print(json.dumps(event), flush=True)

    def send_progress(self, current: int, total: int, message: str, percent: float = None):
        """Send progress update"""
        if percent is None:
            percent = (current / total * 100) if total > 0 else 0

        self.send_event("progress", {
            "current": current,
            "total": total,
            "message": message,
            "percent": percent
        })

    def send_result(self, result: Any):
        """Send processing result"""
        self.send_event("result", result, request_id=self.current_request_id)

    def send_error(self, error_message: str):
        """Send error message"""
        self.send_event("error", {"message": error_message}, request_id=self.current_request_id)

    def handle_command(self, command: Dict[str, Any]):
        """Process incoming command"""
        cmd_type = command.get("command")
        # Extract request_id for request-response correlation
        self.current_request_id = command.get("request_id")
        logger.debug(f"Handling command '{cmd_type}' with request_id: {self.current_request_id}")

        if cmd_type == "analyze":
            self.handle_analyze(command)
        elif cmd_type == "process":
            self.handle_process(command)
        elif cmd_type == "ocr_batch":
            self.handle_ocr_batch(command)
        elif cmd_type == "get_hardware_capabilities":
            self.handle_get_hardware_capabilities(command)
        elif cmd_type == "list_available_languages":
            self.handle_list_available_languages(command)
        elif cmd_type == "list_installed_languages":
            self.handle_list_installed_languages(command)
        elif cmd_type == "get_language_status":
            self.handle_get_language_status(command)
        elif cmd_type == "download_language_pack":
            self.handle_download_language_pack(command)
        elif cmd_type == "cancel":
            self.handle_cancel()
        else:
            self.send_error(f"Unknown command: {cmd_type}")

    def handle_analyze(self, command: Dict[str, Any]):
        """Analyze PDF structure"""
        file_path = command.get("file_path")
        logger.info(f"Analyzing PDF: {file_path}")

        try:
            from services.pdf_processor import PDFProcessor
            from services.ocr_service import OCRService
            from services.cache_manager import get_cache_manager
            
            cache = get_cache_manager()
            
            self.send_progress(0, 100, "Opening PDF...")
            
            # Open PDF and get basic info
            with PDFProcessor(file_path) as pdf:
                total_pages = pdf.get_page_count()
                
                self.send_progress(20, 100, f"Analyzing {total_pages} pages...")
                
                # Run structure analysis
                analysis = pdf.analyze_structure()
                
                self.send_progress(60, 100, "Checking for existing cache...")
                
                # Check if document already processed
                existing_doc = cache.find_document_by_path(file_path)
                if existing_doc:
                    doc_id = existing_doc['doc_id']
                    has_cache = cache.has_embeddings(doc_id)
                    last_phase = cache.get_last_completed_phase(doc_id)
                else:
                    has_cache = False
                    last_phase = None
                    doc_id = None
                
                self.send_progress(100, 100, "Analysis complete")
                
                # Return comprehensive analysis
                self.send_result({
                    "status": "success",
                    "total_pages": total_pages,
                    "needs_ocr": analysis["needs_ocr"],
                    "has_text_layer": analysis["has_valid_text_layer"],
                    "ocr_pages_estimated": total_pages if analysis["needs_ocr"] else 0,
                    "avg_text_confidence": analysis["avg_confidence"],
                    "quality_samples": analysis["quality_samples"],
                    "has_cache": has_cache,
                    "last_completed_phase": last_phase,
                    "doc_id": doc_id,
                    "file_size_mb": round(Path(file_path).stat().st_size / (1024 * 1024), 2)
                })
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            self.send_error(str(e))

    def handle_process(self, command: Dict[str, Any]):
        """Process PDF with splitting and OCR - Complete workflow"""
        file_path = command.get("file_path")
        options = command.get("options", {})
        logger.info(f"Processing PDF: {file_path} with options: {options}")

        try:
            from pathlib import Path
            from services.pdf_processor import PDFProcessor
            from services.ocr_service import OCRService
            from services.cache_manager import get_cache_manager
            from services.embedding_service import generate_embeddings_for_document
            from services.split_detection import detect_splits_for_document
            from services.naming_service import NamingService
            from services.bundler import Bundler
            import gc
            import hashlib
            
            cache = get_cache_manager()
            file_path_obj = Path(file_path)
            
            # Extract options
            force_ocr = options.get("force_ocr", False)
            skip_splitting = options.get("skip_splitting", False)
            
            # NEW: Add LLM options
            use_llm_refinement = options.get("use_llm_refinement", True)  # Default: True
            use_llm_naming = options.get("use_llm_naming", True)  # Default: True
            output_dir = options.get("output_dir", file_path_obj.parent / "output")
            output_format = options.get("output_format", "folder")  # "folder" or "zip"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ===== PHASE 0: Setup and Cache Check =====
            self.send_progress(0, 100, "Initializing processing...")
            
            # Check for existing document
            existing_doc = cache.find_document_by_path(file_path)
            
            if existing_doc:
                doc_id = existing_doc['doc_id']
                last_phase = cache.get_last_completed_phase(doc_id)
                logger.info(f"Found existing document {doc_id[:8]}..., last phase: {last_phase}")
            else:
                # Create new document entry
                with PDFProcessor(file_path) as pdf:
                    total_pages = pdf.get_page_count()
                    file_size = file_path_obj.stat().st_size
                    doc_id = cache.create_document(file_path, total_pages, file_size)
                    last_phase = None
                    logger.info(f"Created new document entry: {doc_id[:8]}...")
            
            # ===== PHASE 1: OCR / Text Extraction =====
            if not last_phase or last_phase != "ocr_complete":
                cache.log_phase(doc_id, "ocr", "started", "Starting OCR/text extraction")
                self.send_progress(5, 100, "Phase 1: Extracting text from pages...")
                
                ocr_service = None
                try:
                    with PDFProcessor(file_path) as pdf:
                        total_pages = pdf.get_page_count()
                        
                        # Check if OCR is needed
                        analysis = pdf.analyze_structure()
                        needs_ocr = analysis["needs_ocr"] or force_ocr
                        
                        if needs_ocr:
                            self.send_progress(6, 100, "Initializing OCR engine...")
                            ocr_service = OCRService(gpu=True)
                            
                            if not ocr_service.is_available():
                                raise RuntimeError("OCR service failed to initialize")
                            
                            engine_info = ocr_service.get_engine_info()
                            logger.info(f"OCR engine: {engine_info['engine']}, GPU: {engine_info['gpu_enabled']}")
                        
                        # Process pages with progress callback
                        def progress_callback(current, total, message):
                            percent = 5 + int((current / total) * 25)  # 5-30%
                            self.send_progress(percent, 100, f"Phase 1: {message}")
                        
                        results = pdf.process_pages_with_ocr(
                            ocr_service=ocr_service,
                            start_page=0,
                            end_page=None,
                            batch_size=10,
                            progress_callback=progress_callback,
                            force_ocr=force_ocr
                        )
                        
                        # Save results to cache
                        self.send_progress(30, 100, "Saving text extraction results...")
                        for result in results:
                            cache.save_page_text(
                                doc_id=doc_id,
                                page_num=result['page_num'],
                                text=result['text'],
                                has_text_layer=(result['method'] == 'text_layer'),
                                ocr_method=result['method'],
                                ocr_confidence=result.get('quality_metrics', {}).get('confidence_score') if result.get('quality_metrics') else None
                            )
                        
                        cache.log_phase(doc_id, "ocr", "completed", f"Extracted text from {len(results)} pages")
                        cache.update_document_status(doc_id, "ocr_complete")
                        
                finally:
                    if ocr_service:
                        ocr_service.cleanup()
                        gc.collect()
                        logger.info("OCR cleanup complete")
            
            else:
                logger.info("Skipping OCR phase - already completed")
                self.send_progress(30, 100, "Phase 1: Using cached text extraction")
            
            # ===== PHASE 2: Embedding Generation =====
            if not cache.has_embeddings(doc_id):
                cache.log_phase(doc_id, "embeddings", "started", "Generating semantic embeddings")
                self.send_progress(30, 100, "Phase 2: Generating semantic embeddings...")
                
                def embedding_progress(current, total, message):
                    percent = 30 + int((current / total) * 20)  # 30-50%
                    self.send_progress(percent, 100, f"Phase 2: {message}")
                
                try:
                    success = generate_embeddings_for_document(doc_id, progress_callback=embedding_progress)
                    if success:
                        cache.log_phase(doc_id, "embeddings", "completed", "Embeddings generated")
                        cache.update_document_status(doc_id, "embeddings_complete")
                    else:
                        raise RuntimeError("Embedding generation failed")
                except Exception as e:
                    logger.error(f"Embedding generation error: {e}", exc_info=True)
                    cache.log_phase(doc_id, "embeddings", "failed", str(e))
                    raise
            else:
                logger.info("Skipping embedding phase - already completed")
                self.send_progress(50, 100, "Phase 2: Using cached embeddings")
            
            # ===== PHASE 3: Split Detection =====
            if not skip_splitting:
                cache.log_phase(doc_id, "split_detection", "started", "Detecting document boundaries")
                self.send_progress(50, 100, "Phase 3: Detecting document boundaries...")
                
                def split_progress(current, total, message):
                    percent = 50 + int((current / total) * 15)  # 50-65%
                    self.send_progress(percent, 100, f"Phase 3: {message}")
                
                try:
                    # UPDATED: Add use_llm_refinement parameter
                    split_count = detect_splits_for_document(
                        doc_id,
                        use_llm_refinement=use_llm_refinement,
                        progress_callback=split_progress
                    )
                    cache.log_phase(doc_id, "split_detection", "completed", f"Detected {split_count} split points")
                    cache.update_document_status(doc_id, "splits_detected")
                    logger.info(f"Detected {split_count} split candidates")
                except Exception as e:
                    logger.error(f"Split detection error: {e}", exc_info=True)
                    cache.log_phase(doc_id, "split_detection", "failed", str(e))
                    raise
            else:
                split_count = 0
                logger.info("Skipping split detection per user request")
                self.send_progress(65, 100, "Phase 3: Split detection skipped")
            
            # ===== PHASE 4: Document Extraction and Naming =====
            cache.log_phase(doc_id, "extraction", "started", "Extracting document segments")
            self.send_progress(65, 100, "Phase 4: Extracting documents...")
            
            # Get split candidates
            split_candidates = cache.get_split_candidates(doc_id, status='pending')
            
            # Build split points list (always include start and end)
            split_points = [0]  # Start at page 0
            
            if split_candidates:
                # Add split points from candidates (sorted)
                for candidate in sorted(split_candidates, key=lambda x: x['split_page']):
                    if candidate['confidence'] >= 0.5:  # Only use confident splits
                        split_points.append(candidate['split_page'])
            
            # Get total pages
            doc_info = cache.get_document(doc_id)
            total_pages = doc_info['total_pages']
            split_points.append(total_pages)  # End at last page
            
            # Remove duplicates and sort
            split_points = sorted(list(set(split_points)))
            
            logger.info(f"Split points: {split_points}")
            
            # Extract document segments
            extracted_files = []
            naming_service = NamingService()
            
            with PDFProcessor(file_path) as pdf:
                num_segments = len(split_points) - 1
                
                for idx in range(num_segments):
                    start_page = split_points[idx]
                    end_page = split_points[idx + 1] - 1  # Inclusive end
                    
                    percent = 65 + int((idx / num_segments) * 15)  # 65-80%
                    self.send_progress(percent, 100, f"Extracting document {idx+1}/{num_segments}...")
                    
                    # Get text for naming
                    first_page_text = cache.get_page_text(doc_id, start_page) or ""
                    second_page_text = cache.get_page_text(doc_id, start_page + 1) if start_page + 1 <= end_page else None
                    
                    # UPDATED: Add LLM naming with second page context
                    suggested_name = naming_service.suggest_name(
                        text_content=first_page_text,
                        page_num=idx + 1,
                        fallback_prefix=file_path_obj.stem,
                        second_page_text=second_page_text,  # NEW
                        use_llm=use_llm_naming  # NEW
                    )
                    
                    # Save naming suggestion to cache
                    cache.save_document_name(
                        doc_id=doc_id,
                        start_page=start_page,
                        end_page=end_page,
                        suggested_name=suggested_name,
                        reasoning="Heuristic-based naming",
                        confidence=0.5
                    )
                    
                    # Extract pages to new PDF
                    output_filename = f"{suggested_name}.pdf"
                    output_path = output_dir / output_filename
                    
                    # Ensure unique filename
                    counter = 1
                    while output_path.exists():
                        output_filename = f"{suggested_name}_{counter}.pdf"
                        output_path = output_dir / output_filename
                        counter += 1
                    
                    pdf.extract_page_range(start_page, end_page, output_path)
                    
                    # Record extraction result
                    file_size = output_path.stat().st_size
                    with cache.get_connection() as conn:
                        conn.execute("""
                            INSERT INTO split_results
                            (doc_id, output_filename, output_path, start_page, end_page, page_count, file_size_bytes)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (doc_id, output_filename, str(output_path), start_page, end_page, end_page - start_page + 1, file_size))
                    
                    extracted_files.append({
                        'filename': output_filename,
                        'path': str(output_path),
                        'start_page': start_page,
                        'end_page': end_page,
                        'page_count': end_page - start_page + 1,
                        'size_mb': round(file_size / (1024 * 1024), 2)
                    })
                    
                    logger.info(f"Extracted {output_filename}: pages {start_page+1}-{end_page+1}")
            
            cache.log_phase(doc_id, "extraction", "completed", f"Extracted {len(extracted_files)} documents")
            
            # ===== PHASE 5: Bundling =====
            self.send_progress(80, 100, "Phase 5: Organizing output...")
            
            bundler = Bundler()
            final_output = None
            
            if output_format == "zip":
                # Create ZIP archive
                zip_path = output_dir.parent / f"{file_path_obj.stem}_split.zip"
                bundler.create_zip([Path(f['path']) for f in extracted_files], zip_path)
                final_output = str(zip_path)
                logger.info(f"Created ZIP archive: {zip_path}")
            else:
                # Files already organized in folder
                final_output = str(output_dir)
                logger.info(f"Documents organized in: {output_dir}")
            
            cache.log_phase(doc_id, "complete", "completed", "Processing complete")
            cache.update_document_status(doc_id, "complete")
            
            # ===== PHASE 5.5: LLM Cleanup (if used) =====
            if use_llm_refinement or use_llm_naming:
                try:
                    from services.llm.manager import cleanup_llm
                    from services.llm.settings import get_settings
                    
                    settings = get_settings()
                    if settings.auto_cleanup_enabled:
                        self.send_progress(95, 100, "Cleaning up LLM resources...")
                        cleanup_llm()
                        logger.info("LLM cleanup complete")
                except Exception as e:
                    logger.warning(f"LLM cleanup error: {e}")
            
            # ===== Final Result =====
            self.send_progress(100, 100, "Processing complete!")
            
            self.send_result({
                "status": "success",
                "doc_id": doc_id,
                "total_pages": total_pages,
                "split_count": len(split_points) - 1,
                "documents_created": len(extracted_files),
                "output_location": final_output,
                "output_format": output_format,
                "extracted_files": extracted_files,
                "cache_stats": cache.get_cache_stats()
            })
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.send_error(str(e))

    def handle_ocr_batch(self, command: Dict[str, Any]):
        """
        Handle batch OCR processing command

        Command format:
        {
            "command": "ocr_batch",
            "files": ["path/to/file1.pdf", "path/to/file2.pdf"],
            "output_dir": "path/to/output"
        }

        Emits:
        - progress events: {"type": "progress", "data": {...}}
        - result event: {"type": "result", "data": {"successful": [...], "failed": [...]}}
        - error event on failure: {"type": "error", "data": {"message": "..."}}
        """
        import os

        current_file = None
        service = None

        try:
            # Extract options from command (Rust sends them nested in 'options')
            options = command.get('options', {})
            files = options.get('files', [])
            output_dir = options.get('output_dir')
            ocr_config = options.get('ocr_config')  # Optional OCR configuration from frontend

            # Validate inputs
            if not files:
                self.send_error("No files provided for OCR batch processing")
                return

            if not output_dir:
                self.send_error("No output directory provided")
                return

            logger.info(f"Starting OCR batch: {len(files)} files")
            logger.info(f"Output directory: {output_dir}")
            if ocr_config:
                logger.info(f"OCR config provided: DPI={ocr_config.get('dpi')}, GPU={ocr_config.get('use_gpu')}, Mode={ocr_config.get('processing_mode')}")
            else:
                logger.info("Using automatic OCR configuration")

            # Create output directory if needed
            os.makedirs(output_dir, exist_ok=True)

            # Import OCRBatchService
            from services.ocr_batch_service import OCRBatchService

            # Create progress callback
            def progress_callback(current, total, message, percent, eta=None):
                # Check cancellation
                if self.cancelled:
                    raise Exception("Processing cancelled by user")

                # Send progress event
                self.send_progress(current, total, message, percent)

            # Create service instance with OCR configuration
            # Note: Cancellation is handled via progress_callback raising exception
            service = OCRBatchService(
                progress_callback=progress_callback,
                cancellation_flag=None,
                ocr_config=ocr_config  # Pass OCR config from frontend
            )

            # Process batch
            result = service.process_batch(files, output_dir)

            # Send result
            self.send_result(result)

            logger.info(f"OCR batch complete: {result['statistics']['total_files']} files processed, "
                       f"{len(result['successful'])} successful, {len(result['failed'])} failed")

        except Exception as e:
            error_msg = f"OCR batch failed: {str(e)}"

            # Add context if available
            if current_file:
                error_msg += f" (processing file: {current_file})"

            logger.error(error_msg, exc_info=True)
            self.send_error(error_msg)
        finally:
            # Cleanup
            if service:
                try:
                    service.cleanup()
                    logger.info("OCR batch service cleanup complete")
                except Exception as e:
                    logger.warning(f"OCR batch cleanup error: {e}")

    def handle_cancel(self):
        """Cancel current operation"""
        logger.info("Cancel requested")
        self.running = False
        self.cancelled = True
        self.send_event('info', {'message': 'Cancellation requested'})

    def handle_get_hardware_capabilities(self, command: Dict[str, Any]):
        """
        Get hardware capabilities for OCR configuration
        Returns GPU/CPU info and recommended settings
        """
        try:
            from services.ocr.config import detect_hardware_capabilities, get_optimal_batch_size, get_adaptive_dpi, detect_model_type

            logger.info("Detecting hardware capabilities...")

            # Detect hardware
            capabilities = detect_hardware_capabilities()

            # Detect model type
            model_type = detect_model_type()
            
            # Calculate optimal batch size
            gpu_batch_size = get_optimal_batch_size(
                use_gpu=True,
                gpu_memory_gb=capabilities.get('gpu_memory_gb', 0),
                system_memory_gb=capabilities.get('system_memory_gb', 0),
                model_type=model_type
            )

            # Calculate recommended DPI
            recommended_dpi = get_adaptive_dpi(
                use_gpu=capabilities.get('gpu_available', False),
                gpu_memory_gb=capabilities.get('gpu_memory_gb', 0),
                system_memory_gb=capabilities.get('system_memory_gb', 0),
                target_quality='balanced',
                model_type=model_type
            )

            # Build response
            result = {
                'gpu_available': capabilities.get('gpu_available', False),
                'cuda_available': capabilities.get('cuda_available', False),
                'gpu_memory_gb': capabilities.get('gpu_memory_gb', 0.0),
                'system_memory_gb': capabilities.get('system_memory_gb', 0.0),
                'cpu_count': capabilities.get('cpu_count', 1),
                'platform': capabilities.get('platform', 'unknown'),
                'recommended_batch_size': gpu_batch_size,
                'recommended_dpi': recommended_dpi
            }

            logger.info(f"Hardware capabilities: GPU={result['gpu_available']}, "
                       f"VRAM={result['gpu_memory_gb']:.1f}GB, "
                       f"RAM={result['system_memory_gb']:.1f}GB")

            # Send result
            self.send_result(result)

        except Exception as e:
            error_msg = f"Failed to detect hardware capabilities: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.send_error(error_msg)

    def handle_list_available_languages(self, command: Dict[str, Any]):
        """
        List all available language packs with installation status.
        Returns list of languages with script-based model information.
        """
        try:
            from services.language_pack_manager import get_language_pack_manager
            from dataclasses import asdict

            logger.info("Listing available languages...")

            manager = get_language_pack_manager()
            language_statuses = manager.get_all_language_statuses()

            # Convert dataclasses to dicts
            languages = [asdict(status) for status in language_statuses]

            logger.info(f"Found {len(languages)} available languages")

            # Send result
            self.send_result({
                'languages': languages,
                'count': len(languages)
            })

        except Exception as e:
            error_msg = f"Failed to list available languages: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.send_error(error_msg)

    def handle_list_installed_languages(self, command: Dict[str, Any]):
        """
        List installed language packs.
        Returns list of language codes that are fully installed.
        """
        try:
            from services.language_pack_manager import get_language_pack_manager

            logger.info("Listing installed languages...")

            manager = get_language_pack_manager()
            installed = manager.get_installed_languages()

            logger.info(f"Found {len(installed)} installed languages: {', '.join(installed)}")

            # Send result
            self.send_result({
                'installed_languages': installed,
                'count': len(installed)
            })

        except Exception as e:
            error_msg = f"Failed to list installed languages: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.send_error(error_msg)

    def handle_get_language_status(self, command: Dict[str, Any]):
        """
        Get installation status for a specific language
        Returns detailed status including missing models
        """
        try:
            from services.language_pack_manager import get_language_pack_manager
            from dataclasses import asdict

            # Get language_code from options (sent by Rust)
            options = command.get("options", {})
            language_code = options.get("language_code")
            if not language_code:
                self.send_error("Missing required parameter: language_code")
                return

            logger.info(f"Checking status for language: {language_code}")

            manager = get_language_pack_manager()
            status = manager.get_language_status(language_code)

            if status is None:
                self.send_error(f"Language not supported: {language_code}")
                return

            # Convert dataclass to dict
            status_dict = asdict(status)

            logger.info(f"Language {language_code} status: installed={status.installed}")

            # Send result
            self.send_result(status_dict)

        except Exception as e:
            error_msg = f"Failed to get language status: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.send_error(error_msg)

    def handle_download_language_pack(self, command: Dict[str, Any]):
        """
        Download and install a language pack using PaddleOCR's auto-download.
        Emits progress events during download and installation.
        """
        try:
            from services.language_pack_manager import get_language_pack_manager
            from dataclasses import asdict

            # Get parameters from options (sent by Rust)
            options = command.get("options", {})
            language_code = options.get("language_code")
            version = options.get("version", "mobile")  # Default to mobile if not specified

            if not language_code:
                self.send_error("Missing required parameter: language_code")
                return

            logger.info(f"Triggering PaddleOCR auto-download for language: {language_code} (version: {version})")

            manager = get_language_pack_manager()

            # Define progress callback that emits events
            def progress_callback(progress):
                # Convert dataclass to dict
                progress_dict = asdict(progress)

                # Emit language_download_progress event
                self.send_event("language_download_progress", progress_dict)

                # Also log progress
                if progress.phase == "downloading":
                    logger.info(f"{progress.message} ({progress.progress_percent:.0f}%)")
                elif progress.phase == "complete":
                    logger.info(f"Successfully installed: {progress.language_name}")
                elif progress.phase == "error":
                    logger.error(f"Download error: {progress.error}")

            # Trigger download via PaddleOCR initialization
            success = manager.trigger_language_download(
                language_code,
                version=version,
                progress_callback=progress_callback
            )

            if success:
                # Send final result
                self.send_result({
                    'success': True,
                    'language_code': language_code,
                    'message': f'Successfully installed language pack: {language_code}'
                })
            else:
                self.send_error(f"Failed to install language pack: {language_code}")

        except Exception as e:
            error_msg = f"Failed to download language pack: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.send_error(error_msg)

    def run(self):
        """Main event loop - read commands from stdin"""
        logger.info("Python backend started, waiting for commands...")

        try:
            for line in sys.stdin:
                if not self.running:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    command = json.loads(line)
                    self.handle_command(command)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    self.send_error(f"Invalid JSON: {str(e)}")
                except Exception as e:
                    logger.error(f"Command handling error: {e}", exc_info=True)
                    self.send_error(str(e))

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            logger.info("Python backend shutting down")


if __name__ == "__main__":
    handler = IPCHandler()
    handler.run()
