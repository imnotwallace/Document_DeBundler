"""
Document De-Bundler - Python Backend
Main entry point for PDF processing via stdin/stdout IPC with Tauri frontend
"""

import sys
import json
import logging
from typing import Dict, Any

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

    def send_event(self, event_type: str, data: Any):
        """Send an event to the frontend via stdout"""
        event = {
            "type": event_type,
            "data": data
        }
        print(json.dumps(event), flush=True)

    def send_progress(self, current: int, total: int, message: str):
        """Send progress update"""
        self.send_event("progress", {
            "current": current,
            "total": total,
            "message": message,
            "percent": (current / total * 100) if total > 0 else 0
        })

    def send_result(self, result: Any):
        """Send processing result"""
        self.send_event("result", result)

    def send_error(self, error_message: str):
        """Send error message"""
        self.send_event("error", {"message": error_message})

    def handle_command(self, command: Dict[str, Any]):
        """Process incoming command"""
        cmd_type = command.get("command")

        if cmd_type == "analyze":
            self.handle_analyze(command)
        elif cmd_type == "process":
            self.handle_process(command)
        elif cmd_type == "cancel":
            self.handle_cancel()
        else:
            self.send_error(f"Unknown command: {cmd_type}")

    def handle_analyze(self, command: Dict[str, Any]):
        """Analyze PDF structure"""
        file_path = command.get("file_path")
        logger.info(f"Analyzing PDF: {file_path}")

        try:
            # TODO: Implement PDF analysis
            self.send_progress(0, 100, "Starting analysis...")
            self.send_progress(50, 100, "Analyzing structure...")
            self.send_progress(100, 100, "Analysis complete")

            self.send_result({
                "status": "success",
                "message": "Analysis placeholder - not yet implemented"
            })
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            self.send_error(str(e))

    def handle_process(self, command: Dict[str, Any]):
        """Process PDF with splitting and OCR"""
        file_path = command.get("file_path")
        options = command.get("options", {})
        logger.info(f"Processing PDF: {file_path} with options: {options}")

        try:
            # TODO: Implement PDF processing
            self.send_progress(0, 100, "Starting processing...")
            self.send_progress(100, 100, "Processing complete")

            self.send_result({
                "status": "success",
                "message": "Processing placeholder - not yet implemented"
            })
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.send_error(str(e))

    def handle_cancel(self):
        """Cancel current operation"""
        logger.info("Cancel requested")
        self.running = False
        self.send_result({"status": "cancelled"})

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
