"""
Test script for OCR batch IPC handler
Tests the handle_ocr_batch command routing and validation
"""

import json
import sys
from io import StringIO
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import IPCHandler


def test_command_routing():
    """Test that ocr_batch command is properly routed"""
    handler = IPCHandler()

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Test routing to ocr_batch handler
        command = {
            "command": "ocr_batch",
            "files": [],  # Will trigger validation error
            "output_dir": "test_output"
        }

        handler.handle_command(command)

        # Get output
        output = sys.stdout.getvalue()

        # Should contain error event
        assert output, "Expected output from handler"

        # Parse JSON output
        lines = [line for line in output.strip().split('\n') if line]
        assert len(lines) > 0, "Expected at least one event"

        event = json.loads(lines[0])
        assert event["type"] == "error", f"Expected error event, got {event['type']}"
        assert "No files provided" in event["data"]["message"], \
            f"Expected validation error, got: {event['data']['message']}"

        print("PASS: Command routing works correctly", file=sys.stderr)

    finally:
        sys.stdout = old_stdout


def test_missing_output_dir():
    """Test validation of missing output_dir"""
    handler = IPCHandler()

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        command = {
            "command": "ocr_batch",
            "files": ["test.pdf"],
            # Missing output_dir
        }

        handler.handle_command(command)

        # Get output
        output = sys.stdout.getvalue()

        # Parse JSON output
        lines = [line for line in output.strip().split('\n') if line]
        event = json.loads(lines[0])

        assert event["type"] == "error", f"Expected error event, got {event['type']}"
        assert "No output directory" in event["data"]["message"], \
            f"Expected output_dir error, got: {event['data']['message']}"

        print("PASS: Output directory validation works", file=sys.stderr)

    finally:
        sys.stdout = old_stdout


def test_unknown_command():
    """Test handling of unknown commands"""
    handler = IPCHandler()

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        command = {
            "command": "unknown_command"
        }

        handler.handle_command(command)

        # Get output
        output = sys.stdout.getvalue()

        # Parse JSON output
        lines = [line for line in output.strip().split('\n') if line]
        event = json.loads(lines[0])

        assert event["type"] == "error", f"Expected error event, got {event['type']}"
        assert "Unknown command" in event["data"]["message"], \
            f"Expected unknown command error, got: {event['data']['message']}"

        print("PASS: Unknown command handling works", file=sys.stderr)

    finally:
        sys.stdout = old_stdout


def test_cancellation_flag():
    """Test cancellation flag mechanism"""
    handler = IPCHandler()

    # Initial state
    assert handler.cancelled == False, "Initial cancelled should be False"
    assert handler.running == True, "Initial running should be True"

    # Trigger cancellation
    handler.handle_cancel()

    assert handler.cancelled == True, "Cancelled should be True after cancel"
    assert handler.running == False, "Running should be False after cancel"

    print("PASS: Cancellation flag works correctly", file=sys.stderr)


def test_send_progress_with_percent():
    """Test send_progress with explicit percent parameter"""
    handler = IPCHandler()

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Test with explicit percent
        handler.send_progress(10, 100, "Test message", percent=25.5)

        output = sys.stdout.getvalue()
        event = json.loads(output.strip())

        assert event["type"] == "progress", f"Expected progress event, got {event['type']}"
        assert event["data"]["current"] == 10
        assert event["data"]["total"] == 100
        assert event["data"]["message"] == "Test message"
        assert event["data"]["percent"] == 25.5, \
            f"Expected percent=25.5, got {event['data']['percent']}"

        print("PASS: send_progress with explicit percent works", file=sys.stderr)

    finally:
        sys.stdout = old_stdout


def test_send_progress_auto_calculate():
    """Test send_progress with auto-calculated percent"""
    handler = IPCHandler()

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Test without percent (auto-calculate)
        handler.send_progress(25, 100, "Test message")

        output = sys.stdout.getvalue()
        event = json.loads(output.strip())

        assert event["type"] == "progress", f"Expected progress event, got {event['type']}"
        assert event["data"]["percent"] == 25.0, \
            f"Expected auto-calculated percent=25.0, got {event['data']['percent']}"

        print("PASS: send_progress auto-calculation works", file=sys.stderr)

    finally:
        sys.stdout = old_stdout


if __name__ == "__main__":
    print("Testing OCR Batch IPC Handler...\n", file=sys.stderr)

    try:
        test_command_routing()
        test_missing_output_dir()
        test_unknown_command()
        test_cancellation_flag()
        test_send_progress_with_percent()
        test_send_progress_auto_calculate()

        print("\nAll tests passed successfully!", file=sys.stderr)
        sys.exit(0)

    except AssertionError as e:
        print(f"\nTest failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
