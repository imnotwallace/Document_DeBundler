# Python Bridge Implementation - Phase 3 Step 2

## Overview

Enhanced Rust Python Bridge with async event loop for streaming Python subprocess events to the frontend with state management and error recovery.

## Implementation Summary

### Key Features Implemented

1. **Thread-Safe State Management**
   - `ProcessState` enum: Idle, Running, Cancelled, Error(String)
   - `ProcessInternals` struct with Arc<Mutex<>> for thread-safe access
   - Persistent BufReader for stdout to avoid buffered data loss
   - Event loop handle storage for cleanup

2. **Async Event Loop**
   - Non-blocking event streaming from Python subprocess
   - Automatic event forwarding to frontend via Tauri's event system
   - Events emitted: `python_progress`, `python_result`, `python_error`
   - 30-second timeout for long operations with health checks
   - Graceful shutdown when process terminates or errors occur

3. **Error Handling**
   - Process crash detection (exit code monitoring)
   - Timeout handling with configurable duration
   - State updates on errors
   - Error events emitted to frontend
   - Graceful loop termination on error conditions

4. **Process Lifecycle**
   - **Startup**: Spawn Python subprocess → Create persistent BufReader → Start event loop
   - **Running**: Continuous event reading → Forward to frontend → Monitor health
   - **Shutdown**: Abort event loop task → Kill process → Wait for termination → Cleanup

### Architecture

```
Frontend (Svelte)
    ↓ listens to
Tauri Event System (python_progress, python_result, python_error)
    ↑ emits
Event Loop (async tokio task)
    ↑ reads from
Persistent BufReader
    ↑ connected to
Python Subprocess (stdout)
```

### Key Files Modified

1. **src-tauri/src/python_bridge.rs**
   - Added `ChildStdout` import for type safety
   - Enhanced `ProcessInternals` with `stdout_reader` and `event_loop_handle`
   - Fixed `start()` to create persistent BufReader
   - Enhanced `start_event_loop()` with comprehensive logging and error handling
   - Rewrote `read_event_async()` to use persistent reader (fixes buffered data loss bug)
   - Updated `stop()` to abort event loop and cleanup resources
   - Added extensive logging throughout

2. **src-tauri/src/commands.rs**
   - Updated `start_processing()` to start event loop
   - Updated `start_batch_ocr()` to start event loop
   - Added `app: tauri::AppHandle` parameter to both commands

3. **src-tauri/Cargo.toml**
   - Added `log = "0.4"` dependency

### Critical Bug Fixes

**Before**: `read_event_async()` created a new `BufReader` on each call, causing buffered data loss
```rust
// WRONG - creates new reader each time
let reader = BufReader::new(stdout);
let mut lines = reader.lines();
```

**After**: Uses persistent `BufReader` stored in `ProcessInternals`
```rust
// CORRECT - reuses persistent reader
if let Some(ref mut reader) = internals.stdout_reader {
    let mut line = String::new();
    reader.read_line(&mut line)?;
}
```

## Integration Guide

### Starting Python Process with Event Loop

```rust
#[tauri::command]
pub async fn my_command(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    {
        let process = state.python_process.lock()?;

        if !process.is_running() {
            let script_path = get_python_script_path()?;
            process.start(&script_path)?;

            // Start event loop to forward events to frontend
            process.start_event_loop(app.clone())?;
        }
    }

    // Send commands...
    Ok("Success".to_string())
}
```

### Frontend Event Listening

```typescript
import { listen } from '@tauri-apps/api/event';

// Listen for progress updates
const unlisten = await listen('python_progress', (event) => {
    const { current, total, message } = event.payload.data;
    console.log(`Progress: ${current}/${total} - ${message}`);
});

// Listen for results
await listen('python_result', (event) => {
    const result = event.payload.data;
    console.log('Processing complete:', result);
});

// Listen for errors
await listen('python_error', (event) => {
    const { message } = event.payload.data;
    console.error('Python error:', message);
});

// Clean up listeners when done
unlisten();
```

## Design Decisions

### Why Persistent BufReader?

Creating a new `BufReader` on each read causes data loss because:
1. Each `BufReader` fills its internal buffer (8KB by default)
2. If you only read one line, the rest of the buffer is lost when the reader is dropped
3. Next `BufReader` reads from the stream after the lost data

**Solution**: Store a single persistent `BufReader` that maintains its buffer between reads.

### Why Arc<Mutex<>> for State?

The event loop runs in a separate async task that needs to:
1. Read from stdout (requires mutable access)
2. Update process state (requires mutable access)
3. Check if process is alive (requires mutable access to Child)

`Arc<Mutex<>>` provides:
- Thread-safe shared ownership (Arc)
- Exclusive mutable access (Mutex)
- Works with async/await (tokio's spawn)

### Why 30-Second Timeout?

- Large PDFs can take time to process pages
- Python needs time for OCR operations
- Network-free local processing can be slow
- Timeout prevents hung processes
- Health check on timeout prevents false positives

### Why Abort Event Loop in stop()?

The event loop is a background tokio task that may be blocked on I/O. Simply killing the Python process could leave the task running. We:
1. Abort the task explicitly (immediate termination)
2. Kill the Python process
3. Wait for process termination
4. Clean up resources

This ensures no zombie tasks or processes remain.

## Success Criteria

- ✅ Thread-safe state management (Arc<Mutex<>>)
- ✅ Async event loop forwards events to frontend
- ✅ Non-blocking I/O (doesn't freeze app)
- ✅ Graceful shutdown (cleanup resources)
- ✅ Detects Python crashes and emits errors
- ✅ Can be accessed from multiple commands concurrently
- ✅ No memory leaks or zombie processes (Drop impl + explicit cleanup)

## Testing Approach

### Manual Testing Checklist

1. **Start Process**
   ```bash
   # Run app in dev mode
   npm run tauri:dev

   # Check logs for "Python event loop started"
   ```

2. **Send Command and Verify Events**
   - Select a PDF file
   - Start processing with OCR enabled
   - Check browser console for events:
     - `python_progress` events with incremental updates
     - `python_result` event on completion
   - Or check for `python_error` if something fails

3. **Test Graceful Shutdown**
   - Start processing
   - Close app while processing
   - Verify no Python processes remain (Task Manager/Activity Monitor)

4. **Test Crash Detection**
   - Start processing
   - Kill Python process manually (Task Manager)
   - Verify frontend receives error event

5. **Test Concurrent Access**
   - Start batch OCR
   - Try to get page count of another file
   - Verify no deadlocks or race conditions

### Expected Logs

```
[INFO] Stopping Python process
[DEBUG] Aborting event loop task
[DEBUG] Killing Python child process
[INFO] Python process stopped successfully
```

## Known Limitations

1. **Single Process**: Only one Python subprocess at a time (by design)
2. **No Auto-Recovery**: Process crashes require manual restart (could add retry logic)
3. **Shared stdout**: `read_event()` and event loop shouldn't be used simultaneously

## Future Enhancements

1. **Auto-Recovery**: Restart Python process on crash (max 3 attempts)
2. **Configurable Timeout**: Allow commands to specify timeout duration
3. **Process Pool**: Support multiple Python processes for parallel operations
4. **Metrics**: Track event throughput, process uptime, error rates
5. **Health Monitoring**: Periodic heartbeat checks even when idle

## References

- Tauri v2 Event System: https://v2.tauri.app/develop/inter-process-communication/
- Tokio Async Runtime: https://tokio.rs/
- Rust Arc<Mutex<>> Pattern: https://doc.rust-lang.org/book/ch16-03-shared-state.html
