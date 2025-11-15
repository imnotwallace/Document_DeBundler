use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::sync::oneshot;
use tokio::time::timeout;
use uuid::Uuid;

/// Represents the current state of the Python process
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessState {
    /// Process is idle, not running
    Idle,
    /// Process is running and processing commands
    Running,
    /// Process has been cancelled by user
    Cancelled,
    /// Process encountered an error
    Error(String),
}

/// Command structure sent to Python process
#[derive(Debug, Serialize, Deserialize)]
pub struct PythonCommand {
    pub command: String,
    pub file_path: Option<String>,
    pub options: Option<serde_json::Value>,
    pub request_id: Option<String>,
}

/// Event structure received from Python process
/// Python sends events as: {"type": "progress|result|error", "data": {...}}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PythonEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: serde_json::Value,
    pub request_id: Option<String>,
}

/// Internal process state with thread-safe access
struct ProcessInternals {
    child: Option<Child>,
    state: ProcessState,
    event_loop_handle: Option<tokio::task::JoinHandle<()>>,
    pending_requests: HashMap<String, oneshot::Sender<PythonEvent>>,
    stdout_reader: Arc<Mutex<Option<BufReader<ChildStdout>>>>,
}

/// Python process manager with async event streaming support
pub struct PythonProcess {
    internals: Arc<Mutex<ProcessInternals>>,
}

impl PythonProcess {
    /// Creates a new Python process manager
    pub fn new() -> Self {
        PythonProcess {
            internals: Arc::new(Mutex::new(ProcessInternals {
                child: None,
                state: ProcessState::Idle,
                event_loop_handle: None,
                pending_requests: HashMap::new(),
                stdout_reader: Arc::new(Mutex::new(None)),
            })),
        }
    }

    /// Creates a lightweight clone that shares the same internal state
    /// Useful for avoiding holding locks across await points
    pub fn clone_ref(&self) -> Self {
        PythonProcess {
            internals: Arc::clone(&self.internals),
        }
    }

    /// Starts the Python backend process
    ///
    /// # Arguments
    /// * `python_script_path` - Path to the Python main.py script
    ///
    /// # Returns
    /// * `Ok(())` if process started successfully
    /// * `Err(String)` with error description if startup failed
    pub fn start(&self, python_script_path: &str) -> Result<(), String> {
        let mut internals = self
            .internals
            .lock()
            .map_err(|e| format!("Failed to acquire lock: {}", e))?;

        // Check if already running
        if let Some(ref mut child) = internals.child {
            if let Ok(None) = child.try_wait() {
                return Err("Python process already running".to_string());
            }
        }

        // Get Python executable from venv
        let python_exe = get_venv_python_path()?;

        // Spawn Python process with piped stdio
        let mut child = Command::new(&python_exe)
            .arg(python_script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Python process: {}", e))?;

        // Take stdout and create a persistent BufReader
        let stdout = child.stdout.take()
            .ok_or_else(|| "Failed to capture stdout".to_string())?;
        let reader = BufReader::new(stdout);

        // Spawn stderr monitoring task for debugging (using blocking I/O)
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                let reader = BufReader::new(stderr);

                for line in reader.lines() {
                    if let Ok(line) = line {
                        eprintln!("[Python stderr] {}", line);
                    }
                }
            });
        }

        internals.child = Some(child);
        *internals.stdout_reader.lock().unwrap() = Some(reader);
        internals.state = ProcessState::Running;

        Ok(())
    }

    /// Sends a command to the Python process via stdin
    ///
    /// # Arguments
    /// * `command` - The command structure to send
    ///
    /// # Returns
    /// * `Ok(())` if command was sent successfully
    /// * `Err(String)` if send failed
    pub fn send_command(&self, command: PythonCommand) -> Result<(), String> {
        let mut internals = self
            .internals
            .lock()
            .map_err(|e| format!("Failed to acquire lock: {}", e))?;

        if let Some(ref mut child) = internals.child {
            if let Some(ref mut stdin) = child.stdin {
                let command_json = serde_json::to_string(&command)
                    .map_err(|e| format!("Failed to serialize command: {}", e))?;

                writeln!(stdin, "{}", command_json)
                    .map_err(|e| format!("Failed to write to stdin: {}", e))?;

                stdin
                    .flush()
                    .map_err(|e| format!("Failed to flush stdin: {}", e))?;

                Ok(())
            } else {
                Err("Python process stdin not available".to_string())
            }
        } else {
            Err("Python process not started".to_string())
        }
    }

    /// Sends a command and waits for the response using request-response correlation
    ///
    /// # Arguments
    /// * `command` - The command structure to send (request_id will be set automatically)
    /// * `timeout_secs` - Timeout in seconds to wait for response
    ///
    /// # Returns
    /// * `Ok(PythonEvent)` if response received successfully
    /// * `Err(String)` if send failed, timeout occurred, or channel closed
    pub async fn send_command_and_wait(
        &self,
        mut command: PythonCommand,
        timeout_secs: u64
    ) -> Result<PythonEvent, String> {
        // Generate request ID
        let request_id = Uuid::new_v4().to_string();
        command.request_id = Some(request_id.clone());

        // Create oneshot channel
        let (tx, rx) = oneshot::channel();

        // Register before sending
        {
            let mut internals = self.internals.lock()
                .map_err(|e| format!("Failed to lock: {}", e))?;
            internals.pending_requests.insert(request_id.clone(), tx);
        }

        // Send command
        self.send_command(command)?;

        // Wait for response with timeout
        match timeout(Duration::from_secs(timeout_secs), rx).await {
            Ok(Ok(event)) => Ok(event),
            Ok(Err(_)) => Err("Channel closed".to_string()),
            Err(_) => {
                // Cleanup on timeout
                let mut internals = self.internals.lock()
                    .map_err(|e| format!("Failed to lock: {}", e))?;
                internals.pending_requests.remove(&request_id);
                Err(format!("Timeout after {} seconds waiting for response", timeout_secs))
            }
        }
    }

    /// Starts the async event loop for streaming events from Python
    ///
    /// This method spawns a background task that continuously reads events from
    /// Python's stdout and emits them to the frontend via Tauri's event system.
    ///
    /// Events are emitted as:
    /// - "python_progress" - Progress updates
    /// - "python_result" - Processing results
    /// - "python_error" - Error messages
    ///
    /// # Arguments
    /// * `app_handle` - Tauri app handle for emitting events
    ///
    /// # Returns
    /// * `Ok(())` if event loop started successfully
    /// * `Err(String)` if startup failed
    pub fn start_event_loop(&self, app_handle: AppHandle) -> Result<(), String> {
        // Check if event loop is already running
        {
            let internals = self.internals.lock()
                .map_err(|e| format!("Failed to lock process: {}", e))?;

            if internals.event_loop_handle.is_some() {
                eprintln!("[INFO] Python event loop already running, skipping start");
                return Ok(());
            }
        }

        let internals_clone = Arc::clone(&self.internals);

        // Spawn async task for event streaming
        let handle = tokio::spawn(async move {
            eprintln!("[INFO] Python event loop started");

            loop {
                // Check process state
                let should_continue = {
                    let internals = match internals_clone.lock() {
                        Ok(i) => i,
                        Err(e) => {
                            eprintln!("Failed to acquire lock in event loop: {}", e);
                            break;
                        }
                    };

                    match internals.state {
                        ProcessState::Running => true,
                        ProcessState::Cancelled => {
                            eprintln!("Event loop stopped: Processing cancelled");
                            let _ = app_handle.emit(
                                "python_error",
                                PythonEvent {
                                    event_type: "error".to_string(),
                                    data: serde_json::json!({
                                        "message": "Processing cancelled by user"
                                    }),
                                    request_id: None,
                                },
                            );
                            false
                        }
                        ProcessState::Error(ref msg) => {
                            eprintln!("Event loop stopped: Error - {}", msg);
                            let _ = app_handle.emit(
                                "python_error",
                                PythonEvent {
                                    event_type: "error".to_string(),
                                    data: serde_json::json!({
                                        "message": msg.clone()
                                    }),
                                    request_id: None,
                                },
                            );
                            false
                        }
                        ProcessState::Idle => {
                            eprintln!("Event loop stopped: Process idle");
                            false
                        }
                    }
                };

                if !should_continue {
                    break;
                }

                // Read event with timeout (30s for long operations)
                let event_result =
                    timeout(Duration::from_secs(30), Self::read_event_async(&internals_clone))
                        .await;

                match event_result {
                    Ok(Ok(Some(event))) => {
                        eprintln!("Received event: {} - {:?}", event.event_type, event.data);
                        if let Some(ref req_id) = event.request_id {
                            eprintln!("[DEBUG] Event has request_id: {}", req_id);
                        } else {
                            eprintln!("[DEBUG] Event has NO request_id");
                        }
                        eprintln!("[DEBUG] Full event JSON: {}", serde_json::to_string(&event).unwrap_or_else(|_| "failed to serialize".to_string()));

                        // Route event based on type and request_id
                        match event.event_type.as_str() {
                            "progress" => {
                                // Broadcast progress to frontend
                                let _ = app_handle.emit("python_progress", event.clone());
                            }
                            "language_download_progress" => {
                                // Broadcast language download progress to frontend
                                let _ = app_handle.emit("language-download-progress", event.clone());
                            }
                            "file_status" => {
                                // Broadcast file status to frontend
                                let _ = app_handle.emit("python_file_status", event.clone());
                            }
                            "result" | "error" => {
                                // Check if this is a response to a pending request
                                if let Some(ref request_id) = event.request_id {
                                    eprintln!("[DEBUG] Event has request_id: {}", request_id);
                                    let mut internals_for_map = match internals_clone.lock() {
                                        Ok(i) => i,
                                        Err(e) => {
                                            eprintln!("Failed to acquire lock for pending requests: {}", e);
                                            continue;
                                        }
                                    };

                                    eprintln!("[DEBUG] Pending requests: {:?}", internals_for_map.pending_requests.keys().collect::<Vec<_>>());
                                    if let Some(sender) = internals_for_map.pending_requests.remove(request_id) {
                                        // Send to waiting command handler
                                        eprintln!("[DEBUG] Found matching pending request, sending to handler");
                                        let _ = sender.send(event.clone());
                                        continue;
                                    } else {
                                        eprintln!("[DEBUG] No matching pending request found for request_id: {}", request_id);
                                    }
                                } else {
                                    eprintln!("[DEBUG] Event has NO request_id");
                                }

                                // Fallback: emit to frontend if no pending request
                                // (for backwards compatibility with old-style direct reading)
                                let emit_result = if event.event_type == "result" {
                                    app_handle.emit("python_result", event.clone())
                                } else {
                                    app_handle.emit("python_error", event.clone())
                                };

                                if let Err(e) = emit_result {
                                    eprintln!("Failed to emit event: {}", e);
                                }

                                // If error event, update state and stop loop
                                if event.event_type == "error" {
                                    if let Ok(mut internals) = internals_clone.lock() {
                                        let error_msg = event.data["message"]
                                            .as_str()
                                            .unwrap_or("Unknown error")
                                            .to_string();
                                        internals.state = ProcessState::Error(error_msg);
                                    }
                                    break;
                                }
                            }
                            _ => {
                                eprintln!("Unknown event type: {}", event.event_type);
                            }
                        }
                    }
                    Ok(Ok(None)) => {
                        // EOF - process terminated
                        eprintln!("Python process stdout closed (EOF)");
                        if let Ok(mut internals) = internals_clone.lock() {
                            internals.state = ProcessState::Idle;
                        }
                        break;
                    }
                    Ok(Err(e)) => {
                        // Read error
                        eprintln!("Error reading event: {}", e);
                        let error_msg = e.clone();
                        if let Ok(mut internals) = internals_clone.lock() {
                            internals.state = ProcessState::Error(e);
                        }
                        let _ = app_handle.emit(
                            "python_error",
                            PythonEvent {
                                event_type: "error".to_string(),
                                data: serde_json::json!({"message": error_msg}),
                                request_id: None,
                            },
                        );
                        break;
                    }
                    Err(_) => {
                        // Timeout - check if process is still alive
                        eprintln!("Event read timeout, checking process health");
                        let is_alive = {
                            if let Ok(mut internals) = internals_clone.lock() {
                                if let Some(ref mut child) = internals.child {
                                    match child.try_wait() {
                                        Ok(None) => true, // Still running
                                        Ok(Some(status)) => {
                                            // Process exited
                                            let msg = format!("Process exited with status: {}", status);
                                            eprintln!("{}", msg);
                                            internals.state = ProcessState::Error(msg);
                                            false
                                        }
                                        Err(e) => {
                                            let msg = format!("Process error: {}", e);
                                            eprintln!("{}", msg);
                                            internals.state = ProcessState::Error(msg);
                                            false
                                        }
                                    }
                                } else {
                                    eprintln!("No child process found");
                                    false
                                }
                            } else {
                                eprintln!("Failed to acquire lock for health check");
                                false
                            }
                        };

                        if !is_alive {
                            let _ = app_handle.emit(
                                "python_error",
                                PythonEvent {
                                    event_type: "error".to_string(),
                                    data: serde_json::json!({
                                        "message": "Python process terminated unexpectedly"
                                    }),
                                    request_id: None,
                                },
                            );
                            break;
                        }
                    }
                }

                // Small delay to prevent tight loop
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            // CRITICAL: Clear event loop handle when loop exits to break reference cycle
            if let Ok(mut internals) = internals_clone.lock() {
                internals.event_loop_handle = None;
                eprintln!("[INFO] Event loop handle cleared on exit");
            }

            eprintln!("Python event loop terminated");
        });

        // Store the handle
        {
            let mut internals = self.internals.lock()
                .map_err(|e| format!("Failed to lock internals: {}", e))?;
            internals.event_loop_handle = Some(handle);
        }

        Ok(())
    }

    /// Async helper to read a single event from stdout using the persistent BufReader
    async fn read_event_async(
        internals: &Arc<Mutex<ProcessInternals>>,
    ) -> Result<Option<PythonEvent>, String> {
        // Extract stdout_reader Arc WITHOUT holding the main lock
        let stdout_reader_arc = {
            let internals = internals
                .lock()
                .map_err(|e| format!("Failed to acquire lock: {}", e))?;
            Arc::clone(&internals.stdout_reader)
        };

        // Now do the blocking read holding ONLY the stdout_reader lock, not the main internals lock
        tokio::task::spawn_blocking(move || {
            let mut stdout_guard = stdout_reader_arc
                .lock()
                .map_err(|e| format!("Failed to acquire stdout lock: {}", e))?;

            // Use the persistent BufReader
            if let Some(ref mut reader) = *stdout_guard {
                let mut line = String::new();

                match reader.read_line(&mut line) {
                    Ok(0) => Ok(None), // EOF
                    Ok(_) => {
                        if line.trim().is_empty() {
                            return Ok(None);
                        }

                        let event: PythonEvent = serde_json::from_str(&line)
                            .map_err(|e| format!("Failed to parse event JSON '{}': {}", line.trim(), e))?;
                        Ok(Some(event))
                    }
                    Err(e) => Err(format!("Failed to read line: {}", e)),
                }
            } else {
                Err("Python process stdout reader not available".to_string())
            }
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    /// Gets the current process state
    ///
    /// # Returns
    /// * Current `ProcessState`
    pub fn get_state(&self) -> ProcessState {
        self.internals
            .lock()
            .map(|i| i.state.clone())
            .unwrap_or(ProcessState::Error("Failed to acquire lock".to_string()))
    }

    /// Checks if the Python process is alive
    ///
    /// # Returns
    /// * `true` if process is running
    /// * `false` if process is not started or has exited
    pub fn is_alive(&self) -> bool {
        if let Ok(mut internals) = self.internals.lock() {
            if let Some(ref mut child) = internals.child {
                match child.try_wait() {
                    Ok(None) => return true, // Still running
                    Ok(Some(_)) => {
                        internals.state = ProcessState::Idle;
                        return false;
                    }
                    Err(_) => return false,
                }
            }
        }
        false
    }

    /// Checks if the Python process is running
    /// Alias for is_alive() for compatibility
    ///
    /// # Returns
    /// * `true` if process is running
    /// * `false` if process is not started or has exited
    pub fn is_running(&self) -> bool {
        self.is_alive()
    }

    /// Reads a single event from Python's stdout (synchronous)
    ///
    /// NOTE: This method should not be used when the event loop is active,
    /// as both would be reading from the same stdout stream.
    ///
    /// # Returns
    /// * `Ok(Some(event))` if event was read successfully
    /// * `Ok(None)` if no event available or EOF
    /// * `Err(String)` if read failed
    pub fn read_event(&self) -> Result<Option<PythonEvent>, String> {
        let stdout_reader_arc = {
            let internals = self
                .internals
                .lock()
                .map_err(|e| format!("Failed to acquire lock: {}", e))?;
            Arc::clone(&internals.stdout_reader)
        };

        // Use the persistent BufReader
        let mut stdout_guard = stdout_reader_arc
            .lock()
            .map_err(|e| format!("Failed to acquire stdout lock: {}", e))?;

        if let Some(ref mut reader) = *stdout_guard {
            let mut line = String::new();

            match reader.read_line(&mut line) {
                Ok(0) => Ok(None), // EOF
                Ok(_) => {
                    if line.trim().is_empty() {
                        return Ok(None);
                    }

                    let event: PythonEvent = serde_json::from_str(&line)
                        .map_err(|e| format!("Failed to parse event: {}", e))?;
                    Ok(Some(event))
                }
                Err(e) => Err(format!("Failed to read from stdout: {}", e)),
            }
        } else {
            Err("Python process stdout reader not available".to_string())
        }
    }

    /// Marks the process as cancelled
    ///
    /// This updates the state to `Cancelled` which will cause the event loop
    /// to terminate gracefully. Also clears all pending requests.
    pub fn set_cancelled(&self) -> Result<(), String> {
        let mut internals = self
            .internals
            .lock()
            .map_err(|e| format!("Failed to acquire lock: {}", e))?;

        internals.state = ProcessState::Cancelled;

        // CRITICAL: Clear all pending requests to prevent memory leak
        // Send cancellation error to all waiting handlers
        let pending_count = internals.pending_requests.len();
        if pending_count > 0 {
            eprintln!("[INFO] Clearing {} pending requests due to cancellation", pending_count);
            for (request_id, _sender) in internals.pending_requests.drain() {
                eprintln!("[DEBUG] Dropped pending request: {}", request_id);
                // oneshot::Sender is dropped here, receivers will get Err(RecvError)
            }
        }

        Ok(())
    }

    /// Stops the Python process
    ///
    /// This gracefully shuts down the event loop, kills the process, and waits for it to terminate.
    ///
    /// # Returns
    /// * `Ok(())` if process stopped successfully
    /// * `Err(String)` if stop failed
    pub fn stop(&self) -> Result<(), String> {
        eprintln!("Stopping Python process");

        // First, abort the event loop if it's running
        {
            let mut internals = self
                .internals
                .lock()
                .map_err(|e| format!("Failed to acquire lock: {}", e))?;

            if let Some(handle) = internals.event_loop_handle.take() {
                eprintln!("Aborting event loop task");
                handle.abort();
            }

            // CRITICAL: Clear all pending requests to prevent memory leak
            let pending_count = internals.pending_requests.len();
            if pending_count > 0 {
                eprintln!("[INFO] Clearing {} pending requests during stop", pending_count);
                for (request_id, _sender) in internals.pending_requests.drain() {
                    eprintln!("[DEBUG] Dropped pending request: {}", request_id);
                }
            }

            // Kill the child process
            if let Some(mut child) = internals.child.take() {
                eprintln!("Killing Python child process");
                child
                    .kill()
                    .map_err(|e| format!("Failed to kill Python process: {}", e))?;
                child
                    .wait()
                    .map_err(|e| format!("Failed to wait for Python process: {}", e))?;
            }

            // Clean up reader
            *internals.stdout_reader.lock().unwrap() = None;
            internals.state = ProcessState::Idle;
        }

        eprintln!("Python process stopped successfully");
        Ok(())
    }
}

impl Drop for PythonProcess {
    fn drop(&mut self) {
        // Only stop the process if this is the last reference to it
        // This prevents stopping the process when temporary clones go out of scope
        if Arc::strong_count(&self.internals) == 1 {
            eprintln!("[INFO] Last PythonProcess reference dropped, stopping process");
            let _ = self.stop();
        } else {
            eprintln!("[DEBUG] PythonProcess reference dropped, but {} other references remain",
                Arc::strong_count(&self.internals) - 1);
        }
    }
}

impl Default for PythonProcess {
    fn default() -> Self {
        Self::new()
    }
}

/// Gets the path to the Python executable in the virtual environment
fn get_venv_python_path() -> Result<String, String> {
    #[cfg(debug_assertions)]
    {
        let current_dir = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;

        let project_root = current_dir.parent()
            .ok_or_else(|| "Failed to get parent directory".to_string())?;

        // Platform-specific venv path
        #[cfg(target_os = "windows")]
        let venv_python = project_root
            .join("python-backend")
            .join(".venv")
            .join("Scripts")
            .join("python.exe");

        #[cfg(not(target_os = "windows"))]
        let venv_python = project_root
            .join("python-backend")
            .join(".venv")
            .join("bin")
            .join("python");

        if !venv_python.exists() {
            return Err(format!(
                "Virtual environment Python not found at: {}. Run setup.bat/setup.sh first.",
                venv_python.display()
            ));
        }

        Ok(venv_python.to_string_lossy().to_string())
    }

    #[cfg(not(debug_assertions))]
    {
        // Production: Python should be bundled with the app
        Err("Production Python bundling not yet implemented".to_string())
    }
}
