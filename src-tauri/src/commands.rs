use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::State;
use serde_json;

use crate::python_bridge::{PythonProcess, PythonCommand};

/// Application state holding the Python process and processing status
pub struct AppState {
    pub python_process: Arc<Mutex<PythonProcess>>,
    pub status: Arc<Mutex<ProcessingStatusInternal>>,
}

/// Internal processing status
#[derive(Debug, Clone)]
pub struct ProcessingStatusInternal {
    pub is_running: bool,
    pub current_page: u32,
    pub total_pages: u32,
    pub current_operation: String,
    pub progress_percent: f32,
}

impl Default for ProcessingStatusInternal {
    fn default() -> Self {
        Self {
            is_running: false,
            current_page: 0,
            total_pages: 0,
            current_operation: "Idle".to_string(),
            progress_percent: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub name: String,
    pub size: u64,
    pub page_count: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessingStatus {
    pub is_running: bool,
    pub current_page: u32,
    pub total_pages: u32,
    pub current_operation: String,
    pub progress_percent: f32,
}

// ===== Language Pack Types =====

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LanguageInfo {
    pub code: String,
    pub name: String,
    pub installed: bool,
    pub script_name: String,  // e.g., "latin", "arabic", "cyrillic"
    pub script_description: String,  // Description of what languages use this script
    pub total_size_mb: f32,
    pub detection_installed: bool,
    pub recognition_installed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LanguageListResponse {
    pub languages: Vec<LanguageInfo>,
    pub count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstalledLanguagesResponse {
    pub installed_languages: Vec<String>,
    pub count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DownloadResponse {
    pub success: bool,
    pub language_code: String,
    pub message: String,
}

/// Opens a file dialog for selecting a PDF file
#[tauri::command]
pub async fn select_pdf_file(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    // In Tauri v2, we use the dialog plugin
    let file_path = app.dialog()
        .file()
        .add_filter("PDF Files", &["pdf"])
        .set_title("Select PDF Document")
        .blocking_pick_file();

    Ok(file_path.map(|p| p.to_string()))
}

/// Opens a file dialog for selecting multiple PDF files
#[tauri::command]
pub async fn select_multiple_pdf_files(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let file_paths = app.dialog()
        .file()
        .add_filter("PDF Files", &["pdf"])
        .set_title("Select PDF Documents")
        .blocking_pick_files();

    match file_paths {
        Some(paths) => Ok(paths.into_iter().map(|p| p.to_string()).collect()),
        None => Ok(Vec::new()),
    }
}

/// Opens a folder selection dialog
#[tauri::command]
pub async fn select_folder(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let folder_path = app.dialog()
        .file()
        .set_title("Select Destination Folder")
        .blocking_pick_folder();

    Ok(folder_path.map(|p| p.to_string()))
}

/// Gets information about the selected file
#[tauri::command]
pub async fn get_file_info(file_path: String, app: tauri::AppHandle, state: State<'_, AppState>) -> Result<FileInfo, String> {
    let path = PathBuf::from(&file_path);

    if !path.exists() {
        return Err("File does not exist".to_string());
    }

    let metadata = std::fs::metadata(&path)
        .map_err(|e| format!("Failed to read file metadata: {}", e))?;

    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Try to get page count via Python
    let page_count = get_pdf_page_count(file_path.clone(), app, state).await.ok().flatten();

    Ok(FileInfo {
        path: file_path.clone(),
        name,
        size: metadata.len(),
        page_count,
    })
}

/// Gets the page count of a PDF file via Python backend
#[tauri::command]
pub async fn get_pdf_page_count(file_path: String, app: tauri::AppHandle, state: State<'_, AppState>) -> Result<Option<u32>, String> {
    use std::time::Duration;
    use tokio::time::timeout;

    // Validate file path
    let path = PathBuf::from(&file_path);
    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }
    if !file_path.to_lowercase().ends_with(".pdf") {
        return Err("File must be a PDF".to_string());
    }

    // Ensure Python process is started
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            // Start Python process
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        }
    }

    // Send analyze command
    let command = PythonCommand {
        command: "analyze".to_string(),
        file_path: Some(file_path.clone()),
        options: None,
        request_id: None,
    };

    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.send_command(command)?;
    }

    // Wait for result with timeout (30 seconds should be enough for analysis)
    let result = timeout(Duration::from_secs(30), async {
        loop {
            let event = {
                let process = state.python_process.lock()
                    .map_err(|e| format!("Failed to lock process: {}", e))?;
                process.read_event()?
            };

            if let Some(evt) = event {
                match evt.event_type.as_str() {
                    "result" => {
                        // Extract page count from result
                        if let Some(total_pages) = evt.data.get("total_pages") {
                            if let Some(count) = total_pages.as_u64() {
                                return Ok(Some(count as u32));
                            }
                        }
                        return Ok(None);
                    }
                    "error" => {
                        let msg = evt.data.get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown error");
                        return Err(format!("Python error: {}", msg));
                    }
                    _ => {
                        // Ignore progress events
                        continue;
                    }
                }
            }

            // Small delay to prevent busy waiting
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }).await;

    match result {
        Ok(Ok(page_count)) => Ok(page_count),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Timeout waiting for page count".to_string()),
    }
}

/// Starts PDF processing via Python backend
#[tauri::command]
pub async fn start_processing(
    file_path: String,
    enable_ocr: bool,
    output_dir: Option<String>,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    // Ensure Python process is started
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        }
    }  // Lock drops here

    // Start event loop in separate scope to avoid deadlock
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        if process.is_running() {
            process.start_event_loop(app.clone())?;
        }
    }

    // Build options
    let mut options = serde_json::Map::new();
    options.insert("force_ocr".to_string(), serde_json::Value::Bool(enable_ocr));

    if let Some(dir) = output_dir {
        options.insert("output_dir".to_string(), serde_json::Value::String(dir));
    }

    // Send process command
    let command = PythonCommand {
        command: "process".to_string(),
        file_path: Some(file_path.clone()),
        options: Some(serde_json::Value::Object(options)),
        request_id: None,
    };

    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.send_command(command)?;
    }

    // Update status
    {
        let mut status = state.status.lock()
            .map_err(|e| format!("Failed to lock status: {}", e))?;
        status.is_running = true;
        status.current_operation = "Starting processing...".to_string();
        status.progress_percent = 0.0;
    }

    Ok(format!("Processing started for: {}", file_path))
}

/// Starts batch OCR processing for multiple files
#[tauri::command]
pub async fn start_batch_ocr(
    files: Vec<String>,
    destination: String,
    ocr_config: Option<serde_json::Value>,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    if files.is_empty() {
        return Err("No files provided".to_string());
    }

    // Validate all file paths exist and are PDFs
    for file in &files {
        let path = PathBuf::from(file);
        if !path.exists() {
            return Err(format!("File not found: {}", file));
        }
        if !file.to_lowercase().ends_with(".pdf") {
            return Err(format!("File must be a PDF: {}", file));
        }
    }

    // Validate destination directory exists
    let dest_path = PathBuf::from(&destination);
    if !dest_path.exists() {
        return Err(format!("Destination directory not found: {}", destination));
    }
    if !dest_path.is_dir() {
        return Err(format!("Destination must be a directory: {}", destination));
    }

    // Check if already processing
    {
        let status = state.status.lock()
            .map_err(|e| format!("Failed to lock status: {}", e))?;
        if status.is_running {
            return Err("Already processing. Please wait or cancel current operation.".to_string());
        }
    }

    // Ensure Python process is started
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        }
    }  // Lock drops here

    // Start event loop in separate scope to avoid deadlock
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        if process.is_running() {
            process.start_event_loop(app.clone())?;
        }
    }

    // Build options for batch processing
    let mut options = serde_json::Map::new();
    options.insert("files".to_string(), serde_json::Value::Array(
        files.iter().map(|f| serde_json::Value::String(f.clone())).collect()
    ));
    options.insert("output_dir".to_string(), serde_json::Value::String(destination.clone()));

    // Add OCR configuration if provided
    if let Some(config) = ocr_config {
        options.insert("ocr_config".to_string(), config);
    }

    // Send batch command
    let command = PythonCommand {
        command: "ocr_batch".to_string(),
        file_path: None,
        options: Some(serde_json::Value::Object(options)),
        request_id: None,
    };

    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.send_command(command)?;
    }

    // Update status
    {
        let mut status = state.status.lock()
            .map_err(|e| format!("Failed to lock status: {}", e))?;
        status.is_running = true;
        status.total_pages = files.len() as u32;
        status.current_operation = "Starting batch OCR...".to_string();
        status.progress_percent = 0.0;
    }

    Ok(format!("Batch OCR started for {} files", files.len()))
}

/// Cancels the current processing operation
#[tauri::command]
pub async fn cancel_processing(state: State<'_, AppState>) -> Result<String, String> {
    // Check if currently processing
    {
        let status = state.status.lock()
            .map_err(|e| format!("Failed to lock status: {}", e))?;
        if !status.is_running {
            return Err("No processing operation in progress".to_string());
        }
    }

    // Send cancel command to Python
    let command = PythonCommand {
        command: "cancel".to_string(),
        file_path: None,
        options: None,
        request_id: None,
    };

    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if process.is_running() {
            process.send_command(command)?;
        }
    }

    // Update status
    {
        let mut status = state.status.lock()
            .map_err(|e| format!("Failed to lock status: {}", e))?;
        status.is_running = false;
        status.current_operation = "Cancelled".to_string();
    }

    Ok("Cancellation requested".to_string())
}

/// Cancels batch OCR operation (alias for cancel_processing)
#[tauri::command]
pub async fn cancel_batch_ocr(state: State<'_, AppState>) -> Result<String, String> {
    cancel_processing(state).await
}

/// Gets the current processing status
#[tauri::command]
pub async fn get_processing_status(state: State<'_, AppState>) -> Result<ProcessingStatus, String> {
    let status = state.status.lock()
        .map_err(|e| format!("Failed to lock status: {}", e))?;

    Ok(ProcessingStatus {
        is_running: status.is_running,
        current_page: status.current_page,
        total_pages: status.total_pages,
        current_operation: status.current_operation.clone(),
        progress_percent: status.progress_percent,
    })
}

/// Quits the application
#[tauri::command]
pub fn quit_app(app: tauri::AppHandle) -> Result<(), String> {
    app.exit(0);
    Ok(())
}

/// Helper function to get the Python script path
/// In development: use python-backend/main.py (navigate from src-tauri to project root)
/// In production: use bundled script in resources
fn get_python_script_path(_app: &tauri::AppHandle) -> Result<String, String> {
    // For development, use the local python-backend directory
    #[cfg(debug_assertions)]
    {
        let current_dir = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;

        // In dev mode, current_dir is src-tauri/, so navigate up to project root
        let project_root = current_dir.parent()
            .ok_or_else(|| "Failed to get parent directory".to_string())?;

        let script_path = project_root.join("python-backend").join("main.py");

        if !script_path.exists() {
            return Err(format!(
                "Python script not found at: {}. Current dir: {}, Project root: {}",
                script_path.display(),
                current_dir.display(),
                project_root.display()
            ));
        }

        Ok(script_path.to_string_lossy().to_string())
    }

    // For production, use the bundled resource
    #[cfg(not(debug_assertions))]
    {
        // Use Tauri's resource resolver for production builds
        let resource_path = _app.path()
            .resolve("python-backend/main.py", tauri::path::BaseDirectory::Resource)
            .map_err(|e| format!("Failed to resolve resource path: {}", e))?;

        if !resource_path.exists() {
            return Err(format!(
                "Python script not found in resources: {}",
                resource_path.display()
            ));
        }

        Ok(resource_path.to_string_lossy().to_string())
    }
}

/// Get hardware capabilities for OCR configuration
#[tauri::command]
pub async fn get_hardware_capabilities(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // Ensure Python process is started
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        }
    }

    // Start event loop
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        if process.is_running() {
            process.start_event_loop(app.clone())?;
        }
    }

    // Send command and wait for response
    let command = PythonCommand {
        command: "get_hardware_capabilities".to_string(),
        file_path: None,
        options: None,
        request_id: None, // Will be set by send_command_and_wait
    };

    // Clone the process to avoid holding lock across await
    let temp_process = {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.clone_ref()
    };

    let event = temp_process.send_command_and_wait(command, 10).await?;

    match event.event_type.as_str() {
        "result" => {
            // Return the hardware capabilities data directly
            Ok(event.data)
        }
        "error" => {
            let msg = event.data.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            Err(format!("Python error: {}", msg))
        }
        _ => Err(format!("Unexpected event type: {}", event.event_type))
    }
}

// ===== Language Pack Commands =====

/// List all available language packs with installation status
#[tauri::command]
pub async fn list_available_languages(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<LanguageListResponse, String> {
    // Ensure Python process is started
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        }
    }

    // Start event loop
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        if process.is_running() {
            process.start_event_loop(app.clone())?;
        }
    }

    // Send command and wait for response
    let command = PythonCommand {
        command: "list_available_languages".to_string(),
        file_path: None,
        options: None,
        request_id: None, // Will be set by send_command_and_wait
    };

    // Clone the process to avoid holding lock across await
    let temp_process = {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.clone_ref()
    };

    let event = temp_process.send_command_and_wait(command, 10).await?;

    match event.event_type.as_str() {
        "result" => {
            let response: LanguageListResponse = serde_json::from_value(event.data)
                .map_err(|e| format!("Parse error: {}", e))?;
            Ok(response)
        }
        "error" => {
            let msg = event.data.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            Err(format!("Python error: {}", msg))
        }
        _ => Err(format!("Unexpected event type: {}", event.event_type))
    }
}

/// List installed language packs
#[tauri::command]
pub async fn list_installed_languages(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<InstalledLanguagesResponse, String> {
    // Ensure Python process is started
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        }
    }

    // Start event loop
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        if process.is_running() {
            process.start_event_loop(app.clone())?;
        }
    }

    // Send command and wait for response
    let command = PythonCommand {
        command: "list_installed_languages".to_string(),
        file_path: None,
        options: None,
        request_id: None, // Will be set by send_command_and_wait
    };

    // Clone the process to avoid holding lock across await
    let temp_process = {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.clone_ref()
    };

    let event = temp_process.send_command_and_wait(command, 10).await?;

    match event.event_type.as_str() {
        "result" => {
            let response: InstalledLanguagesResponse = serde_json::from_value(event.data)
                .map_err(|e| format!("Parse error: {}", e))?;
            Ok(response)
        }
        "error" => {
            let msg = event.data.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            Err(format!("Python error: {}", msg))
        }
        _ => Err(format!("Unexpected event type: {}", event.event_type))
    }
}

/// Get installation status for a specific language
#[tauri::command]
pub async fn get_language_status(
    language_code: String,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<LanguageInfo, String> {
    // Ensure Python process is started
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        }
    }

    // Start event loop
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        if process.is_running() {
            process.start_event_loop(app.clone())?;
        }
    }

    // Send command and wait for response
    let command = PythonCommand {
        command: "get_language_status".to_string(),
        file_path: None,
        options: Some(serde_json::json!({
            "language_code": language_code
        })),
        request_id: None, // Will be set by send_command_and_wait
    };

    // Clone the process to avoid holding lock across await
    let temp_process = {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.clone_ref()
    };

    let event = temp_process.send_command_and_wait(command, 10).await?;

    match event.event_type.as_str() {
        "result" => {
            let status: LanguageInfo = serde_json::from_value(event.data)
                .map_err(|e| format!("Parse error: {}", e))?;
            Ok(status)
        }
        "error" => {
            let msg = event.data.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            Err(format!("Python error: {}", msg))
        }
        _ => Err(format!("Unexpected event type: {}", event.event_type))
    }
}

/// Download and install a language pack
#[tauri::command]
pub async fn download_language_pack(
    language_code: String,
    enable_angle_classification: Option<bool>,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<DownloadResponse, String> {
    eprintln!("[DEBUG] download_language_pack called for language: {}", language_code);

    // Ensure Python process is started
    eprintln!("[DEBUG] Checking if Python process is started...");
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;

        if !process.is_running() {
            eprintln!("[DEBUG] Python not running, starting...");
            let script_path = get_python_script_path(&app)?;
            process.start(&script_path)?;
        } else {
            eprintln!("[DEBUG] Python already running");
        }
    }

    // Start event loop (progress events will be broadcast to frontend automatically)
    eprintln!("[DEBUG] Starting event loop...");
    {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        if process.is_running() {
            process.start_event_loop(app.clone())?;
        }
    }
    eprintln!("[DEBUG] Event loop started");

    // Send command and wait for response (5 minutes timeout for download)
    eprintln!("[DEBUG] Creating command...");
    let command = PythonCommand {
        command: "download_language_pack".to_string(),
        file_path: None,
        options: Some(serde_json::json!({
            "language_code": language_code,
            "enable_angle_classification": enable_angle_classification.unwrap_or(false)
        })),
        request_id: None, // Will be set by send_command_and_wait
    };

    // Clone the process to avoid holding lock across await
    eprintln!("[DEBUG] Cloning process reference...");
    let temp_process = {
        let process = state.python_process.lock()
            .map_err(|e| format!("Failed to lock process: {}", e))?;
        process.clone_ref()
    };

    eprintln!("[DEBUG] Calling send_command_and_wait with 300s timeout...");
    let event = temp_process.send_command_and_wait(command, 300).await?;
    eprintln!("[DEBUG] Received event response");

    match event.event_type.as_str() {
        "result" => {
            let response: DownloadResponse = serde_json::from_value(event.data)
                .map_err(|e| format!("Parse error: {}", e))?;
            Ok(response)
        }
        "error" => {
            let msg = event.data.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            Err(format!("Python error: {}", msg))
        }
        _ => Err(format!("Unexpected event type: {}", event.event_type))
    }
}
