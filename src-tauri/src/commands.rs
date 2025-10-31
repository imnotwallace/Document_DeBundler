use serde::{Deserialize, Serialize};
use tauri::api::dialog::blocking::FileDialogBuilder;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub name: String,
    pub size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessingStatus {
    pub is_running: bool,
    pub current_page: u32,
    pub total_pages: u32,
    pub current_operation: String,
    pub progress_percent: f32,
}

/// Opens a file dialog for selecting a PDF file
#[tauri::command]
pub fn select_pdf_file() -> Result<Option<String>, String> {
    let file_path = FileDialogBuilder::new()
        .add_filter("PDF Files", &["pdf"])
        .set_title("Select PDF Document")
        .pick_file();

    Ok(file_path.map(|p| p.to_string_lossy().to_string()))
}

/// Gets information about the selected file
#[tauri::command]
pub fn get_file_info(file_path: String) -> Result<FileInfo, String> {
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

    Ok(FileInfo {
        path: file_path,
        name,
        size: metadata.len(),
    })
}

/// Starts PDF processing via Python backend
#[tauri::command]
pub async fn start_processing(
    file_path: String,
    enable_ocr: bool,
    output_dir: Option<String>,
) -> Result<String, String> {
    // TODO: Implement Python subprocess call
    // This will be implemented when we create the Python bridge
    Ok(format!(
        "Processing started for: {} (OCR: {})",
        file_path, enable_ocr
    ))
}

/// Cancels the current processing operation
#[tauri::command]
pub fn cancel_processing() -> Result<(), String> {
    // TODO: Implement cancellation logic
    Ok(())
}

/// Gets the current processing status
#[tauri::command]
pub fn get_processing_status() -> Result<ProcessingStatus, String> {
    // TODO: Implement status retrieval from Python process
    Ok(ProcessingStatus {
        is_running: false,
        current_page: 0,
        total_pages: 0,
        current_operation: "Idle".to_string(),
        progress_percent: 0.0,
    })
}
