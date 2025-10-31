// Prevents additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod python_bridge;

use commands::*;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            select_pdf_file,
            get_file_info,
            start_processing,
            cancel_processing,
            get_processing_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
