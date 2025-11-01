// Prevents additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod python_bridge;

use commands::*;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .invoke_handler(tauri::generate_handler![
            select_pdf_file,
            select_multiple_pdf_files,
            select_folder,
            get_file_info,
            get_pdf_page_count,
            start_processing,
            cancel_processing,
            get_processing_status,
            quit_app
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
