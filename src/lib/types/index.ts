// Type definitions for Document De-Bundler application

// Module types for navigation
export type Module = 'main_menu' | 'ocr' | 'debundle' | 'bundle';

// Theme types
export type Theme = 'light' | 'dark';

// File information (matches Rust FileInfo struct)
export interface FileInfo {
  path: string;
  name: string;
  size: number;
  page_count?: number;
}

// OCR queue item
export interface OCRQueueItem {
  id: string;
  fileName: string;
  filePath: string;
  pages: number;
  size: number;
  status: 'queued' | 'processing' | 'complete' | 'failed';
  progress?: number;
  error?: string;

  // Timing fields
  queuedAt?: number;        // Unix timestamp (ms)
  startedAt?: number;       // Unix timestamp (ms)
  completedAt?: number;     // Unix timestamp (ms)
  elapsedTime?: number;     // Seconds
  currentPage?: number;
  totalPages?: number;
}

// Sort configuration for OCR queue
export interface SortConfig {
  column: 'fileName' | 'pages' | 'size' | 'status';
  direction: 'asc' | 'desc';
}

// Document row for de-bundling grid
export interface DocumentRow {
  id: number;
  itemNumber: number;
  startPage: number;
  endPage: number;
  documentDate: string; // ISO format YYYY-MM-DD or 'undated'
  documentType: string;
  documentName: string;
  confidenceScore: number;
}

// Document metadata (for LLM extraction)
export interface DocumentMetadata {
  date: string;
  doc_type: string;
  name: string;
}

// Processing status
export interface ProcessingStatus {
  is_running: boolean;
  current_page: number;
  total_pages: number;
  current_operation: string;
  progress_percent: number;
}

// De-bundle result
export interface DebundleResult {
  status: string;
  files_created: number;
}

// Progress callback data
export interface ProgressData {
  current: number;
  total: number;
  message: string;
  percentage?: number;
}

// OCR result
export interface OCRResult {
  file: string;
  status: 'success' | 'failed';
  pages?: number;
  output?: string;
  error?: string;
}

// Button component props
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'disabled';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
}

// Modal component props
export interface ModalProps {
  title: string;
  isOpen: boolean;
  onClose: () => void;
}
