<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { onMount, onDestroy } from "svelte";
  import { navigateToMainMenu } from "../stores/navigation";
  import {
    ocrQueue,
    destinationFolder,
    addMultipleToQueue,
    removeFromQueue,
    updateFileStatus,
  } from "../stores/ocrQueue";
  import Button from "./shared/Button.svelte";
  import FileGrid from "./shared/FileGrid.svelte";
  import Terminal from "./shared/Terminal.svelte";
  import type { FileInfo } from "../types";

  // Event payload types
  interface ProgressEvent {
    current: number;
    total: number;
    message: string;
    percent?: number;
  }

  interface ResultEvent {
    successful: Array<{ file: string; pages: number; output: string }>;
    failed: Array<{ file: string; error: string }>;
    total_files: number;
    total_pages_processed: number;
    duration_seconds: number;
  }

  interface ErrorEvent {
    message: string;
  }

  // Local state
  let selectedFileIds: string[] = [];
  let terminalLogs: string[] = [];
  let isProcessing = false;
  let processingProgress = 0;

  // Event listener cleanup functions
  let unlistenProgress: UnlistenFn | null = null;
  let unlistenResult: UnlistenFn | null = null;
  let unlistenError: UnlistenFn | null = null;

  // Reactive statements
  $: hasSelection = selectedFileIds.length > 0;
  $: canStartOCR = $ocrQueue.length > 0 && !isProcessing && $destinationFolder.length > 0;

  /**
   * Handle progress events from Python backend
   */
  function handleProgress(event: { payload: ProgressEvent }) {
    const { current, total, message, percent } = event.payload;

    // Update terminal with progress message
    addLog(message, 'info');

    // Update overall progress
    processingProgress = percent || (current / total) * 100;

    // Extract filename from message and update file status
    // Expected format: "Processing file 2/5: doc.pdf (page 10/50)"
    const fileMatch = message.match(/Processing file \d+\/\d+: (.+?) \(/);
    if (fileMatch) {
      const filename = fileMatch[1];
      updateFileStatusByName(filename, 'processing', percent);
    }
  }

  /**
   * Handle result events from Python backend
   */
  function handleResult(event: { payload: ResultEvent }) {
    const { successful, failed, total_pages_processed, duration_seconds } = event.payload;

    // Mark successful files as complete
    successful.forEach(item => {
      updateFileStatusByName(item.file, 'complete');
      addLog(`✓ Completed: ${item.file} (${item.pages} pages)`, 'success');
    });

    // Mark failed files as failed
    failed.forEach(item => {
      updateFileStatusByName(item.file, 'failed');
      addLog(`ERROR: Failed: ${item.file} - ${item.error}`, 'error');
    });

    // Final summary
    addLog('', 'info'); // Empty line for spacing
    addLog(`Batch complete: ${successful.length} succeeded, ${failed.length} failed`, 'info');
    addLog(`Total pages processed: ${total_pages_processed} in ${duration_seconds.toFixed(1)}s`, 'info');

    // Reset processing state
    isProcessing = false;
    processingProgress = 0;
  }

  /**
   * Handle error events from Python backend
   */
  function handleError(event: { payload: ErrorEvent }) {
    const { message } = event.payload;

    // Add error to terminal
    addLog(`ERROR: ${message}`, 'error');

    // Stop processing state
    isProcessing = false;
    processingProgress = 0;
  }

  /**
   * Update file status by filename (helper function)
   */
  function updateFileStatusByName(
    filename: string,
    status: 'pending' | 'processing' | 'complete' | 'failed',
    progress?: number
  ) {
    // Find file by name and update its status
    const file = $ocrQueue.find(f => f.fileName === filename);
    if (file) {
      updateFileStatus(file.id, status, progress);
    }
  }

  /**
   * Load page counts for all files in queue
   */
  async function loadPageCounts() {
    for (const file of $ocrQueue) {
      if (file.pages === 0 || file.pages === null) {
        try {
          const info = await invoke<FileInfo>("get_file_info", {
            filePath: file.filePath
          });

          if (info.page_count) {
            // Update the file in the queue with the actual page count
            ocrQueue.update(queue =>
              queue.map(item => {
                if (item.id === file.id) {
                  return { ...item, pages: info.page_count || 0 };
                }
                return item;
              })
            );
          }
        } catch (error) {
          console.error(`Failed to get page count for ${file.fileName}:`, error);
        }
      }
    }
  }

  /**
   * Add files to the OCR queue
   */
  async function handleAddFiles() {
    try {
      const filePaths = await invoke<string[]>("select_multiple_pdf_files");

      if (!filePaths || filePaths.length === 0) {
        return;
      }

      addLog(`Adding ${filePaths.length} file(s) to queue...`, 'info');

      // Get file info for each file
      const fileInfos: FileInfo[] = [];
      for (const filePath of filePaths) {
        try {
          const info = await invoke<FileInfo>("get_file_info", { filePath });
          fileInfos.push(info);
        } catch (error) {
          addLog(`ERROR: Failed to get info for ${filePath}: ${error}`, 'error');
        }
      }

      // Add to queue
      addMultipleToQueue(
        fileInfos.map((info) => ({
          fileName: info.name,
          filePath: info.path,
          pages: info.page_count || 0,
          size: info.size,
        }))
      );

      addLog(`✓ Added ${fileInfos.length} file(s) to queue`, 'success');

      // Load page counts asynchronously for files that don't have them
      loadPageCounts();
    } catch (error) {
      console.error("Failed to add files:", error);
      addLog(`ERROR: Failed to add files: ${error}`, 'error');
    }
  }

  /**
   * Browse for destination folder
   */
  async function handleBrowseDestination() {
    try {
      const folder = await invoke<string | null>("select_folder");
      if (folder) {
        $destinationFolder = folder;
        addLog(`Destination folder changed to: ${folder}`);
      }
    } catch (error) {
      console.error("Failed to select folder:", error);
      addLog(`ERROR: Failed to select folder: ${error}`, 'error');
    }
  }

  /**
   * Remove selected files from queue
   */
  async function handleRemoveSelected() {
    if (selectedFileIds.length === 0) {
      return;
    }

    try {
      // Remove from queue store
      removeFromQueue(selectedFileIds);
      addLog(`Removed ${selectedFileIds.length} file(s) from queue`);
      selectedFileIds = [];
    } catch (error) {
      console.error("Failed to remove files:", error);
      addLog(`ERROR: Failed to remove files: ${error}`, 'error');
    }
  }

  /**
   * Start OCR batch processing
   */
  async function handleStartOCR() {
    // Input validation
    if ($ocrQueue.length === 0) {
      addLog('WARNING: No files in queue. Please add files first.', 'info');
      return;
    }

    if (!$destinationFolder || $destinationFolder.length === 0) {
      addLog('WARNING: No destination folder selected. Please select a destination folder.', 'info');
      return;
    }

    if (isProcessing) {
      // Cancel processing
      try {
        await invoke("cancel_batch_ocr");
        addLog('Cancelling OCR batch processing...', 'info');
        isProcessing = false;
        processingProgress = 0;
      } catch (error) {
        console.error("Failed to cancel OCR:", error);
        addLog(`ERROR: Failed to cancel OCR: ${error}`, 'error');
      }
      return;
    }

    try {
      isProcessing = true;
      processingProgress = 0;

      const filePaths = $ocrQueue.map((item) => item.filePath);

      // Clear terminal (optional - keeps initialization message)
      addLog('', 'info'); // Empty line for spacing
      addLog(`Starting OCR batch: ${filePaths.length} file(s)`, 'info');

      // Reset all file statuses to pending
      $ocrQueue.forEach(file => {
        updateFileStatus(file.id, 'pending', 0);
      });

      // Start batch OCR
      await invoke("start_batch_ocr", {
        files: filePaths,
        destination: $destinationFolder,
      });

      addLog('OCR batch started successfully. Processing...', 'info');
    } catch (error) {
      console.error("OCR processing failed:", error);
      addLog(`ERROR: Failed to start OCR: ${error}`, 'error');
      isProcessing = false;
      processingProgress = 0;
    }
  }

  /**
   * Add log message to terminal
   */
  function addLog(message: string, type: 'info' | 'success' | 'error' = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = type === 'error' ? 'ERROR:' : type === 'success' ? '✓' : '>';
    terminalLogs = [...terminalLogs, `[${timestamp}] ${prefix} ${message}`];

    // Keep max 1000 lines
    if (terminalLogs.length > 1000) {
      terminalLogs = terminalLogs.slice(-1000);
    }
  }

  /**
   * Setup event listeners on component mount
   */
  onMount(async () => {
    addLog("OCR Module initialized", 'info');

    try {
      // Listen to progress events from Python backend
      unlistenProgress = await listen('python_progress', handleProgress);
      addLog('Event listener registered: python_progress', 'info');

      // Listen to result events
      unlistenResult = await listen('python_result', handleResult);
      addLog('Event listener registered: python_result', 'info');

      // Listen to error events
      unlistenError = await listen('python_error', handleError);
      addLog('Event listener registered: python_error', 'info');
    } catch (error) {
      console.error('Failed to setup event listeners:', error);
      addLog(`ERROR: Failed to setup event listeners: ${error}`, 'error');
    }
  });

  /**
   * Cleanup event listeners on component destroy
   */
  onDestroy(() => {
    // Cleanup all event listeners to prevent memory leaks
    if (unlistenProgress) {
      unlistenProgress();
      unlistenProgress = null;
    }

    if (unlistenResult) {
      unlistenResult();
      unlistenResult = null;
    }

    if (unlistenError) {
      unlistenError();
      unlistenError = null;
    }
  });
</script>

<div class="h-full flex flex-col bg-white dark:bg-gray-900">
  <!-- Header -->
  <div
    class="bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700 px-6 py-4"
  >
    <div class="flex items-center justify-between">
      <Button variant="secondary" size="sm" on:click={navigateToMainMenu}>
        ← Return to Main Menu
      </Button>
      <h2 class="text-2xl font-bold text-gray-900 dark:text-white">
        OCR Module
      </h2>
      <div class="w-40"></div>
      <!-- Spacer for centering -->
    </div>
  </div>

  <!-- Three-Panel Layout -->
  <div class="flex-1 flex overflow-hidden">
    <!-- Left Panel (20%) -->
    <div
      class="w-[20%] border-r border-gray-300 dark:border-gray-700 p-4 flex flex-col gap-4"
    >
      <!-- Add Files Button -->
      <div>
        <Button
          variant="primary"
          size="lg"
          on:click={handleAddFiles}
          disabled={isProcessing}
        >
          Add Files
        </Button>
      </div>

      <!-- Destination Folder -->
      <div class="space-y-2">
        <label
          for="destination"
          class="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Destination Folder:
        </label>
        <input
          id="destination"
          type="text"
          value={$destinationFolder}
          readonly
          class="w-full px-2 py-1.5 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded text-xs text-gray-900 dark:text-gray-100 truncate"
          title={$destinationFolder}
        />
        <div class="w-full">
          <Button
            variant="secondary"
            size="sm"
            on:click={handleBrowseDestination}
            disabled={isProcessing}
          >
            Browse...
          </Button>
        </div>
      </div>

      <!-- Start/Cancel OCR Button -->
      <div class="mt-4">
        <Button
          variant={isProcessing ? "danger" : "success"}
          size="lg"
          on:click={handleStartOCR}
          disabled={!canStartOCR && !isProcessing}
        >
          {isProcessing ? "Cancel OCR" : "Start OCR"}
        </Button>
      </div>

      <!-- Progress Bar -->
      {#if isProcessing && processingProgress > 0}
        <div class="mt-4 space-y-2">
          <div class="text-xs text-gray-600 dark:text-gray-400">
            Progress: {processingProgress.toFixed(1)}%
          </div>
          <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
            <div
              class="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
              style="width: {processingProgress}%"
            ></div>
          </div>
        </div>
      {/if}

      <!-- Queue Info -->
      <div class="mt-auto pt-6 border-t border-gray-300 dark:border-gray-700">
        <div class="text-sm text-gray-600 dark:text-gray-400 space-y-1">
          <div>Files in queue: {$ocrQueue.length}</div>
          {#if hasSelection}
            <div class="text-blue-600 dark:text-blue-400">
              {selectedFileIds.length} selected
            </div>
          {/if}
        </div>
      </div>
    </div>

    <!-- Center Panel (50%) - Hero Section -->
    <div class="w-[50%] border-r border-gray-300 dark:border-gray-700 p-4 flex flex-col">
      <!-- Remove Button (conditional) -->
      {#if hasSelection}
        <div class="mb-4">
          <Button
            variant="danger"
            size="sm"
            on:click={handleRemoveSelected}
            disabled={isProcessing}
          >
            Remove Selected ({selectedFileIds.length})
          </Button>
        </div>
      {/if}

      <!-- File Grid -->
      <div class="flex-1 overflow-auto">
        <FileGrid
          bind:selectedIds={selectedFileIds}
        />
      </div>
    </div>

    <!-- Right Panel (30%) -->
    <div class="w-[30%] p-4">
      <div class="h-full">
        <Terminal logs={terminalLogs} maxLines={1000} />
      </div>
    </div>
  </div>
</div>
