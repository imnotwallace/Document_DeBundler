<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { navigateToMainMenu } from "../stores/navigation";
  import {
    ocrQueue,
    destinationFolder,
    addMultipleToQueue,
    removeFromQueue,
  } from "../stores/ocrQueue";
  import Button from "./shared/Button.svelte";
  import FileGrid from "./shared/FileGrid.svelte";
  import Terminal from "./shared/Terminal.svelte";
  import type { FileInfo } from "../types";

  // Local state
  let selectedFileIds: string[] = [];
  let terminalLogs: string[] = [];
  let isProcessing = false;

  // Reactive statements
  $: hasSelection = selectedFileIds.length > 0;
  $: canStartOCR = $ocrQueue.length > 0 && !isProcessing;

  /**
   * Add files to the OCR queue
   */
  async function handleAddFiles() {
    try {
      const filePaths = await invoke<string[]>("select_multiple_pdf_files");
      
      if (!filePaths || filePaths.length === 0) {
        return;
      }

      addLog(`Adding ${filePaths.length} file(s) to queue...`);

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
    if ($ocrQueue.length === 0) {
      return;
    }

    try {
      isProcessing = true;
      addLog(`Starting OCR batch processing for ${$ocrQueue.length} file(s)...`, 'info');

      const filePaths = $ocrQueue.map((item) => item.filePath);

      await invoke("start_batch_ocr", {
        files: filePaths,
        destination: $destinationFolder,
      });

      addLog(`✓ OCR batch processing completed`, 'success');
    } catch (error) {
      console.error("OCR processing failed:", error);
      addLog(`ERROR: OCR processing failed: ${error}`, 'error');
    } finally {
      isProcessing = false;
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

  // Initialize
  addLog("OCR Module initialized", 'info');
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
        <Button
          variant="secondary"
          size="sm"
          on:click={handleBrowseDestination}
          disabled={isProcessing}
          class="w-full"
        >
          Browse...
        </Button>
      </div>

      <!-- Start OCR Button -->
      <div class="mt-4">
        <Button
          variant="success"
          size="lg"
          on:click={handleStartOCR}
          disabled={!canStartOCR}
        >
          {isProcessing ? "Processing..." : "Start OCR"}
        </Button>
      </div>

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
