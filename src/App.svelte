<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { theme } from "./lib/stores/theme";
  import { currentModule, navigateToMainMenu } from "./lib/stores/navigation";
  import MainMenu from "./lib/components/MainMenu.svelte";
  import OCRModule from "./lib/components/OCRModule.svelte";
  import Button from "./lib/components/shared/Button.svelte";
  import Modal from "./lib/components/shared/Modal.svelte";
  import ProgressBar from "./lib/components/shared/ProgressBar.svelte";

  // Existing state from original App.svelte
  let selectedFile: string | null = null;
  let fileInfo: any = null;
  let isProcessing = false;
  let enableOCR = true;

  // Apply theme class to document root for Tailwind dark mode
  $: {
    if (typeof document !== 'undefined') {
      if ($theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    }
  }

  // Existing functions from original App.svelte
  async function selectFile() {
    try {
      const filePath = await invoke<string | null>("select_pdf_file");
      if (filePath) {
        selectedFile = filePath;
        fileInfo = await invoke("get_file_info", { filePath });
      }
    } catch (error) {
      console.error("Failed to select file:", error);
      alert(`Error: ${error}`);
    }
  }

  async function startProcessing() {
    if (!selectedFile) {
      alert("Please select a PDF file first");
      return;
    }

    try {
      isProcessing = true;
      const result = await invoke("start_processing", {
        filePath: selectedFile,
        enableOcr: enableOCR,
        outputDir: null,
      });
      console.log("Processing result:", result);
      alert(result);
    } catch (error) {
      console.error("Processing failed:", error);
      alert(`Error: ${error}`);
    } finally {
      isProcessing = false;
    }
  }

  function formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + " KB";
    if (bytes < 1024 * 1024 * 1024)
      return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  }
</script>

<main class="w-full h-full">
  <div class="w-full h-full bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    {#if $currentModule === 'main_menu'}
      <!-- Main Menu Module -->
      <MainMenu />

    {:else if $currentModule === 'ocr'}
      <!-- OCR Module -->
      <OCRModule />

    {:else if $currentModule === 'debundle'}
      <!-- De-Bundling Module (Placeholder - Phase 4) -->
      <div class="h-full flex flex-col">
        <!-- Header -->
        <div class="bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700 px-6 py-4">
          <div class="flex items-center justify-between">
            <Button variant="secondary" size="sm" on:click={navigateToMainMenu}>
              ← Return to Main Menu
            </Button>
            <h2 class="text-2xl font-bold text-gray-900 dark:text-white">De-Bundling Module</h2>
            <div class="w-40"></div> <!-- Spacer -->
          </div>
        </div>

        <!-- Content -->
        <div class="flex-1 p-8">
          <div class="max-w-4xl mx-auto">
            <div class="bg-green-50 dark:bg-green-900/20 border-2 border-green-300 dark:border-green-700 rounded-lg p-8 text-center">
              <h3 class="text-xl font-semibold mb-4 text-green-900 dark:text-green-100">
                De-Bundling Module - Phase 4
              </h3>
              <p class="text-green-800 dark:text-green-200 mb-4">
                This module will be implemented in Phase 4 with:
              </p>
              <ul class="text-left text-green-700 dark:text-green-300 space-y-2 max-w-md mx-auto">
                <li>✓ File selection with drag & drop</li>
                <li>✓ LLM-powered boundary detection</li>
                <li>✓ Document grid editor with metadata</li>
                <li>✓ PDF preview panel</li>
                <li>✓ Smart file naming and organization</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

    {:else if $currentModule === 'bundle'}
      <!-- Bundling Module (Placeholder - Future) -->
      <div class="h-full flex flex-col">
        <!-- Header -->
        <div class="bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700 px-6 py-4">
          <div class="flex items-center justify-between">
            <Button variant="secondary" size="sm" on:click={navigateToMainMenu}>
              ← Return to Main Menu
            </Button>
            <h2 class="text-2xl font-bold text-gray-900 dark:text-white">Bundling Module</h2>
            <div class="w-40"></div> <!-- Spacer -->
          </div>
        </div>

        <!-- Content -->
        <div class="flex-1 p-8">
          <div class="max-w-4xl mx-auto">
            <div class="bg-gray-50 dark:bg-gray-800 border-2 border-gray-300 dark:border-gray-700 rounded-lg p-8 text-center">
              <h3 class="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
                Bundling Module
              </h3>
              <p class="text-gray-700 dark:text-gray-300 mb-4">
                Coming Soon
              </p>
              <p class="text-gray-600 dark:text-gray-400">
                This feature will allow you to combine multiple PDF documents into a single file.
              </p>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>
</main>
