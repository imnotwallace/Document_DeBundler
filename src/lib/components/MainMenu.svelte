<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { navigateToOCR, navigateToDebundle } from "../stores/navigation";
  import Button from "./shared/Button.svelte";

  async function quitApplication() {
    try {
      await invoke("quit_app");
    } catch (error) {
      console.error("Failed to quit application:", error);
      // Fallback to window.close() if Tauri command fails
      window.close();
    }
  }
</script>

<div class="container mx-auto px-4 py-8 h-full flex flex-col items-center justify-center">
  <!-- Title -->
  <h1 class="text-5xl font-bold mb-4 text-center text-gray-900 dark:text-white">
    Sam's PDF OCR and (De)Bundling Tool
  </h1>
  <p class="text-lg text-gray-600 dark:text-gray-400 mb-12 text-center">
    Process, split, and organize PDF documents with OCR capabilities
  </p>

  <!-- Module Buttons -->
  <div class="w-full max-w-2xl space-y-4">
    <!-- OCR Module Button -->
    <button
      on:click={navigateToOCR}
      class="w-full bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-600 text-white rounded-lg p-8 transition-all duration-200 hover:shadow-xl transform hover:-translate-y-1 focus:outline-none focus:ring-4 focus:ring-blue-300 dark:focus:ring-blue-800"
    >
      <div class="text-2xl font-bold mb-2">OCR Module</div>
      <div class="text-blue-100 dark:text-blue-200 text-sm">
        Batch OCR processing & queue management
      </div>
    </button>

    <!-- De-Bundling Module Button -->
    <button
      on:click={navigateToDebundle}
      class="w-full bg-green-600 hover:bg-green-700 dark:bg-green-700 dark:hover:bg-green-600 text-white rounded-lg p-8 transition-all duration-200 hover:shadow-xl transform hover:-translate-y-1 focus:outline-none focus:ring-4 focus:ring-green-300 dark:focus:ring-green-800"
    >
      <div class="text-2xl font-bold mb-2">De-Bundling Module</div>
      <div class="text-green-100 dark:text-green-200 text-sm">
        Split & organize bundled PDFs with LLM assistance
      </div>
    </button>

    <!-- Bundling Module Button (disabled) -->
    <button
      disabled
      class="w-full bg-gray-400 dark:bg-gray-600 text-gray-200 dark:text-gray-400 rounded-lg p-8 cursor-not-allowed opacity-50 relative"
    >
      <div class="text-2xl font-bold mb-2">Bundling Module</div>
      <div class="text-sm">Coming Soon - Disabled</div>
      <div class="absolute top-2 right-2 bg-yellow-500 dark:bg-yellow-600 text-white text-xs px-2 py-1 rounded-full">
        Coming Soon
      </div>
    </button>

    <!-- Quit Button -->
    <div class="flex justify-end mt-8">
      <Button variant="danger" size="md" on:click={quitApplication}>
        Quit
      </Button>
    </div>
  </div>
</div>
