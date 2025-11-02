<script lang="ts">
  export let useLLMRefinement = true;
  export let useLLMNaming = true;
  export let forceOCR = false;
  export let skipSplitting = false;
  
  // Optional: Pass LLM availability status
  export let llmAvailable = true;
</script>

<div class="processing-options space-y-4">
  <!-- OCR Options -->
  <div class="option-group">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
      OCR Options
    </h4>
    
    <label class="flex items-start gap-2 cursor-pointer">
      <input 
        type="checkbox" 
        bind:checked={forceOCR}
        class="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
      />
      <div class="flex-1">
        <span class="text-sm text-gray-900 dark:text-gray-100">Force OCR on all pages</span>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Re-run OCR even if text layer exists
        </p>
      </div>
    </label>
  </div>

  <!-- Splitting Options -->
  <div class="option-group">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
      Document Splitting
    </h4>
    
    <label class="flex items-start gap-2 cursor-pointer">
      <input 
        type="checkbox" 
        bind:checked={skipSplitting}
        class="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
      />
      <div class="flex-1">
        <span class="text-sm text-gray-900 dark:text-gray-100">Skip document splitting</span>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Process entire PDF as single document
        </p>
      </div>
    </label>
  </div>

  <!-- AI Features -->
  <div class="option-group">
    <h4 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
      AI Features
      {#if !llmAvailable}
        <span class="text-xs text-red-500">(Unavailable)</span>
      {/if}
    </h4>

    <label class="flex items-start gap-2 cursor-pointer" class:opacity-50={!llmAvailable}>
      <input 
        type="checkbox" 
        bind:checked={useLLMRefinement}
        disabled={!llmAvailable || skipSplitting}
        class="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
      />
      <div class="flex-1">
        <span class="text-sm text-gray-900 dark:text-gray-100">Use AI to refine split points</span>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          LLM analyzes content to validate splits (~20s overhead)
        </p>
      </div>
    </label>

    <label class="flex items-start gap-2 cursor-pointer mt-3" class:opacity-50={!llmAvailable}>
      <input 
        type="checkbox" 
        bind:checked={useLLMNaming}
        disabled={!llmAvailable}
        class="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
      />
      <div class="flex-1">
        <span class="text-sm text-gray-900 dark:text-gray-100">Use AI for document naming</span>
        <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          LLM generates intelligent filenames (~5s per document)
        </p>
      </div>
    </label>
  </div>
</div>

<style>
  .option-group {
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(156, 163, 175, 0.3);
  }
  
  .option-group:last-child {
    border-bottom: none;
  }
</style>
