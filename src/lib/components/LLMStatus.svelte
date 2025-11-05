<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { onMount } from 'svelte';

  interface LLMStatus {
    available: boolean;
    initialized: boolean;
    model_name?: string;
    gpu_enabled: boolean;
    expected_vram_gb?: number;
  }

  let llmAvailable = false;
  let llmInitialized = false;
  let modelName = '';
  let gpuEnabled = false;
  let vramUsage = 0;

  async function checkLLMStatus() {
    try {
      // Call Rust command to check LLM status via Python
      const status = await invoke<LLMStatus>('get_llm_status');
      llmAvailable = status.available;
      llmInitialized = status.initialized;
      modelName = status.model_name || 'Not loaded';
      gpuEnabled = status.gpu_enabled;
      vramUsage = status.expected_vram_gb || 0;
    } catch (e) {
      console.error('Failed to get LLM status:', e);
    }
  }

  onMount(() => {
    checkLLMStatus();
  });
</script>

<div class="llm-status-card">
  <h3>AI Features</h3>

  <div class="status-row">
    <span class="label">Status:</span>
    <span class="value {llmAvailable ? 'available' : 'unavailable'}">
      {llmAvailable ? '✓ Available' : '✗ Unavailable'}
    </span>
  </div>

  {#if llmAvailable}
    <div class="status-row">
      <span class="label">Model:</span>
      <span class="value">{modelName}</span>
    </div>

    <div class="status-row">
      <span class="label">GPU:</span>
      <span class="value">{gpuEnabled ? `✓ Enabled (${vramUsage.toFixed(1)}GB VRAM)` : 'CPU Only'}</span>
    </div>
  {/if}
</div>

<style>
  .llm-status-card {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    margin: 0.5rem 0;
  }

  .available {
    color: green;
  }

  .unavailable {
    color: red;
  }
</style>
