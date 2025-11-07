<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    availableLanguages,
    downloadProgress,
    isLoadingLanguages,
    errorMessage,
    installedCount,
    totalCount,
    hasActiveDownloads,
    clearError,
    type LanguagePackInfo,
    type DownloadProgress
  } from '../stores/languagePackStore';
  import {
    loadAvailableLanguages,
    downloadLanguagePack,
    setupDownloadProgressListener,
    cleanupListeners
  } from '../services/languagePackService';

  // Component state
  let showInstalledOnly = false;
  let searchQuery = '';
  let sortBy: 'name' | 'size' | 'status' = 'name';
  
  // Track selected model version for each language (default to 'server' if available, else 'mobile')
  let selectedVersions: Map<string, 'server' | 'mobile'> = new Map();

  // Lifecycle
  onMount(async () => {
    console.log('[LanguagePackManager] Component mounted, loading languages...');

    // Setup event listeners
    setupDownloadProgressListener();

    // Only load languages if not already loaded or loading
    if ($availableLanguages.length === 0 && !$isLoadingLanguages) {
      try {
        await loadAvailableLanguages();
        console.log('[LanguagePackManager] Languages loaded successfully');
      } catch (err) {
        console.error('[LanguagePackManager] Failed to load languages on mount:', err);
      }
    } else {
      console.log('[LanguagePackManager] Languages already loaded, skipping fetch');
    }
  });

  onDestroy(() => {
    cleanupListeners();
  });

  // Computed values
  $: filteredLanguages = $availableLanguages
    .filter(lang => {
      // Filter by installed status
      if (showInstalledOnly && !lang.installed) return false;

      // Filter by search query
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          lang.name.toLowerCase().includes(query) ||
          lang.code.toLowerCase().includes(query)
        );
      }

      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'size':
          return b.total_size_mb - a.total_size_mb;
        case 'status':
          return (b.installed ? 1 : 0) - (a.installed ? 1 : 0);
        default:
          return 0;
      }
    });

  // Initialize selected versions when languages load
  $: if ($availableLanguages.length > 0) {
    $availableLanguages.forEach(lang => {
      if (!selectedVersions.has(lang.code)) {
        // Default to server if available, otherwise mobile
        const defaultVersion = lang.has_server_version ? 'server' : 'mobile';
        selectedVersions.set(lang.code, defaultVersion);
      }
    });
    selectedVersions = selectedVersions; // Trigger reactivity
  }

  // Event handlers
  async function handleDownload(languageCode: string) {
    try {
      const version = selectedVersions.get(languageCode) || 'mobile';
      await downloadLanguagePack(languageCode, version, false);
    } catch (err) {
      console.error(`Download failed for ${languageCode}:`, err);
    }
  }
  
  function handleVersionChange(languageCode: string, version: 'server' | 'mobile') {
    selectedVersions.set(languageCode, version);
    selectedVersions = selectedVersions; // Trigger reactivity
  }
  
  function getSelectedVersion(languageCode: string): 'server' | 'mobile' {
    return selectedVersions.get(languageCode) || 'mobile';
  }

  function getProgressForLanguage(code: string): DownloadProgress | undefined {
    return $downloadProgress.get(code);
  }

  function formatSize(sizeMb: number): string {
    return `${sizeMb.toFixed(1)} MB`;
  }

  function formatSpeed(speedMbps: number | undefined): string {
    if (!speedMbps) return '';
    return `${speedMbps.toFixed(1)} Mbps`;
  }
</script>

<div class="language-pack-manager">
  <!-- Header -->
  <div class="header">
    <div class="title-section">
      <h2 class="title">Language Packs</h2>
      <div class="stats">
        <span class="stat">{$installedCount} / {$totalCount} installed</span>
      </div>
    </div>

    <!-- Search and Filters -->
    <div class="controls">
      <input
        type="text"
        class="search-input"
        placeholder="Search languages..."
        bind:value={searchQuery}
      />

      <label class="filter-checkbox">
        <input type="checkbox" bind:checked={showInstalledOnly} />
        <span>Show installed only</span>
      </label>

      <select class="sort-select" bind:value={sortBy}>
        <option value="name">Sort by Name</option>
        <option value="size">Sort by Size</option>
        <option value="status">Sort by Status</option>
      </select>
    </div>
  </div>

  <!-- Error Message -->
  {#if $errorMessage}
    <div class="error-banner">
      <span class="error-text">{$errorMessage}</span>
      <button class="error-close" on:click={clearError}>×</button>
    </div>
  {/if}

  <!-- Loading State -->
  {#if $isLoadingLanguages}
    <div class="loading">
      <div class="spinner"></div>
      <p>Loading language packs...</p>
    </div>
  {:else}
    <!-- Language List -->
    <div class="language-list">
      {#each filteredLanguages as lang (lang.code)}
        {@const progress = getProgressForLanguage(lang.code)}

        <div class="language-item" class:downloading={progress !== undefined}>
          <!-- Language Info -->
          <div class="language-info">
            <div class="language-header">
              <h3 class="language-name">{lang.name}</h3>
              <span class="language-code">{lang.code}</span>
            </div>

            <div class="language-details">
              <span class="language-size">{formatSize(lang.total_size_mb)}</span>
              <span class="detail-badge">{lang.script_name} script</span>
              
              {#if lang.has_server_version && !lang.installed}
                <div class="version-selector">
                  <label class="version-label">
                    <span class="version-label-text">Version:</span>
                    <select 
                      class="version-select"
                      value={getSelectedVersion(lang.code)}
                      on:change={(e) => handleVersionChange(lang.code, e.currentTarget.value as 'server' | 'mobile')}
                    >
                      <option value="server">Server (Accurate)</option>
                      <option value="mobile">Mobile (Fast)</option>
                    </select>
                  </label>
                </div>
              {:else if lang.has_server_version && lang.installed}
                <span class="version-badge">{lang.model_version || 'mobile'}</span>
              {/if}
            </div>
          </div>

          <!-- Status / Actions -->
          <div class="language-actions">
            {#if lang.installed}
              <!-- Installed -->
              <div class="status-badge installed">
                <svg class="status-icon" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <span>Installed</span>
              </div>
            {:else if progress}
              <!-- Downloading -->
              <div class="download-progress">
                <div class="progress-info">
                  <span class="progress-phase">{progress.phase}</span>
                  <span class="progress-percent">{progress.progress_percent.toFixed(0)}%</span>
                </div>

                <!-- Progress Bar -->
                <div class="progress-bar">
                  <div
                    class="progress-fill"
                    style="width: {progress.progress_percent}%"
                  ></div>
                </div>

                <!-- Additional Info -->
                <div class="progress-details">
                  <span class="progress-message">{progress.message}</span>
                  {#if progress.speed_mbps && progress.phase === 'downloading'}
                    <span class="progress-speed">{formatSpeed(progress.speed_mbps)}</span>
                  {/if}
                </div>

                {#if progress.phase === 'error' && progress.error}
                  <div class="progress-error">
                    Error: {progress.error}
                  </div>
                {/if}
              </div>
            {:else}
              <!-- Not Installed -->
              <button
                class="download-button"
                on:click={() => handleDownload(lang.code)}
                disabled={$hasActiveDownloads}
              >
                <svg class="button-icon" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
                </svg>
                <span>Download</span>
              </button>
            {/if}
          </div>
        </div>
      {/each}

      {#if filteredLanguages.length === 0}
        <div class="empty-state">
          <p>No languages found matching your criteria.</p>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Help Text -->
  <div class="help-text">
    <p>
      Download language packs to enable OCR for different languages. Language models are stored locally and only need to be downloaded once.
    </p>
  </div>
</div>

<style>
  .language-pack-manager {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    padding: 1rem;
    max-height: 70vh;
    overflow-y: auto;
  }

  /* Header */
  .header {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .title-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .title {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
    color: #1f2937;
  }

  .stats {
    display: flex;
    gap: 1rem;
  }

  .stat {
    font-size: 0.875rem;
    color: #6b7280;
    padding: 0.25rem 0.75rem;
    background: #f3f4f6;
    border-radius: 0.375rem;
  }

  /* Controls */
  .controls {
    display: flex;
    gap: 1rem;
    align-items: center;
  }

  .search-input {
    flex: 1;
    padding: 0.5rem 1rem;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }

  .search-input:focus {
    outline: none;
    border-color: #3b82f6;
    ring: 2px;
    ring-color: #3b82f630;
  }

  .filter-checkbox {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #4b5563;
    cursor: pointer;
    user-select: none;
  }

  .sort-select {
    padding: 0.5rem 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    background: white;
    cursor: pointer;
  }

  /* Error Banner */
  .error-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 0.375rem;
    color: #991b1b;
  }

  .error-text {
    font-size: 0.875rem;
  }

  .error-close {
    background: none;
    border: none;
    font-size: 1.5rem;
    line-height: 1;
    color: #991b1b;
    cursor: pointer;
    padding: 0;
    margin: 0;
  }

  /* Loading */
  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 3rem;
    color: #6b7280;
  }

  .spinner {
    width: 2rem;
    height: 2rem;
    border: 3px solid #e5e7eb;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Language List */
  .language-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .language-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    transition: all 0.2s;
  }

  .language-item:hover {
    border-color: #d1d5db;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .language-item.downloading {
    border-color: #3b82f6;
    background: #eff6ff;
  }

  /* Language Info */
  .language-info {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    flex: 1;
  }

  .language-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .language-name {
    margin: 0;
    font-size: 1rem;
    font-weight: 500;
    color: #1f2937;
  }

  .language-code {
    font-size: 0.75rem;
    font-family: monospace;
    color: #6b7280;
    padding: 0.125rem 0.375rem;
    background: #f3f4f6;
    border-radius: 0.25rem;
  }

  .language-details {
    display: flex;
    gap: 1rem;
    align-items: center;
  }

  .language-size {
    font-size: 0.875rem;
    color: #6b7280;
  }

  .detail-badge {
    font-size: 0.75rem;
    color: #059669;
    background: #d1fae5;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
  }
  
  .version-selector {
    margin-left: auto;
  }
  
  .version-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
  }
  
  .version-label-text {
    color: #6b7280;
    font-weight: 500;
  }
  
  .version-select {
    padding: 0.25rem 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    background: white;
    cursor: pointer;
    color: #1f2937;
  }
  
  .version-select:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }
  
  .version-badge {
    font-size: 0.75rem;
    color: #6366f1;
    background: #eef2ff;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
    font-weight: 500;
    text-transform: capitalize;
    margin-left: auto;
  }

  /* Actions */
  .language-actions {
    display: flex;
    align-items: center;
    min-width: 150px;
    justify-content: flex-end;
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .status-badge.installed {
    color: #065f46;
    background: #d1fae5;
  }

  .status-icon {
    width: 1.25rem;
    height: 1.25rem;
  }

  .download-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .download-button:hover:not(:disabled) {
    background: #2563eb;
  }

  .download-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .button-icon {
    width: 1rem;
    height: 1rem;
  }

  /* Progress */
  .download-progress {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    flex: 1;
    max-width: 300px;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .progress-phase {
    font-size: 0.75rem;
    font-weight: 500;
    color: #1f2937;
    text-transform: capitalize;
  }

  .progress-percent {
    font-size: 0.75rem;
    font-weight: 600;
    color: #3b82f6;
  }

  .progress-bar {
    width: 100%;
    height: 0.5rem;
    background: #e5e7eb;
    border-radius: 0.25rem;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s ease;
  }

  .progress-details {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .progress-message {
    font-size: 0.75rem;
    color: #6b7280;
  }

  .progress-speed {
    font-size: 0.75rem;
    color: #059669;
    font-weight: 500;
  }

  .progress-error {
    font-size: 0.75rem;
    color: #dc2626;
    padding: 0.25rem 0.5rem;
    background: #fef2f2;
    border-radius: 0.25rem;
  }

  /* Empty State */
  .empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #6b7280;
  }

  /* Help Text */
  .help-text {
    padding: 0.75rem 1rem;
    background: #f9fafb;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    color: #6b7280;
  }

  .help-text p {
    margin: 0;
  }
</style>
