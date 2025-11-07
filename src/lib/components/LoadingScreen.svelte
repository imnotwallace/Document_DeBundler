<script lang="ts">
  export let progress = 0; // 0-100
  export let message = "Loading...";
  export let currentStep = "";
  export let totalSteps = 0;
  export let completedSteps = 0;
</script>

<div class="loading-screen">
  <div class="loading-content">
    <!-- App Logo/Title -->
    <div class="app-header">
      <h1 class="app-title">Document De-Bundler</h1>
      <p class="app-subtitle">Initializing...</p>
    </div>

    <!-- Progress Information -->
    <div class="progress-section">
      <div class="progress-info">
        <span class="progress-message">{message}</span>
        {#if totalSteps > 0}
          <span class="progress-steps">{completedSteps} / {totalSteps}</span>
        {/if}
      </div>

      <!-- Progress Bar -->
      <div class="progress-bar">
        <div class="progress-fill" style="width: {progress}%"></div>
      </div>

      <!-- Current Step -->
      {#if currentStep}
        <div class="current-step">{currentStep}</div>
      {/if}
    </div>

    <!-- Loading Spinner -->
    <div class="spinner"></div>
  </div>
</div>

<style>
  .loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
  }

  .loading-content {
    text-align: center;
    padding: 3rem;
    max-width: 500px;
    width: 90%;
  }

  .app-header {
    margin-bottom: 3rem;
  }

  .app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
  }

  .app-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
  }

  .progress-section {
    margin-bottom: 2rem;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .progress-message {
    font-size: 0.875rem;
    color: #cbd5e1;
    font-weight: 500;
  }

  .progress-steps {
    font-size: 0.75rem;
    color: #64748b;
    font-variant-numeric: tabular-nums;
  }

  .progress-bar {
    width: 100%;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    transition: width 0.3s ease;
    border-radius: 3px;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
  }

  .current-step {
    margin-top: 0.75rem;
    font-size: 0.75rem;
    color: #64748b;
    font-style: italic;
  }

  .spinner {
    margin: 0 auto;
    width: 40px;
    height: 40px;
    border: 3px solid #1e293b;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
