/**
 * Language Pack Service
 *
 * Handles language pack operations: loading, downloading, status checking.
 * Sets up event listeners for download progress updates.
 */

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import {
    availableLanguages,
    downloadProgress,
    isLoadingLanguages,
    errorMessage,
    updateDownloadProgress,
    setError,
    clearError,
    type LanguagePackInfo,
    type DownloadProgress
} from '../stores/languagePackStore';

// Event listener cleanup function
let unlistenProgress: UnlistenFn | null = null;

/**
 * Load all available languages with installation status
 */
export async function loadAvailableLanguages(): Promise<void> {
    console.log('[languagePackService] Starting to load available languages...');
    isLoadingLanguages.set(true);
    clearError();

    try {
        console.log('[languagePackService] Invoking list_available_languages command...');
        const response = await invoke<{ languages: LanguagePackInfo[]; count: number }>(
            'list_available_languages'
        );

        console.log('[languagePackService] Received response:', response);
        availableLanguages.set(response.languages);
        console.log(`Loaded ${response.count} available languages`);
    } catch (err) {
        const errorMsg = `Failed to load languages: ${err}`;
        console.error(errorMsg);
        setError(errorMsg);
        throw err;
    } finally {
        console.log('[languagePackService] Setting isLoadingLanguages to false');
        isLoadingLanguages.set(false);
    }
}

/**
 * Load list of installed language codes
 */
export async function loadInstalledLanguages(): Promise<string[]> {
    try {
        const response = await invoke<{ installed_languages: string[]; count: number }>(
            'list_installed_languages'
        );

        console.log(`Found ${response.count} installed languages:`, response.installed_languages);
        return response.installed_languages;
    } catch (err) {
        console.error('Failed to load installed languages:', err);
        throw err;
    }
}

/**
 * Get detailed status for a specific language
 */
export async function getLanguageStatus(languageCode: string): Promise<LanguagePackInfo> {
    try {
        const status = await invoke<LanguagePackInfo>(
            'get_language_status',
            { languageCode }
        );

        console.log(`Language ${languageCode} status:`, status);
        return status;
    } catch (err) {
        console.error(`Failed to get status for ${languageCode}:`, err);
        throw err;
    }
}

/**
 * Download and install a language pack
 */
export async function downloadLanguagePack(
    languageCode: string,
    version: 'server' | 'mobile' = 'mobile',
    enableAngleClassification: boolean = false
): Promise<void> {
    clearError();

    try {
        console.log(`Starting download for language: ${languageCode} (version: ${version})`);

        const response = await invoke<{ success: boolean; language_code: string; message: string }>(
            'download_language_pack',
            {
                languageCode,
                version,
                enableAngleClassification
            }
        );

        if (response.success) {
            console.log(`Successfully installed: ${response.message}`);

            // Refresh language list after successful download
            await loadAvailableLanguages();
        } else {
            throw new Error('Download failed: success=false');
        }
    } catch (err) {
        const errorMsg = `Failed to download ${languageCode}: ${err}`;
        console.error(errorMsg);
        setError(errorMsg);
        throw err;
    }
}

/**
 * Setup event listener for download progress updates
 */
export function setupDownloadProgressListener(): void {
    // Cleanup existing listener
    if (unlistenProgress) {
        unlistenProgress();
        unlistenProgress = null;
    }

    // Setup new listener
    listen<{ type: string; data: DownloadProgress; request_id?: string }>('language-download-progress', (event) => {
        // Extract progress from the data field (Rust sends PythonEvent wrapper)
        const progress = event.payload.data;

        console.log(
            `[${progress.language}] ${progress.phase}: ${progress.message} (${progress.progress_percent.toFixed(1)}%)`
        );

        // Update store with progress
        updateDownloadProgress(progress);

        // Handle completion or error
        if (progress.phase === 'complete') {
            console.log(`Language ${progress.language_name} installed successfully`);
        } else if (progress.phase === 'error') {
            console.error(`Language ${progress.language_name} download failed:`, progress.error);
            setError(`Download failed for ${progress.language_name}: ${progress.error}`);
        }
    }).then((unlisten) => {
        unlistenProgress = unlisten;
        console.log('Language download progress listener setup');
    }).catch((err) => {
        console.error('Failed to setup download progress listener:', err);
    });
}

/**
 * Cleanup event listeners
 */
export function cleanupListeners(): void {
    if (unlistenProgress) {
        unlistenProgress();
        unlistenProgress = null;
        console.log('Language pack listeners cleaned up');
    }
}

/**
 * Check if language pack service is ready
 */
export function isServiceReady(): boolean {
    return unlistenProgress !== null;
}

/**
 * Initialize language pack service
 * Call this once when app starts
 */
export async function initializeLanguagePackService(): Promise<void> {
    console.log('Initializing language pack service...');

    // Setup event listeners
    setupDownloadProgressListener();

    // Load initial data
    try {
        await loadAvailableLanguages();
        console.log('Language pack service initialized successfully');
    } catch (err) {
        console.error('Failed to initialize language pack service:', err);
        throw err;
    }
}

/**
 * Download multiple language packs sequentially
 */
export async function downloadMultipleLanguages(
    languageCodes: string[],
    enableAngleClassification: boolean = false
): Promise<void> {
    console.log(`Downloading ${languageCodes.length} languages:`, languageCodes);

    for (const code of languageCodes) {
        try {
            await downloadLanguagePack(code, enableAngleClassification);
        } catch (err) {
            console.error(`Failed to download ${code}, continuing with next...`);
            // Continue with other downloads even if one fails
        }
    }

    console.log('Batch download complete');
}

/**
 * Refresh language list
 * Useful after downloads or external changes
 */
export async function refreshLanguages(): Promise<void> {
    console.log('Refreshing language list...');
    await loadAvailableLanguages();
}
