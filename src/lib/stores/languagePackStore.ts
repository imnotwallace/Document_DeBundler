/**
 * Language Pack Store
 *
 * Manages state for language pack downloads and installations.
 * Tracks available languages, installation status, and download progress.
 */

import { writable, derived } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';

// ===== Types =====

export interface LanguagePackInfo {
    code: string;
    name: string;
    installed: boolean;  // True if ANY version is installed
    script_name: string;  // e.g., "latin", "arabic", "cyrillic"
    script_description: string;  // Description of what languages use this script
    total_size_mb: number;
    detection_installed: boolean;
    recognition_installed: boolean;  // For current model_version
    model_version?: string;  // "server" or "mobile" - currently selected version
    has_server_version?: boolean;  // Whether server version is available
    available_versions?: string[];  // List of available versions
    server_installed?: boolean;  // Whether server version is installed
    mobile_installed?: boolean;  // Whether mobile version is installed
}

export interface DownloadProgress {
    language: string;
    language_name: string;
    phase: 'initializing' | 'downloading' | 'complete' | 'error';
    progress_percent: number;
    message: string;
    speed_mbps?: number;
    error?: string;
}

// ===== Stores =====

/**
 * All available language packs with their installation status
 */
export const availableLanguages: Writable<LanguagePackInfo[]> = writable([]);

/**
 * Download progress for each language currently being downloaded
 * Key: language code, Value: progress info
 */
export const downloadProgress: Writable<Map<string, DownloadProgress>> = writable(new Map());

/**
 * Loading state for language list
 */
export const isLoadingLanguages: Writable<boolean> = writable(false);

/**
 * Error message for language operations
 */
export const errorMessage: Writable<string | null> = writable(null);

// ===== Derived Stores =====

/**
 * Only installed languages
 */
export const installedLanguages: Readable<LanguagePackInfo[]> = derived(
    availableLanguages,
    ($availableLanguages) => $availableLanguages.filter(lang => lang.installed)
);

/**
 * Installed language codes
 */
export const installedLanguageCodes: Readable<string[]> = derived(
    installedLanguages,
    ($installedLanguages) => $installedLanguages.map(lang => lang.code)
);

/**
 * Not installed languages
 */
export const notInstalledLanguages: Readable<LanguagePackInfo[]> = derived(
    availableLanguages,
    ($availableLanguages) => $availableLanguages.filter(lang => !lang.installed)
);

/**
 * Count of installed languages
 */
export const installedCount: Readable<number> = derived(
    installedLanguages,
    ($installedLanguages) => $installedLanguages.length
);

/**
 * Total count of available languages
 */
export const totalCount: Readable<number> = derived(
    availableLanguages,
    ($availableLanguages) => $availableLanguages.length
);

/**
 * Check if any downloads are in progress
 */
export const hasActiveDownloads: Readable<boolean> = derived(
    downloadProgress,
    ($downloadProgress) => $downloadProgress.size > 0
);

/**
 * Get list of languages currently downloading
 */
export const downloadingLanguages: Readable<string[]> = derived(
    downloadProgress,
    ($downloadProgress) => Array.from($downloadProgress.keys())
);

// ===== Helper Functions =====

/**
 * Check if a specific language is installed
 */
export function isLanguageInstalled(languageCode: string): boolean {
    let installed = false;
    availableLanguages.subscribe(langs => {
        const lang = langs.find(l => l.code === languageCode);
        installed = lang?.installed || false;
    })();
    return installed;
}

/**
 * Check if a specific language is currently downloading
 */
export function isLanguageDownloading(languageCode: string): boolean {
    let downloading = false;
    downloadProgress.subscribe(progress => {
        downloading = progress.has(languageCode);
    })();
    return downloading;
}

/**
 * Get language info by code
 */
export function getLanguageInfo(languageCode: string): LanguagePackInfo | null {
    let found: LanguagePackInfo | null = null;
    availableLanguages.subscribe(langs => {
        found = langs.find(l => l.code === languageCode) || null;
    })();
    return found;
}

/**
 * Clear error message
 */
export function clearError() {
    errorMessage.set(null);
}

/**
 * Set error message
 */
export function setError(message: string) {
    errorMessage.set(message);
}

/**
 * Update download progress for a language
 */
export function updateDownloadProgress(progress: DownloadProgress) {
    downloadProgress.update(map => {
        const newMap = new Map(map);
        newMap.set(progress.language, progress);
        return newMap;
    });

    // Auto-remove from progress map after completion or error (after 3 seconds)
    if (progress.phase === 'complete' || progress.phase === 'error') {
        setTimeout(() => {
            downloadProgress.update(map => {
                const newMap = new Map(map);
                newMap.delete(progress.language);
                return newMap;
            });
        }, 3000);
    }
}

/**
 * Clear download progress for a language
 */
export function clearDownloadProgress(languageCode: string) {
    downloadProgress.update(map => {
        const newMap = new Map(map);
        newMap.delete(languageCode);
        return newMap;
    });
}

/**
 * Clear all download progress
 */
export function clearAllDownloadProgress() {
    downloadProgress.set(new Map());
}

/**
 * Reset all stores to initial state
 */
export function resetStores() {
    availableLanguages.set([]);
    downloadProgress.set(new Map());
    isLoadingLanguages.set(false);
    errorMessage.set(null);
}
