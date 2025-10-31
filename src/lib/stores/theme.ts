// Theme detection and management store
import { readable } from 'svelte/store';
import type { Theme } from '../types';

/**
 * Detects the current system theme preference
 */
function detectSystemTheme(): Theme {
  if (typeof window === 'undefined') {
    return 'dark'; // Default for SSR
  }

  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
  return prefersDark.matches ? 'dark' : 'light';
}

/**
 * Creates a readable store that automatically detects and follows system theme changes
 */
function createThemeStore() {
  return readable<Theme>(detectSystemTheme(), (set) => {
    if (typeof window === 'undefined') {
      return; // No cleanup needed for SSR
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    // Update theme when system preference changes
    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      set(e.matches ? 'dark' : 'light');
    };

    // Listen for changes
    // Note: Safari doesn't support addEventListener on MediaQueryList
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
    } else {
      // Fallback for older browsers
      mediaQuery.addListener(handleChange);
    }

    // Cleanup function
    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', handleChange);
      } else {
        mediaQuery.removeListener(handleChange);
      }
    };
  });
}

/**
 * Reactive theme store that follows system preferences
 *
 * Usage:
 * ```svelte
 * <script>
 *   import { theme } from '$lib/stores/theme';
 * </script>
 *
 * <div class={$theme === 'dark' ? 'dark' : ''}>
 *   Current theme: {$theme}
 * </div>
 * ```
 */
export const theme = createThemeStore();

/**
 * Get the current theme value (non-reactive)
 */
export function getCurrentTheme(): Theme {
  return detectSystemTheme();
}
