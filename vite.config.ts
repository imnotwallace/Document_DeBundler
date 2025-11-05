import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      // Exclude Python virtual environment and build artifacts from file watching
      ignored: [
        '**/python-backend/.venv/**',
        '**/python-backend/**/*.pyc',
        '**/python-backend/**/__pycache__/**',
        '**/src-tauri/target/**',
        '**/.git/**'
      ],
      // Use polling on Windows to avoid filesystem watcher issues
      usePolling: false
    }
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: ['es2021', 'chrome100', 'safari13'],
    minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
})
