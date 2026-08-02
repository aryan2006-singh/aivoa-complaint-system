import path from "path"
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    host: true,
    // Vite's dev server rejects requests whose Host header it doesn't
    // recognize. Deployed on a host like Render, incoming requests carry a
    // dynamic *.onrender.com Host header, so the default (localhost-only)
    // allowlist must be relaxed for the deployed app to respond at all.
    allowedHosts: true,
    // This container has no reason to hot-reload -- the source is baked
    // into the image at build time, not edited live. Vite's chokidar file
    // watcher was hitting the container's file-descriptor limit at startup
    // (EMFILE: too many open files) and crashing the process before it
    // could even bind to a port; disabling the watcher entirely avoids that
    // without needing OS-level ulimit changes this deploy target doesn't
    // expose.
    watch: null,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
        // Platforms like Render route incoming requests by Host header.
        // Without changeOrigin, the proxy forwards this frontend's own
        // hostname instead of the backend's, so the request never reaches
        // the right service and just hangs.
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
  },
})
