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
    proxy: {
      "/api": process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
  },
})
