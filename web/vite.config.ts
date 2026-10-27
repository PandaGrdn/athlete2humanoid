import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API_BASE (or VITE_API_BASE): the backend's origin, e.g. https://<user>-a2h-api.hf.space.
// Empty in local dev: /api is proxied to uvicorn on :8000.
const apiBase = (process.env.API_BASE ?? process.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export default defineConfig({
  plugins: [react()],
  define: { __API_BASE__: JSON.stringify(apiBase) },
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.API_PROXY ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
