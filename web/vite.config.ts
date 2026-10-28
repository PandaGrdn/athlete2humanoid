import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// API_BASE (or VITE_API_BASE): the backend's origin, e.g. https://<user>-a2h-api.hf.space.
// Empty in local dev: /api is proxied to uvicorn on :8000.
const apiBase = (process.env.API_BASE ?? process.env.VITE_API_BASE ?? "").replace(/\/$/, "");
if (apiBase && !/^https?:\/\/[^/\s'";]+$/.test(apiBase)) throw new Error(`API_BASE must be an origin, got ${apiBase}`);

// Cloudflare Pages reads dist/_headers. Scripts and styles only from this origin; the API origin only
// for fetch/EventSource, videos and posters; blob: only for media (the local preview of an upload).
// Canvases need nothing. See docs/SECURITY.md.
export function securityHeaders(api: string): string {
  const csp = [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    `img-src 'self' ${api}`.trim(),
    `media-src 'self' blob: ${api}`.trim(),
    `connect-src 'self' ${api}`.trim(),
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "upgrade-insecure-requests",
  ].join("; ");
  return [
    "/*",
    `  Content-Security-Policy: ${csp}`,
    "  X-Content-Type-Options: nosniff",
    "  X-Frame-Options: DENY",
    "  Referrer-Policy: strict-origin-when-cross-origin",
    "  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()",
    "  Strict-Transport-Security: max-age=31536000",
    "  Cross-Origin-Opener-Policy: same-origin",
    "/assets/*",
    "  Cache-Control: public, max-age=31536000, immutable",
    "",
  ].join("\n");
}

function cloudflareHeaders(): Plugin {
  let outDir = "dist";
  return {
    name: "a2h-cloudflare-headers",
    apply: "build",
    configResolved(c) {
      outDir = resolve(c.root, c.build.outDir);
    },
    closeBundle() {
      writeFileSync(resolve(outDir, "_headers"), securityHeaders(apiBase));
    },
  };
}

export default defineConfig({
  plugins: [react(), cloudflareHeaders()],
  define: { __API_BASE__: JSON.stringify(apiBase) },
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.API_PROXY ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
