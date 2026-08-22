import http from 'node:http'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The API runs separately (uvicorn on :8000). Proxying /api keeps the frontend
// origin-relative, so there is no CORS juggling in dev and the same code works
// if the built bundle is later served by FastAPI itself.
//
// keepAlive is deliberately OFF. Node pools and reuses proxy sockets, but
// uvicorn closes idle keep-alive connections after ~5s. When the proxy sends a
// request down a socket the server is closing at that instant, the request is
// lost — no error, no retry, no timeout. Measured: ~25% of requests through the
// proxy hung indefinitely while the same calls made directly to :8000 all
// returned in under 100ms. That was the "app takes forever to load" bug. A fresh
// connection per request costs about a millisecond on localhost.
const apiProxy = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    agent: new http.Agent({ keepAlive: false }),
    // Belt and braces: if anything still wedges, fail in 60s rather than hang
    // forever, so the UI can show an error instead of spinning.
    timeout: 60000,
    proxyTimeout: 60000,
  },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: apiProxy,
  },
  // `vite preview` serves the built bundle — one pre-bundled file instead of the
  // ~900 separate module requests the dev server issues, and no on-demand
  // transforming. That is the difference between a demo that opens instantly and
  // one that appears to hang, so preview needs the same /api proxy as dev.
  preview: {
    port: 4173,
    proxy: apiProxy,
  },
})
