import path from "path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The frontend is built to ../web-dist and served by the FastAPI app; `pnpm dev` proxies /api to it.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  build: { outDir: "../web-dist", emptyOutDir: true, chunkSizeWarningLimit: 4096 },
  server: {
    port: 5273,
    proxy: { "/api": { target: "http://127.0.0.1:8600", changeOrigin: true } },
  },
});
