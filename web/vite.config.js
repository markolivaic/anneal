import { defineConfig } from "vite";

// The API runs as a separate Python process (uvicorn). Proxying it here keeps
// the browser on one origin, so there is no CORS configuration to get wrong.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        // Server-Sent Events must not be buffered by the proxy or the whole
        // point of streaming is lost.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
            }
          });
        },
      },
    },
  },
});
