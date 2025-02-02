import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  server: {
    proxy: {
      // Proxying API requests from / to a local server on port 5078
      "/": {
        target: "http://localhost:5555",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      // You can add more proxy rules here
    },
  },
  plugins: [react()],
});
