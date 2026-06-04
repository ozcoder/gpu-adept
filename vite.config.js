import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "gpu-msss": path.resolve(__dirname, "node_modules/gpu-msss/src/msss.js"),
    },
  },
  server: {
    open: true,
  },
});
