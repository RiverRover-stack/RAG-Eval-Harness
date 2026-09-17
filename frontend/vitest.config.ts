import { defineConfig } from "vitest/config";

export default defineConfig({
  // No component/CSS tests here (see lib/reducer.test.ts). Vite 5 (pulled
  // in by vitest) can't load postcss.config.mjs's `@tailwindcss/postcss`
  // plugin (ESM/CJS interop mismatch, unrelated to Next's own bundler) --
  // an inline empty postcss config bypasses that file lookup entirely.
  css: { postcss: { plugins: [] } },
  test: {
    environment: "node",
  },
});
