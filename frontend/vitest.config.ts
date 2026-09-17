import { fileURLToPath } from "url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  // No CSS-dependent tests here (see lib/reducer.test.ts). Vite 5 (pulled
  // in by vitest) can't load postcss.config.mjs's `@tailwindcss/postcss`
  // plugin (ESM/CJS interop mismatch, unrelated to Next's own bundler) --
  // an inline empty postcss config bypasses that file lookup entirely.
  css: { postcss: { plugins: [] } },
  resolve: {
    // Mirrors tsconfig.json's `"@/*": ["./*"]` -- Next resolves this on its
    // own, but Vite (which vitest runs on) doesn't read tsconfig paths,
    // so anything under test that imports via `@/...` (e.g. lib/inline.tsx
    // importing components/SourceTile) needs it spelled out here too.
    alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
  },
  // Next compiles JSX itself (tsconfig.json's "jsx": "preserve"), but
  // vitest's default esbuild transform needs its own jsx setting or it
  // emits bare `<span>` calls expecting a global `React` -- "automatic"
  // matches Next's own react-jsx runtime (no React import needed per file).
  esbuild: { jsx: "automatic" },
  test: {
    environment: "node",
  },
});
