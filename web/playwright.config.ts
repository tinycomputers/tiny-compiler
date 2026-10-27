import { defineConfig, devices } from "@playwright/test";

// The smoke test runs against the static build in dist/ (the `e2e` script builds it first).
export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  use: { baseURL: "http://127.0.0.1:4321" },
  webServer: {
    // A plain static server: Astro 7's `preview` daemonises, which Playwright reads as an early exit.
    command: "python3 -m http.server 4321 --bind 127.0.0.1 --directory dist",
    url: "http://127.0.0.1:4321/app",
    reuseExistingServer: true,
    timeout: 60_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
