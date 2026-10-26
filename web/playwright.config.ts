import { defineConfig, devices } from "@playwright/test";

// The smoke test runs against `astro preview` of a fresh build (see the `e2e` script).
export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  use: { baseURL: "http://127.0.0.1:4321" },
  webServer: {
    command: "pnpm preview --port 4321 --host 127.0.0.1",
    url: "http://127.0.0.1:4321/app",
    reuseExistingServer: true,
    timeout: 60_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
