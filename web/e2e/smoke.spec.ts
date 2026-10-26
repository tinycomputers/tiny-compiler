import { expect, test } from "@playwright/test";

// The first compile downloads Pyodide and numpy, so it is excluded from the warm-cache budget.
const WARM_BUDGET_MS = 5000;

test.describe("Compiler Explorer smoke", () => {
  test("compiles and runs the default example bit-exact on both chips", async ({ page }) => {
    await page.goto("/app");
    const button = page.getByRole("button", { name: "Compile & run" });
    await button.click();
    await expect(page.getByText("bit-exact").first()).toBeVisible({ timeout: 150_000 });
    const rows = page.getByRole("row").filter({ hasText: "%1" });
    await expect(rows.getByText("bit-exact")).toHaveCount(2);
  });

  test("a warm second compile finishes within the budget", async ({ page }) => {
    await page.goto("/app");
    const button = page.getByRole("button", { name: /Compile & run|Working/ });
    await button.click();
    await expect(page.getByText("bit-exact").first()).toBeVisible({ timeout: 150_000 });
    await expect(page.getByRole("button", { name: "Compile & run" })).toBeEnabled();

    const start = Date.now();
    await page.getByRole("button", { name: "Compile & run" }).click();
    await expect(page.getByRole("button", { name: "Working…" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Compile & run" })).toBeEnabled({ timeout: WARM_BUDGET_MS });
    expect(Date.now() - start).toBeLessThan(WARM_BUDGET_MS);
    await expect(page.getByText("bit-exact").first()).toBeVisible();
  });

  test("the tiny-tpu array plays by itself, cycle by cycle, from the RTL trace", async ({ page }) => {
    await page.goto("/app");
    await page.getByRole("button", { name: "Compile & run" }).click();
    await expect(page.getByText("bit-exact").first()).toBeVisible({ timeout: 150_000 });
    await page.getByRole("tab", { name: "tiny-tpu" }).click();
    await expect(page.getByRole("img", { name: /^PE\[0\]\[0\] / }).first()).toBeVisible();
    // Playing on arrival: the cycle counter moves without any click.
    await expect(page.getByText(/cycle [1-9]\d* \/ \d+/)).toBeVisible({ timeout: 10_000 });
    await page.getByRole("button", { name: "Pause" }).click();
    await page.getByRole("button", { name: "pass 2" }).click();
    await expect(page.getByText(/cycle 0 \/ \d+/)).toBeVisible();
  });

  test("the pass replay steps through the compiler stages", async ({ page }) => {
    await page.goto("/app");
    await page.getByRole("button", { name: "Compile & run" }).click();
    await expect(page.getByText("bit-exact").first()).toBeVisible({ timeout: 150_000 });
    await page.getByRole("tab", { name: "Passes" }).click();
    await page.getByRole("button", { name: "Replay the passes" }).click();
    // Each stage lasts under a second, so wait for the replay to finish rather than catch one stage.
    await expect(page.getByRole("button", { name: "Replay the passes" })).toBeVisible({ timeout: 60_000 });
    const stage = await page.getByText(/stage \d+ of \d+/).innerText();
    const [, last] = stage.match(/of (\d+)/) ?? [];
    expect(stage).toBe(`stage ${last} of ${last}`);
  });

  test("choosing an example replaces the editor's source", async ({ page }) => {
    await page.goto("/app");
    await page.getByLabel("example").selectOption("mlp_gpu");
    await expect(page.getByRole("textbox", { name: "source" })).toContainText("H = relu(X @ W1 + b1)");
  });

  test("the landing page plays the hero clip and links to the explorer", async ({ page }) => {
    await page.goto("/");
    const video = page.locator("video");
    await expect(video).toHaveAttribute("src", "/hero.webm");
    const ready = await video.evaluate((v: HTMLVideoElement) => v.readyState >= 1 || new Promise((r) => (v.onloadedmetadata = () => r(true))));
    expect(ready).toBeTruthy();
    await expect(page.getByRole("link", { name: "Open the Compiler Explorer" })).toHaveAttribute("href", "/app");
  });

  test("a rejected program is shown as an error, not a crash", async ({ page }) => {
    await page.goto("/app");
    await page.getByLabel("source").fill("input A : i8[2, 2]\nC = A @ A +\noutput C\n");
    await page.getByRole("button", { name: "Compile & run" }).click();
    await expect(page.getByText("expected an expression")).toBeVisible({ timeout: 150_000 });
  });

  test("phone width has no horizontal scroll", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/app");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });

  test("the page renders in both light and dark color schemes", async ({ page }) => {
    for (const scheme of ["light", "dark"] as const) {
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto("/app");
      const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      expect(bg).not.toBe("rgba(0, 0, 0, 0)");
      await expect(page.getByRole("button", { name: "Compile & run" })).toBeVisible();
    }
  });
});
