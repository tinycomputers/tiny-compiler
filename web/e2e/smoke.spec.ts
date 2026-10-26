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
