// Records the hero clip: one program compiled and run on both chips, then the tiny-tpu array
// played cycle by cycle. Run against a built site (`pnpm build && pnpm preview --port 4322`):
//   node scripts/record-hero.mjs http://127.0.0.1:4322 <out-dir>
// Output: hero-raw.webm in <out-dir>. The shipped clip is trimmed from 5.5 s (past the loading
// screen) and re-encoded: ffmpeg -ss 5.5 -i hero-raw.webm -an -c:v libvpx-vp9 -crf 40 -vf scale=960:-2 public/hero.webm
import { chromium } from "@playwright/test";
import { mkdirSync, renameSync } from "node:fs";
import { join } from "node:path";

const [base = "http://127.0.0.1:4322", outDir = "hero-out"] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1280, height: 960 },
  recordVideo: { dir: outDir, size: { width: 1280, height: 960 } },
});
const page = await context.newPage();
await page.goto(`${base}/app`);
await page.getByRole("button", { name: "Compile & run" }).click();
await page.getByText("bit-exact").first().waitFor({ timeout: 150_000 });
await page.getByRole("tab", { name: "tiny-tpu" }).click();
await page.getByRole("img", { name: /^PE\[0\]\[0\] / }).first().scrollIntoViewIfNeeded();
await page.getByRole("button", { name: "Play" }).click();
await page.waitForTimeout(6000);
await page.getByRole("tab", { name: "Run & compare" }).click();
await page.waitForTimeout(3000);
const video = page.video();
await context.close();
await browser.close();
const path = await video.path();
const target = join(outDir, "hero-raw.webm");
renameSync(path, target);
console.log(target);
