// Dev helper: bundle once, render a list of global frames to out/stills/*.png
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";
import { readFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const timing = JSON.parse(readFileSync("src/timing.json", "utf8"));
const spec = process.argv.slice(2); // like 3:60 4:100 or g:1234
const frames = spec.map((s) => {
  const [a, b] = s.split(":");
  return a === "g" ? { name: `g${b}`, frame: +b } : { name: `s${a}_${b}`, frame: timing.scenes[+a - 1].start + +b };
});
const serveUrl = await bundle({ entryPoint: path.resolve("src/index.ts") });
const comp = await selectComposition({ serveUrl, id: "TinyCompilerIntro" });
mkdirSync("out/stills", { recursive: true });
for (const { name, frame } of frames) {
  await renderStill({ composition: comp, serveUrl, frame, output: `out/stills/${name}.png`, scale: 0.75 });
  console.log("rendered", name);
}
