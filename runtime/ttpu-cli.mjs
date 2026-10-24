#!/usr/bin/env node
// ttpu-cli.mjs - run tiny-tpu passes from a JSON file and print JSON results.
//
//   node runtime/ttpu-cli.mjs <passes.json>
//
//   in : { "passes": [ { "a": [16 ints], "b": [16 ints] }, ... ] }
//   out: { "passes": [ { "c": [16 ints], "cycles": n }, ... ], "totalCycles": n }
//
// The tiny-tpu build is taken from $TINY_TPU_DIST (a dir holding tiny_tpu.mjs + tiny_tpu.wasm),
// defaulting to the pinned submodule's web/public.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createTpu } from "./ttpu-exec.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const dist = process.env.TINY_TPU_DIST ?? join(here, "..", "targets", "tiny-tpu", "web", "public");

const [file] = process.argv.slice(2);
if (!file) {
  console.error("usage: ttpu-cli.mjs <passes.json>");
  process.exit(2);
}

const { default: factory } = await import(pathToFileURL(resolve(dist, "tiny_tpu.mjs")).href);
const tpu = await createTpu(factory);
const { passes } = JSON.parse(readFileSync(file, "utf8"));
const results = passes.map(({ a, b }) => {
  const { c, cycles } = tpu.runPass(a, b);
  return { c, cycles };
});
console.log(
  JSON.stringify({ passes: results, totalCycles: results.reduce((n, r) => n + r.cycles, 0) }),
);
