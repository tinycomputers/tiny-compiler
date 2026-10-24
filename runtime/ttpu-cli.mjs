#!/usr/bin/env node
// ttpu-cli.mjs - run tiny-tpu work from a JSON file and print JSON results.
//
//   node runtime/ttpu-cli.mjs <request.json>
//
//   request A (raw passes):  { "passes": [ { "a": [16 ints], "b": [16 ints] }, ... ] }
//     -> { "passes": [ { "c": [16 ints], "cycles": n }, ... ], "totalCycles": n }
//
//   request B (a program):   { "program": <.ttpu.json>, "inputs": { name: nested lists } }
//     -> { "outputs": { name: {shape, data} }, "passes": [ {id, cycles} ], "totalCycles": n,
//          "hostOps": { alloc, acc, epilogue, host_add, host_relu } }
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
  console.error("usage: ttpu-cli.mjs <request.json>");
  process.exit(2);
}

const { default: factory } = await import(pathToFileURL(resolve(dist, "tiny_tpu.mjs")).href);
const tpu = await createTpu(factory);
const request = JSON.parse(readFileSync(file, "utf8"));

try {
  if (request.program) {
    console.log(JSON.stringify(tpu.executeProgram(request.program, request.inputs ?? {})));
  } else {
    const results = request.passes.map(({ a, b }) => {
      const { c, cycles } = tpu.runPass(a, b);
      return { c, cycles };
    });
    console.log(
      JSON.stringify({ passes: results, totalCycles: results.reduce((n, r) => n + r.cycles, 0) }),
    );
  }
} catch (err) {
  console.error(`${err.name}: ${err.message}`);
  process.exit(1);
}
