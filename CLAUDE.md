# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always read `CONTEXT.md` before any task.** The full roadmap is in `docs/PLAN.md`.

## Project Summary

TinyCompiler compiles a tiny tensor language (`.tiny`) to two real chips: tiny-gpu (`.tgpu`
assembly on its Python simulator) and tiny-tpu (`.ttpu.json` tile schedules on its
Verilator→WASM RTL). Pipeline: `.tiny → AST → Graph IR → Tile IR | Kernel IR → .ttpu.json | .tgpu`.

## Commands (planned; fill in as Phase 0 lands)

```bash
git submodule update --init
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pip install -e targets/tiny-gpu --no-deps
source scripts/env.sh
tinyc examples/mm_relu.tiny --target both -O2 --run --stats
pytest -q                                   # full suite incl. 200 fuzz examples
cd web && pnpm dev | build | lint | typecheck | test
```

## Branching
`main` is protected: work on `feat/*`, `fix/*`, `docs/*` branches and open PRs.
