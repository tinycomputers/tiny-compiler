# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always read `CONTEXT.md` before any task.** The full roadmap is in `docs/PLAN.md`.

## Project Summary

TinyCompiler compiles a tiny tensor language (`.tiny`) to two real chips: tiny-gpu (`.tgpu`
assembly on its Python simulator) and tiny-tpu (`.ttpu.json` tile schedules on its
Verilator→WASM RTL). Pipeline: `.tiny → AST → Graph IR → Tile IR | Kernel IR → .ttpu.json | .tgpu`.

## Commands (backends and `--run` arrive in later phases)

```bash
git submodule update --init
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pip install -e targets/tiny-gpu --no-deps
source scripts/env.sh
tinyc examples/mm_relu.tiny --locs          # optimized Graph IR (--emit ast, -O0, --target gpu|tpu,
                                            #   --print-after-all, --verify-each, --time-passes, --disable PASS)
tinyc examples/mm_relu.tiny --target tpu --run          # compile + run on the real RTL (also --emit tile|ttpu, --stats)
pytest -q                                   # full suite
pytest --update-golden                      # rewrite tests/golden/* after reviewing the diff
cd web && pnpm dev | build | lint | typecheck | test
```

## Branching
`main` is protected: work on `feat/*`, `fix/*`, `docs/*` branches and open PRs.

## Commit and PR rules
- **Never add Claude (or any AI) as author or co-author** of a commit, PR or anything else: no
  `Co-Authored-By: Claude ...` trailer, no "Generated with Claude Code" footer, no AI `--author`.
  Commits carry only the repository owner's git identity.
- Before pushing: `git log --format=%B | grep -i co-authored` must print nothing.
