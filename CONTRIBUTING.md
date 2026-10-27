# Contributing

Thanks for looking at TinyCompiler. It is a small compiler for a small tensor language, and the
bar for changes is that every result stays checked against a numpy oracle on the real chips.

## Set up

```bash
git submodule update --init
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pip install -e targets/tiny-gpu --no-deps
source scripts/env.sh
```

For the Compiler Explorer:

```bash
cd web && pnpm install && pnpm dev
```

`pnpm dev` regenerates the Python bundle and the example list first (`scripts/build_web.py`).

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check . && black --check .
pytest -q
cd web && pnpm lint && pnpm typecheck && pnpm test && pnpm e2e
```

- **Golden files** (`tests/golden/`) pin the emitted IR and assembly. If you change emitted
  output on purpose, review the diff and run `pytest --update-golden`.
- **Measured numbers** in `docs/ARCHITECTURE.md` come from `scripts/opt_table.py --write`. A test
  fails if they drift.
- Do not format `targets/` (the hardware submodules); the formatter config excludes it.

## Branches and pull requests

- Work on a `feat/*`, `fix/*` or `docs/*` branch. `main` is the default branch and is not pushed to
  directly.
- Keep a pull request to one change. Describe what changed and how you checked it.
- Add a regression test for each bug you fix. A minimized failing program can go in
  `tests/regressions/`.

## Reporting bugs

Open an issue with the `.tiny` program that fails, the command you ran, and what you expected. A
minimal program makes the fix much faster. For security problems, see [SECURITY.md](SECURITY.md).
