# TinyCompiler

One tiny tensor language, two real chips: `C = relu(A @ B)` compiles to
[tiny-gpu](https://github.com/deaneeth/tiny-gpu) assembly or to a
[tiny-tpu](https://github.com/deaneeth/tiny-tpu) tile schedule, and runs on the real simulator / RTL.

**Status: Phase 8 (launch).** The [Compiler Explorer](web/) runs the whole compiler in the browser. The frontend, optimizing pass pipeline and **both backends** work. One program,
two chips, bit-exact against numpy on each:

```bash
tinyc examples/mm_relu.tiny --target tpu --run   # tile passes on the real tiny-tpu RTL
tinyc examples/mm_relu.tiny --target gpu --run   # SIMT kernels on the tiny-gpu simulator
```

`-O2` tiles operands into tiny-gpu's shared memory and unrolls the loops (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the measured effect of every optimization).
Both targets can be driven from pytest. See [docs/PLAN.md](docs/PLAN.md) for the roadmap,
[docs/LANGUAGE.md](docs/LANGUAGE.md), [docs/IR.md](docs/IR.md) and [docs/PASSES.md](docs/PASSES.md) and [docs/TTPU_FORMAT.md](docs/TTPU_FORMAT.md) and [docs/TGPU_FORMAT.md](docs/TGPU_FORMAT.md).

```bash
git clone --recurse-submodules <repo> && cd tiny-compiler
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pip install -e targets/tiny-gpu --no-deps
source scripts/env.sh
pytest
```

Requires Python 3.12+ and Node 20+ (the tiny-tpu WASM is loaded from the pinned submodule).

Everything is checked against a numpy oracle on the real chips; see [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Compiler Explorer (web)

```bash
cd web && npm install && npm run dev      # predev copies the Python bundle and the tiny-tpu WASM into web/public
```

Open `/app`: edit a program, pick an example and `-O` level, and see every pass, both chips' output, and
the results of running them on the real simulator and RTL.

## Verification

Every result is checked against a numpy oracle computed from the source program, on the real chips.
Details: [docs/VERIFICATION.md](docs/VERIFICATION.md).

| What | Count / result |
|---|---|
| Tests (CPython; run on every push) | 620 |
| Random programs, differential fuzzing, two independent runs | 5,000 + 5,000, no failures |
| Corner-case regression programs, replayed on every backend | 13 |
| tiny-gpu optimization levels checked on the simulator | -O0, -O1, -O2 |
| TPU cost model vs. measured RTL cycles | exact (pinned in `tests/perf_baseline.json`) |

Documentation: [docs](https://tiny-compiler.vercel.app/docs/) (generated from `docs/`).
