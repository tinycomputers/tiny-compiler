# TinyCompiler

One tiny tensor language, two real chips: `C = relu(A @ B)` compiles to
[tiny-gpu](https://github.com/deaneeth/tiny-gpu) assembly or to a
[tiny-tpu](https://github.com/deaneeth/tiny-tpu) tile schedule, and runs on the real simulator / RTL.

**Status: Phase 4.** The frontend, optimizing pass pipeline and **both backends** work. One program,
two chips, bit-exact against numpy on each:

```bash
tinyc examples/mm_relu.tiny --target tpu --run   # tile passes on the real tiny-tpu RTL
tinyc examples/mm_relu.tiny --target gpu --run   # SIMT kernels on the tiny-gpu simulator
```

GPU kernel optimizations are next.
Both targets can be driven from pytest. See [docs/PLAN.md](docs/PLAN.md) for the roadmap,
[docs/LANGUAGE.md](docs/LANGUAGE.md), [docs/IR.md](docs/IR.md) and [docs/PASSES.md](docs/PASSES.md) and [docs/TTPU_FORMAT.md](docs/TTPU_FORMAT.md) and [docs/TGPU_FORMAT.md](docs/TGPU_FORMAT.md).

```bash
git clone --recurse-submodules <repo> && cd tiny-compiler
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pip install -e targets/tiny-gpu --no-deps
source scripts/env.sh
pytest
```

Requires Python 3.12+ and Node 20+ (the tiny-tpu WASM is loaded from the pinned submodule).
