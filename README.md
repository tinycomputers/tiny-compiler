# TinyCompiler

One tiny tensor language, two real chips: `C = relu(A @ B)` compiles to
[tiny-gpu](https://github.com/deaneeth/tiny-gpu) assembly or to a
[tiny-tpu](https://github.com/deaneeth/tiny-tpu) tile schedule, and runs on the real simulator / RTL.

**Status: Phase 2.** The frontend, Graph IR and optimizing pass pipeline work
(`tinyc examples/mm_bias_relu.tiny --target gpu --print-after-all`); the chip backends come next.
Both targets can be driven from pytest. See [docs/PLAN.md](docs/PLAN.md) for the roadmap,
[docs/LANGUAGE.md](docs/LANGUAGE.md), [docs/IR.md](docs/IR.md) and [docs/PASSES.md](docs/PASSES.md).

```bash
git clone --recurse-submodules <repo> && cd tiny-compiler
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pip install -e targets/tiny-gpu --no-deps
source scripts/env.sh
pytest
```

Requires Python 3.12+ and Node 20+ (the tiny-tpu WASM is loaded from the pinned submodule).
