# Graph IR (text format)

One function, straight-line SSA. Values are `%name`: inputs keep their source names, op results
are numbered `%0, %1, ...`. Every type is printed without spaces (`i8[4,8]`).

```
func @main(%A: i8[4,8], %B: i8[8,4]) -> (i32[4,4]) {
  %0 = matmul %A, %B : i32[4,4]  loc("mm_relu.tiny":4:10)
  %1 = relu %0 : i32[4,4]        loc("mm_relu.tiny":4:5)
  return %1
}
```

| Op | Syntax |
|---|---|
| `const` | `%x = const [[1, 2], [3, 4]] : i8[2,2]` |
| `matmul` | `%x = matmul %a, %b : T` |
| `add` | `%x = add %a, %b : T` |
| `relu` | `%x = relu %a : T` |

- `loc("file":line:col)` is optional and points at the start of the source expression. It survives
  every lowering so the UI can map any output line back to the source.
- `//` starts a comment. Layout is free-form when parsing; the printer is canonical.
- `ir.printer.print_func` and `ir.parser.parse_func` round-trip exactly (tested over every example).

## Code map
| Module | Role |
|---|---|
| `ir/types.py` | `TensorType` and `infer_result_type`, the one source of typing rules |
| `ir/graph.py` | `Value`, `Op`, `Func`, `FuncBuilder` |
| `ir/verifier.py` | structural + type checks; reports every problem |
| `ir/ref.py` | **the oracle**: numpy int64 evaluation, `RangeError` on i32 overflow |
| `ir/interp.py` | independent scalar interpreter (plain Python ints), cross-checked against the oracle |

Inspect a program: `tinyc examples/mm_relu.tiny --locs` (or `--emit ast`).
