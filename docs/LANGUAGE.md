# The `.tiny` language (v1)

Integer tensors, static shapes, rank 1 or 2, no control flow. Statements end at a newline
(newlines inside `(...)` and `[...]` do not end a statement). `#` starts a comment.

```
input  A : i8[4, 8]                 # declare an input
const  W : i8[2, 2] = [[1, -2],     # declare a constant (shape and range are checked)
                       [3,  4]]
C = relu(A @ B + b)                 # define a value (names are single-assignment)
output C, D                         # one or more outputs (at least one is required)
```

Reserved words: `input const output relu i8 i32`.

## Types
`i8` or `i32`, with 1 or 2 positive dimensions: `i8[4, 8]`, `i32[6]`.

## Expressions
`@` binds tighter than `+`; both are left-associative; parentheses group.

| Expression | Rule | Result |
|---|---|---|
| `a @ b` | both rank 2, `a` is `[M,K]` and `b` is `[K,N]`; operand types may be `i8` or `i32` | `i32[M,N]` |
| `a + b` | same shape, or `[M,N] + [N]` in either order (the row is broadcast); `i8` operands are widened | `i32`, the larger shape |
| `relu(a)` | any | same shape and type as `a` |

## Semantics
All arithmetic is exact (widened); **every op result must fit in `i32`**, otherwise evaluation
fails with `RangeError`, because tiny-gpu raises on overflow rather than wrapping. Inputs must
fit their declared type.

## Errors
The first problem stops compilation and is shown with the source line and a caret:

```
matmul_inner_mismatch.tiny:3:5: error: matmul inner dimensions differ: 8 vs 4 (i8[4,8] @ i8[4,4])
3 | C = A @ B
  |     ^^^^^
```
Every message is snapshot-tested (`tests/golden/errors/`).

## Target legality (checked later, Phase 2)
Whether a program can run on a chip is not a language rule: tiny-tpu needs `i8` matmul operands,
and tiny-gpu has a memory budget. See `docs/PLAN.md` §3.3.
