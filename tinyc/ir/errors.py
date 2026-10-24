"""Exception types shared by the IR layer."""


class OpTypeError(Exception):
    """An op's operands do not type-check (message is user-facing)."""


class VerifyError(Exception):
    """The IR violates a structural invariant. `problems` lists every violation found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("invalid IR:\n  " + "\n  ".join(problems))


class InputError(ValueError):
    """Inputs given to an evaluator do not match the function signature."""


class RangeError(ArithmeticError):
    """A value does not fit in i32. tiny-gpu raises on this, so the semantics reject it."""
