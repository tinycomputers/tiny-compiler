import numpy as np
import pytest

SEED = 42  # fixed seed for every test (CONTEXT.md conventions)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden", action="store_true", help="rewrite tests/golden/* from current output"
    )


@pytest.fixture
def golden(request):
    """`golden(path, actual)`: compare to a checked-in file; `--update-golden` rewrites it."""
    update = request.config.getoption("--update-golden")

    def check(path, actual: str) -> None:
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual)
            return
        assert path.exists(), f"missing golden file {path}; run `pytest --update-golden`"
        assert actual == path.read_text(), f"{path} differs; review, then `pytest --update-golden`"

    return check
