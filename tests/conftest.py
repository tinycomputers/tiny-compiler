import numpy as np
import pytest

SEED = 42  # fixed seed for every test (CONTEXT.md conventions)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
