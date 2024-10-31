from typing import TYPE_CHECKING, Any

import pytest
from numpy.random import default_rng, randint


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--long", action="store_false", help="If set, long tests will run")
    parser.addoption(
        "--long-costly",
        action="store_false",
        help="If set, long tests that cost credit will run",
    )
    parser.addoption(
        "--long-local",
        action="store_false",
        help="If set, only local long tests will run",
    )
    parser.addoption(
        "--seed",
        action="store",
        default=None,
        type=int,
        help="Set a global random seed for tests (default is None for random behavior).",
    )


@pytest.fixture(autouse=True)
def mock_random(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    seed = request.config.getoption("--seed")
    if TYPE_CHECKING:
        assert isinstance(seed, int) or isinstance(seed, type(None))
    if seed is None:
        seed = randint(0, 1024)
    print(f"Using seed {seed}")

    def stable_random(*args: Any, **kwargs: Any):
        user_seed = args[0] if len(args) != 0 else None
        return default_rng(user_seed or seed)

    monkeypatch.setattr('numpy.random.default_rng', stable_random)
