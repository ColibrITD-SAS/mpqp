from typing import Optional

import pytest


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--long", action="store_false", help="If set, long tests will run")
    parser.addoption(
        "--long-local",
        action="store_false",
        help="If set, only local long tests will run",
    )
    parser.addoption(
        "--seed",
        action="store",
        default=None,
        help="Set a global random seed for tests (default is None for random behavior).",
    )


@pytest.fixture
def global_seed(request: pytest.FixtureRequest) -> Optional[int]:
    seed = request.config.getoption("--seed")

    if seed is None or seed == 'None':
        print("No seed is provided, using the default random behavior.")
        return None

    if isinstance(seed, str):
        try:
            seed = int(seed)
            print(f"\nSetting global random seed to {seed}")
            return seed
        except ValueError:
            print(f"Invalid seed value: {seed}. Seed is an integer or None.")
            return None
    else:
        print(f"Invalid seed type: {type(seed)}")
        return None
