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
        type=int,
        help="Set a global random seed for tests (default is None for random behavior).",
    )


@pytest.fixture
def global_seed(request: pytest.FixtureRequest):
    seed = request.config.getoption("--seed")
    return seed
