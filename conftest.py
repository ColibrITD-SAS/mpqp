from pytest import Parser


def pytest_addoption(parser: Parser):
    parser.addoption("--long", action="store_false", help="If set, long tests will run")
    parser.addoption(
        "--long-local",
        action="store_false",
        help="If set, only local long tests will run",
    )
