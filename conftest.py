import os

from pytest import Metafunc, Parser


def pytest_addoption(parser: Parser):
    parser.addoption("--long", action="store_false", help="If set, long tests will run")


def pytest_generate_tests(metafunc: Metafunc):
    print("ho")
    if metafunc.config.option.long:
        os.environ["LONG_TESTS"] = "True"
