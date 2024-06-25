import os

from pytest import Metafunc, Parser


def pytest_addoption(parser: Parser):
    parser.addoption("--long", action="store_false", help="If set, long tests will run")
    parser.addoption(
        "--longlocal", action="store_false", help="If set, local long tests will run"
    )


def pytest_generate_tests(metafunc: Metafunc):
    if metafunc.config.option.long:
        os.environ["LONG_TESTS"] = "True"
    elif metafunc.config.option.longlocal:
        os.environ["LONG_TESTS_LOCAL"] = "True"
