"""Test `mpqp.__version__`."""

from importlib.metadata import PackageNotFoundError, version

import mpqp


def test_version():
    try:
        installed_version = version("mpqp")
    except PackageNotFoundError:
        assert mpqp.__version__ == "0.0.0+unknown"
    else:
        assert mpqp.__version__ == installed_version
