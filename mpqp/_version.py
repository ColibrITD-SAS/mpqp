"""Package version for mpqp.
Release version values are managed by ``setuptools_scm`` from Git tags
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mpqp")
except PackageNotFoundError:
    try:
        from _setuptools_scm_version import (  # pyright: ignore[reportMissingImports]
            __version__,
        )
    except ModuleNotFoundError:
        __version__ = "0.0.0+unknown"
