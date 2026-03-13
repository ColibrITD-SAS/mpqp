"""Package version for mpqp.
Release version values are managed by ``setuptools_scm`` from Git tags
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mpqp")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
