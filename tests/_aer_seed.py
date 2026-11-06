"""Test-only Aer seed sequences, independent of NumPy and test ordering."""

from functools import wraps
from hashlib import sha256
from importlib.util import find_spec
from random import Random
from typing import Any

import pytest


class AerSeedPlugin:
    """Seed Aer runs by phase and test ID without overriding explicit seeds."""

    def __init__(self, seed: int):
        self.seed = seed
        self.monkeypatch = pytest.MonkeyPatch()
        self.reset("database-setup")

    def reset(self, scope: str) -> None:
        """Start a reproducible stream without using Python's randomized hash."""
        digest = sha256(f"{self.seed}:{scope}".encode("utf-8")).digest()
        self.random = Random(int.from_bytes(digest, "big"))

    def install(self) -> None:
        """Patch Aer only when its optional SDK is installed."""
        if find_spec("qiskit_aer") is None:
            return

        from qiskit_aer import AerSimulator

        original_run = AerSimulator.run

        @wraps(original_run)
        def seeded_run(simulator: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                "seed_simulator" not in kwargs
                and simulator.options.seed_simulator is None
            ):
                kwargs["seed_simulator"] = self.random.getrandbits(32)
            return original_run(simulator, *args, **kwargs)

        self.monkeypatch.setattr(AerSimulator, "run", seeded_run)

    def uninstall(self) -> None:
        """Restore Aer when the pytest configuration is cleaned up."""
        self.monkeypatch.undo()

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item: pytest.Item) -> None:
        """Reset before setup fixtures, execution and teardown of each test."""
        self.reset(f"test:{item.nodeid}")


def configure_aer_seed(config: pytest.Config) -> None:
    """Install session seeding before the local storage database is populated."""
    seed = config.getoption("--seed")
    if seed is None:
        return
    plugin = AerSeedPlugin(seed)  # pyright: ignore[reportArgumentType]
    config.add_cleanup(plugin.uninstall)
    plugin.install()
    config.pluginmanager.register(plugin, "mpqp-aer-seed")
