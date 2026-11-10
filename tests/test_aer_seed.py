"""Regression checks for the pytest Aer seed option."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tests import _aer_seed
from tests._aer_seed import AerSeedPlugin


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("seed", [0, 351])
def test_aer_seed_streams(
    seed: int, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    aer = pytest.importorskip("qiskit_aer")

    def record_seed(_simulator: Any, *_args: Any, **kwargs: Any) -> Any:
        return kwargs.get("seed_simulator")

    monkeypatch.setattr(aer.AerSimulator, "run", record_seed)
    plugin = AerSeedPlugin(seed)
    plugin.install()
    request.addfinalizer(plugin.uninstall)
    simulator = aer.AerSimulator()

    setup_seeds = [simulator.run(None), simulator.run(None)]
    plugin.pytest_runtest_protocol(request.node)  # pyright: ignore[reportArgumentType]
    test_seeds = [simulator.run(None), simulator.run(None)]
    assert test_seeds[0] != test_seeds[1]
    assert test_seeds != setup_seeds

    # Another test and its call count cannot change this test's stream.
    plugin.reset("test:another-test")
    simulator.run(None)
    plugin.pytest_runtest_protocol(request.node)  # pyright: ignore[reportArgumentType]
    assert [simulator.run(None), simulator.run(None)] == test_seeds

    plugin.pytest_runtest_protocol(request.node)  # pyright: ignore[reportArgumentType]
    assert simulator.run(None, seed_simulator=0) == 0
    assert simulator.run(None, seed_simulator=None) is None
    configured_simulator = aer.AerSimulator(seed_simulator=123)
    assert configured_simulator.run(None) is None
    # Explicit seeds do not consume the automatically generated sequence.
    assert simulator.run(None) == test_seeds[0]
    assert configured_simulator.options.seed_simulator == 123

    plugin.reset("database-setup")
    assert [simulator.run(None), simulator.run(None)] == setup_seeds
    plugin.uninstall()
    assert aer.AerSimulator.run is record_seed


@pytest.mark.provider("qiskit")
def test_aer_seed_reproduces_sampling(request: pytest.FixtureRequest):
    aer = pytest.importorskip("qiskit_aer")
    qiskit = pytest.importorskip("qiskit")
    plugin = AerSeedPlugin(351)
    plugin.install()
    request.addfinalizer(plugin.uninstall)
    circuit = qiskit.QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    simulator = aer.AerSimulator()

    plugin.pytest_runtest_protocol(request.node)  # pyright: ignore[reportArgumentType]
    first = [simulator.run(circuit, shots=1024).result().get_counts() for _ in range(2)]
    plugin.pytest_runtest_protocol(request.node)  # pyright: ignore[reportArgumentType]
    replay = [
        simulator.run(circuit, shots=1024).result().get_counts() for _ in range(2)
    ]
    assert replay == first


def test_aer_seed_without_optional_sdk(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_aer_seed, "find_spec", lambda _name: None)
    plugin = AerSeedPlugin(351)
    plugin.install()
    plugin.uninstall()


def test_aer_seed_only_configures_explicit_option(monkeypatch: pytest.MonkeyPatch):
    installed: list[AerSeedPlugin] = []
    registered: list[AerSeedPlugin] = []
    cleanups: list[Any] = []
    monkeypatch.setattr(
        AerSeedPlugin, "install", lambda plugin: installed.append(plugin)
    )
    config = Mock(spec=pytest.Config)
    config.getoption.return_value = None
    config.add_cleanup.side_effect = cleanups.append
    config.pluginmanager = Mock()
    config.pluginmanager.register.side_effect = lambda plugin, _name: registered.append(
        plugin
    )
    # Exercise configuration without needing to import an optional SDK.
    _aer_seed.configure_aer_seed(config)
    assert installed == registered == cleanups == []
    config.getoption.return_value = 0
    _aer_seed.configure_aer_seed(config)
    assert installed == registered
    assert len(installed) == len(cleanups) == 1
    assert installed[0].seed == 0
    cleanups[0]()


@pytest.mark.provider("qiskit")
def test_aer_seed_pytest_lifecycle(tmp_path: Path):
    """Replay setup and fixture streams across processes and test orders."""
    pytest.importorskip("qiskit_aer")
    (tmp_path / "conftest.py").write_text(
        '''
import json
from pathlib import Path

import pytest
from qiskit_aer import AerSimulator

from tests._aer_seed import configure_aer_seed

streams = {}

def pytest_addoption(parser):
    parser.addoption("--seed", type=int, default=None)

def pytest_configure(config):
    # Observe actual run options without executing a simulator in subprocesses.
    def record_seed(self, *args, **kwargs):
        return kwargs.get("seed_simulator")
    AerSimulator.run = record_seed
    configure_aer_seed(config)
    simulator = AerSimulator()
    streams["database-setup"] = [simulator.run(None), simulator.run(None)]

@pytest.fixture(autouse=True)
def fixture_stream(request):
    simulator = AerSimulator()
    streams[request.node.nodeid] = [simulator.run(None), simulator.run(None)]
    yield
    streams[request.node.nodeid].append(simulator.run(None))

def pytest_unconfigure(config):
    Path("streams.json").write_text(json.dumps(streams))
''',
        encoding="utf-8",
    )
    (tmp_path / "test_streams.py").write_text(
        "def test_first(): pass\ndef test_second(): pass\n", encoding="utf-8"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    def run_tests(order: list[str], seed: int | None) -> dict[str, list[int | None]]:
        command = [sys.executable, "-m", "pytest", "-q", *order]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        completed = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads((tmp_path / "streams.json").read_text(encoding="utf-8"))

    first = "test_streams.py::test_first"
    second = "test_streams.py::test_second"
    forward = run_tests([first, second], 351)
    reverse = run_tests([second, first], 351)
    assert forward == reverse
    assert len(set(forward[first])) == 3
    assert forward[first] != forward[second]
    assert forward["database-setup"] != forward[first][:2]

    unseeded = run_tests([first, second], None)
    assert all(value is None for stream in unseeded.values() for value in stream)
