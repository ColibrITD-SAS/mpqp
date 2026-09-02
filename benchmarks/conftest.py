"""Shared, deterministic data for MPQP performance benchmarks."""

from collections.abc import Callable
from pathlib import Path

import pytest

from mpqp import CNOT, H, QCircuit, Rx, Rz
from mpqp.core.instruction.instruction import Instruction


def pytest_sessionstart(session: pytest.Session) -> None:
    """Keep all generated benchmark files away from the repository root."""
    Path(".benchmarks/results").mkdir(parents=True, exist_ok=True)


def build_layered_circuit(nb_qubits: int, nb_layers: int) -> QCircuit:
    """Build a representative circuit without randomness or external providers."""
    instructions: list[Instruction] = []
    for layer in range(nb_layers):
        angle = (layer + 1) / 100
        for qubit in range(nb_qubits):
            instructions.extend([H(qubit), Rx(angle, qubit), Rz(-angle, qubit)])
        instructions.extend(CNOT(qubit, qubit + 1) for qubit in range(nb_qubits - 1))
    return QCircuit(instructions)


@pytest.fixture(scope="session")
def circuit_factory() -> Callable[[int, int], QCircuit]:
    return build_layered_circuit


@pytest.fixture(scope="session")
def medium_circuit(circuit_factory: Callable[[int, int], QCircuit]) -> QCircuit:
    return circuit_factory(12, 30)


@pytest.fixture(scope="session")
def matrix_circuit(circuit_factory: Callable[[int, int], QCircuit]) -> QCircuit:
    return circuit_factory(7, 4)


@pytest.fixture(scope="session")
def medium_qasm() -> str:
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "qreg q[12];",
    ]
    for layer in range(30):
        angle = (layer + 1) / 100
        lines.extend(f"h q[{qubit}];" for qubit in range(12))
        lines.extend(f"rx({angle}) q[{qubit}];" for qubit in range(12))
        lines.extend(f"cx q[{qubit}],q[{qubit + 1}];" for qubit in range(11))
    return "\n".join(lines)
