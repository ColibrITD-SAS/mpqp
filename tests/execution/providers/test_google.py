from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from mpqp import (
    CNOT,
    CZ,
    SWAP,
    BasisMeasure,
    ExpectationMeasure,
    GOOGLEDevice,
    H,
    Observable,
    QCircuit,
    Rx,
    Ry,
    S,
    T,
    pZ,
    run,
)
from mpqp.translation.qasm.qasm_to_cirq import qasm2_to_cirq_Circuit

if TYPE_CHECKING:
    from cirq.circuits.circuit import Circuit


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("shots", [0, 1000])
def test_single_grouped_observable_preserves_variance(shots: int):
    from cirq.sim.sparse_simulator import Simulator

    circuit = QCircuit([H(0), ExpectationMeasure(Observable(pZ), shots=shots)])
    with patch("cirq.sim.sparse_simulator.Simulator", return_value=Simulator(seed=42)):
        result = run(circuit, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR)

    mean = result.expectation_values
    assert isinstance(mean, float)
    # For Pauli outcomes +/-1, the unbiased variance of the sample mean is
    # (1 - mean**2) / (shots - 1). Exact expectation values have zero variance.
    expected_variance = 0.0 if shots == 0 else (1 - mean**2) / (shots - 1)
    assert result.error == pytest.approx({"pZ": expected_variance})
    assert result.shots == shots


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "circuit",
    [
        # SAMPLE JOB
        QCircuit(
            [
                T(0),
                CNOT(0, 1),
                Ry(np.pi / 2, 2),
                S(1),
                CZ(2, 1),
                SWAP(2, 0),
                BasisMeasure(list(range(3)), shots=2000),
            ]
        ),
        # STATEVECTOR JOB
        QCircuit(
            [
                T(0),
                CNOT(0, 1),
                Ry(np.pi / 2, 2),
                S(1),
                CZ(2, 1),
                SWAP(2, 0),
                BasisMeasure(list(range(3)), shots=0),
            ]
        ),
        # OBSERVABLE JOB
        QCircuit(
            [
                H(0),
                Rx(1.76, 1),
                ExpectationMeasure(
                    observable=Observable(
                        np.array(
                            [
                                [0.63, 0.5, 1, 1],
                                [0.5, 0.82, 1, 1],
                                [1, 1, 1, 0.33],
                                [1, 1, 0.33, 0.3],
                            ],
                        )
                    ),
                    targets=[0, 1],
                    shots=1000,
                ),
            ]
        ),
    ],
)
def test_running_local_cirq(circuit: QCircuit):
    run(circuit, GOOGLEDevice.CIRQ_LOCAL_SIMULATOR)


@pytest.fixture
def list_cirq() -> list[tuple["Circuit", str]]:
    from cirq.circuits.circuit import Circuit
    from cirq.ops.common_gates import CNOT as CirqCNOT
    from cirq.ops.identity import I as CirqI
    from cirq.ops.measure_util import measure
    from cirq.ops.named_qubit import NamedQubit
    from cirq.ops.pauli_gates import X as CirqX

    return [
        (
            Circuit(
                CirqI(NamedQubit("q_0")),
                CirqI(NamedQubit("q_1")),
                CirqX(NamedQubit("q_0")),
                CirqCNOT(NamedQubit("q_0"), NamedQubit("q_1")),
                measure(NamedQubit("q_1"), key="c_1"),
                measure(NamedQubit("q_0"), key="c_0"),
            ),
            "all",
        )
    ]


@pytest.mark.provider("cirq")
def test_qasm2_to_cirq_Circuit(list_cirq: list[tuple[QCircuit, str]]):
    # 3M-TODO test everything
    for circuit, qasm_filename in list_cirq:
        with open(
            f"tests/core/test_circuit/{qasm_filename}.qasm2",
            "r",
            encoding="utf-8",
        ) as f:
            assert qasm2_to_cirq_Circuit(f.read()) == circuit
