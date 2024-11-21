import numpy as np
import pytest
from cirq.circuits.circuit import Circuit
from cirq.ops.common_gates import CNOT as CirqCNOT
from cirq.ops.measure_util import measure
from cirq.ops.named_qubit import NamedQubit
from cirq.ops.pauli_gates import X as CirqX
from cirq.ops.identity import I as CirqI

from mpqp import QCircuit
from mpqp.core.instruction.measurement import ExpectationMeasure, Observable
from mpqp.execution import GOOGLEDevice, run
from mpqp.gates import *
from mpqp.measures import BasisMeasure
from mpqp.qasm import qasm2_to_cirq_Circuit


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


@pytest.mark.parametrize(
    "circuit, qasm_filename",
    [
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
    ],
)
def test_qasm2_to_cirq_Circuit(circuit: QCircuit, qasm_filename: str):
    # 3M-TODO test everything
    with open(
        f"tests/core/test_circuit/{qasm_filename}.qasm2",
        "r",
        encoding="utf-8",
    ) as f:
        assert qasm2_to_cirq_Circuit(f.read()) == circuit
