import pytest

from mpqp.qasm import mpqp_to_qasm2
from mpqp.all import *
from mpqp.tools.circuit import random_circuit

@pytest.mark.parametrize(
    "instructions",
    [
        [X(0), Y(1), Z(2), H(3), CNOT(0, 1), CNOT(1, 2), BasisMeasure([0])],
        [
            X(0),
            X(1),
            X(2),
            X(3),
            Y(1),
            Z(2),
            H(3),
            CNOT(0, 1),
            CNOT(1, 2),
            BasisMeasure(),
        ],
        [
            S(0),
            X(0),
            Y(0),
            Z(0),
            P(np.pi, 0),
            U(np.pi, np.pi / 2, 2.5, 0),
            T(0),
            CNOT(0, 1),
            CRk(1, 1, 0),
            Rk(1, 1),
            CZ(0, 1),
            Rx(0, 1),
            Ry(0, 1),
            Rz(0, 1),
            TOF([0, 1], 2),
            BasisMeasure([0, 1, 2]),
        ],
        [
            P(np.pi, 0),
            P(np.pi, 1),
            P(np.pi / 2, 0),
            P(np.pi, 1),
            P(np.pi, 1),
            P(np.pi, 0),
            P(np.pi, 1),
        ],
        [
            P(np.pi, 0),
            P(np.pi, 1),
            P(np.pi / 2, 0),
            P(np.pi, 1),
            P(np.pi, 1),
            P(np.pi, 0),
            P(np.pi, 2),
        ],
        [X(0), X(0), X(1), Y(1), Y(0), Y(1), Y(0), Z(3)],
        [X(0), X(0), X(1), Barrier(), Y(1), Y(0), Y(0), Y(1)],
        [
            S(0),
            X(0),
            Y(0),
            Z(0),
            P(np.pi, 0),
            U(np.pi, np.pi / 2, 2.5, 0),
            T(0),
            CNOT(0, 1),
            CRk(1, 1, 0),
            Rk(1, 1),
            CZ(0, 1),
            Rx(0, 1),
            Ry(0, 1),
            Rz(0, 1),
            TOF([0, 1], 2),
        ],
        [Barrier()],
        [X(0), Y(1), BasisMeasure([0])],
        [CNOT(1, 0), CNOT(0, 1)],
    ],
)
def test_mpqp_to_qasm(instructions: list[Instruction]):
    circuit = QCircuit(instructions)
    assert mpqp_to_qasm2(circuit, False) == circuit.to_qasm2()


def test_random_mpqp_to_qasm():
    for _ in range(15):
        qcircuit = random_circuit(nb_qubits=6, nb_gates=20)
        assert mpqp_to_qasm2(qcircuit) == qcircuit.to_qasm2()
