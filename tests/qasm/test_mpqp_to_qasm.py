import pytest

from mpqp.all import *
from mpqp.tools.circuit import random_circuit
from mpqp.qasm.mpqp_to_qasm import mpqp_to_qasm2


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
    from qiskit import qasm2, QuantumCircuit

    qiskit_circuit = circuit.to_other_language(Language.QISKIT)
    assert isinstance(qiskit_circuit, QuantumCircuit)
    assert qasm2.dumps(qiskit_circuit) == circuit.to_other_language(Language.QASM2)


@pytest.mark.parametrize(
    "instructions, qasm_expectation",
    [
        (
            [X(0), X(1), X(2), X(3), X(0), X(1), BasisMeasure()],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q;
x q[0],q[1];
measure q -> c;""",
        ),
        (
            [
                CustomGate(
                    UnitaryMatrix(
                        np.array(
                            [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]
                        )
                    ),
                    [1, 2],
                )
            ],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
u(0.6165457696943014,-pi/2,-pi) q[1];
u(pi/2,-pi/2,-pi) q[2];
cx q[2],q[1];
u(2.299175815834263,2.45724313480728,0.6843495187825104) q[1];
u(pi/2,-pi/2,-pi/2) q[2];
cx q[2],q[1];
u(1.0795118874220926,0.8555794527724245,-2.7526761788691734) q[1];
u(pi/2,-pi,pi/2) q[2];
cx q[2],q[1];
u(pi/2,2.5250468838954925,-pi) q[1];
u(pi/2,0,-pi/2) q[2];""",
        ),
        (
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
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q;
y q[1];
z q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;""",
        ),
        (
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
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
s q[0];
x q[0];
y q[0];
z q[0];
p(pi) q[0];
u(pi,pi/2,2.5) q[0];
t q[0];
cx q[0],q[1];
cp(pi) q[1],q[0];
p(pi) q[1];
cz q[0],q[1];
rx(0) q[1];
ry(0) q[1];
rz(0) q[1];
ccx q[0],q[1],q[2];
measure q -> c;""",
        ),
        (
            [
                P(np.pi, 0),
                P(np.pi, 1),
                P(np.pi / 2, 0),
                P(np.pi, 1),
                P(np.pi, 1),
                P(np.pi, 0),
                P(np.pi, 1),
            ],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
p(pi) q;
p(pi/2) q[0];
p(pi) q;
p(pi) q[1];
p(pi) q[1];""",
        ),
        (
            [
                P(np.pi, 0),
                P(np.pi, 1),
                P(np.pi / 2, 0),
                P(np.pi, 1),
                P(np.pi, 1),
                P(np.pi, 0),
                P(np.pi, 2),
            ],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
p(pi) q[0],q[1];
p(pi/2) q[0];
p(pi) q;
p(pi) q[1];""",
        ),
        (
            [X(0), X(0), X(1), Y(1), Y(0), Y(1), Y(0), Z(3)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
x q[0],q[1];
x q[0];
y q[0],q[1];
y q[0],q[1];
z q[3];""",
        ),
        (
            [X(0), X(0), X(1), Barrier(), Y(1), Y(0), Y(0), Y(1)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q;
x q[0];
barrier q[0],q[1];
y q;
y q;""",
        ),
        (
            [
                S(0),
                S(0),
                Y(0),
                Y(1),
                P(np.pi, 0),
                P(np.pi, 1),
                P(np.pi, 0),
                P(np.pi, 2),
                U(np.pi, np.pi / 2, 2.5, 0),
                T(0),
                CNOT(0, 1),
                CRk(1, 1, 0),
                Rk(1, 1),
                CZ(0, 1),
                Ry(0, 0),
                Ry(0, 1),
                Ry(0, 2),
                TOF([0, 1], 2),
            ],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
s q[0];
s q[0];
y q[0],q[1];
p(pi) q;
p(pi) q[0];
u(pi,pi/2,2.5) q[0];
t q[0];
cx q[0],q[1];
cp(pi) q[1],q[0];
p(pi) q[1];
cz q[0],q[1];
ry(0) q;
ccx q[0],q[1],q[2];""",
        ),
        (
            [Barrier()],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
barrier q[0];""",
        ),
        (
            [Y(0), Y(1), BasisMeasure([0])],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
y q;
measure q[0] -> c[0];""",
        ),
        (
            [CNOT(1, 0), CNOT(0, 1)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cx q[1],q[0];
cx q[0],q[1];""",
        ),
    ],
)
def test_mpqp_to_qasm_simplify(instructions: list[Instruction], qasm_expectation: str):
    circuit = QCircuit(instructions)
    qasm, _ = mpqp_to_qasm2(circuit, True)
    assert qasm_expectation == qasm


def test_random_mpqp_to_qasm():
    for _ in range(15):
        qcircuit = random_circuit(nb_qubits=6, nb_gates=20)
        from qiskit import qasm2, QuantumCircuit

        qiskit_circuit = qcircuit.to_other_language(Language.QISKIT)
        assert isinstance(qiskit_circuit, QuantumCircuit)
        assert qasm2.dumps(qiskit_circuit) == qcircuit.to_other_language(Language.QASM2)
