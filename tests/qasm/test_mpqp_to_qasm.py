import pytest

from mpqp.all import *
from mpqp.tools.circuit import random_circuit
from mpqp.qasm.mpqp_to_qasm import mpqp_to_qasm2
from mpqp.qasm.open_qasm_2_and_3 import remove_user_gates
from mpqp.tools.display import format_element


@pytest.mark.parametrize(
    "instructions, qasm_expectation",
    [
        (
            [X(0), Y(1), Z(2), H(3), CNOT(0, 1), CNOT(1, 2), BasisMeasure([0])],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[1];
x q[0];
y q[1];
z q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
measure q[0] -> c[0];""",
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
x q[0];
x q[1];
x q[2];
x q[3];
y q[1];
z q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];""",
        ),
        (
            [
                S(0),
                X(0),
                Y(0),
                Z(0),
                P(3.141592653589793, 0),
                U(3.141592653589793, 1.5707963267948966, 2.5, 0),
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
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];""",
        ),
        (
            [
                P(3.141592653589793, 0),
                P(3.141592653589793, 1),
                P(1.5707963267948966, 0),
                P(3.141592653589793, 1),
                P(3.141592653589793, 1),
                P(3.141592653589793, 0),
                P(3.141592653589793, 1),
            ],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
p(pi) q[0];
p(pi) q[1];
p(pi/2) q[0];
p(pi) q[1];
p(pi) q[1];
p(pi) q[0];
p(pi) q[1];""",
        ),
        (
            [
                P(3.141592653589793, 0),
                P(3.141592653589793, 1),
                P(1.5707963267948966, 0),
                P(3.141592653589793, 1),
                P(3.141592653589793, 1),
                P(3.141592653589793, 0),
                P(3.141592653589793, 2),
            ],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
p(pi) q[0];
p(pi) q[1];
p(pi/2) q[0];
p(pi) q[1];
p(pi) q[1];
p(pi) q[0];
p(pi) q[2];""",
        ),
        (
            [X(0), X(0), X(1), Y(1), Y(0), Y(1), Y(0), Z(3)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
x q[0];
x q[0];
x q[1];
y q[1];
y q[0];
y q[1];
y q[0];
z q[3];""",
        ),
        (
            [X(0), X(0), X(1), Barrier(0), Y(1), Y(0), Y(0), Y(1)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
x q[0];
x q[1];
barrier q[0],q[1];
y q[1];
y q[0];
y q[0];
y q[1];""",
        ),
        (
            [
                S(0),
                X(0),
                Y(0),
                Z(0),
                P(3.141592653589793, 0),
                U(3.141592653589793, 1.5707963267948966, 2.5, 0),
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
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
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
ccx q[0],q[1],q[2];""",
        ),
        (
            [Barrier(1)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
barrier q[0];""",
        ),
        (
            [X(0), Y(1), BasisMeasure([0])],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
x q[0];
y q[1];
measure q[0] -> c[0];""",
        ),
        (
            [X(0), BasisMeasure([0]), Y(0)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
x q[0];
y q[0];
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
def test_mpqp_to_qasm_gate(instructions: list[Instruction], qasm_expectation: str):
    circuit = QCircuit(instructions)
    str_circuit = circuit.to_other_language(Language.QASM2)
    assert str_circuit == qasm_expectation


@pytest.mark.parametrize(
    "instructions",
    [
        [
            CustomGate(
                UnitaryMatrix(
                    np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
                ),
                [1, 2],
            )
        ]
    ],
)
def test_mpqp_to_qasm_custom_gate(instructions: list[Instruction]):
    circuit = QCircuit(instructions)
    from qiskit import QuantumCircuit, qasm2

    qiskit_circuit = circuit.to_other_language(Language.QISKIT)
    assert isinstance(qiskit_circuit, QuantumCircuit)
    str_qiskit_circuit = remove_user_gates(qasm2.dumps(qiskit_circuit), True)
    str_circuit = circuit.to_other_language(Language.QASM2)
    assert isinstance(str_circuit, str)
    for i in str_circuit:
        assert i in str_qiskit_circuit
    for i in str_qiskit_circuit:
        assert i in str_circuit


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
            [X(0), BasisMeasure([0]), Y(0)],
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
x q;
y q;
measure q -> c;""",
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
                CP(1, 1, 0),
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
cp(1) q[1],q[0];
p(pi) q[1];
cz q[0],q[1];
ry(0) q;
ccx q[0],q[1],q[2];""",
        ),
        (
            [Barrier(1)],
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


def normalize_string(string: str):
    import re
    from typing import Match

    def simplify_expression(match: Match[str]):
        from numpy import pi, e

        components = match.group(1).split(',')
        simplified = [
            format_element(eval(comp, {"pi": pi, "e": e}), 4) for comp in components
        ]
        return f"({','.join(simplified)})"

    pattern = r'\(([^()]+)\)'
    return re.sub(pattern, simplify_expression, string)


def test_random_mpqp_to_qasm():
    for _ in range(15):
        qcircuit = random_circuit(nb_qubits=6, nb_gates=20)
        from qiskit import QuantumCircuit, qasm2

        qiskit_circuit = qcircuit.to_other_language(Language.QISKIT)
        assert isinstance(qiskit_circuit, QuantumCircuit)
        qiskit_qasm = normalize_string(qasm2.dumps(qiskit_circuit))
        mpqp_qasm = qcircuit.to_other_language(Language.QASM2)
        assert isinstance(mpqp_qasm, str)
        mpqp_qasm = normalize_string(mpqp_qasm)
        assert qiskit_qasm == mpqp_qasm
