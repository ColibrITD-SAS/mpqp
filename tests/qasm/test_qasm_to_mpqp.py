from typing import TYPE_CHECKING

import pytest

from mpqp import CNOT, CP, BasisMeasure, H, Language
from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.native_gates import Rx
from mpqp.qasm.qasm_to_mpqp import qasm2_parse
from mpqp.tools.circuit import random_circuit
from sympy import Symbol


@pytest.mark.parametrize(
    "qasm_code, gate_names",
    [
        (
            """OPENQASM 2.0;
            """,
            [],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];

            measure q[0] -> c[0];
            measure q[1] -> c[1];""",
            [
                H(0),
                CNOT(0, 1),
                BasisMeasure([0, 1], [0, 1]),
            ],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[2];
            creg c[2];
            h q[0],q[1];
            cx q[0],q[1];

            measure q[0] -> c[0];
            measure q[1] -> c[1];""",
            [
                H(0),
                H(1),
                CNOT(0, 1),
                BasisMeasure([0, 1], [0, 1]),
            ],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            gate MyGate a, b {
                h a;
                cx a, b;
            }

            qreg q[2];
            creg c[2];

            MyGate q[0], q[1];

            measure q -> c;""",
            [H(0), CNOT(0, 1), BasisMeasure()],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            gate MyGate a, b {
                h a;
                cx a, b;
            }

            gate MyGate2 a, b, c{
                h a;
                cx a, c;
                h c;
            }

            qreg q[3];
            creg c[3];

            MyGate q[0], q[1];
            MyGate2 q[0], q[1], q[2];

            measure q -> c;""",
            [H(0), CNOT(0, 1), H(0), CNOT(0, 2), H(2), BasisMeasure()],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[3];
            cx q[0], q[1];
            cx q[1], q[2];""",
            [
                CNOT(0, 1),
                CNOT(1, 2),
            ],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[3];
            creg c[3];
            
            h q[0];
            cx q[0], q[1];
            measure q[0] -> c[2];
            measure q[1] -> c[1];
            measure q[2] -> c[0];""",
            [
                H(0),
                CNOT(0, 1),
                BasisMeasure([0, 1, 2], [2, 1, 0]),
            ],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            gate MyMixedGate a, b {
                h a;
                cx a, b;
                measure b -> c[0];
            }

            qreg q[2];
            creg c[2];

            MyMixedGate q[0], q[1];
            measure q[0] -> c[1];""",
            [H(0), CNOT(0, 1), BasisMeasure([1, 0], [0, 1])],
        ),
        (
            """OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[3];
            creg c[3];
            // this is a comment
            
            h q[0];
            cx q[0], q[1];
            cp(0.5) q[0], q[1];
            measure q[0] -> c[2];
            measure q[1] -> c[1];
            measure q[2] -> c[0];""",
            [
                H(0),
                CNOT(0, 1),
                CP(0.5, 0, 1),
                BasisMeasure([0, 1, 2], [2, 1, 0]),
            ],
        ),
    ],
)
def test_qasm2_to_mpqp(qasm_code: str, gate_names: list[str]):
    circ = qasm2_parse(qasm_code)
    for operations, expected_gate in zip(circ.instructions, gate_names):
        assert repr(operations) == repr(expected_gate)


@pytest.mark.parametrize(
    "qasm_code",
    [
        ("""OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[1];
            h q[0]
            cx q[0], """),
    ],
)
def test_invalid_qasm_code(qasm_code: str):
    try:
        qasm2_parse(qasm_code)
    except SyntaxError:
        pass


def test_random_qasm_code():
    for _ in range(15):
        qcircuit = random_circuit(nb_qubits=6, nb_gates=20)
        qasm_code = qcircuit.to_other_language(Language.QASM2)
        if TYPE_CHECKING:
            assert isinstance(qasm_code, str)
        assert qcircuit.is_equivalent(qasm2_parse(qasm_code))


θ = Symbol("θ")
σ = Symbol("σ")


@pytest.mark.parametrize(
    "qasm_code, expected",
    [
        (
            "OPENQASM 3.0;\ninclude 'stdgates.inc';\ninput float[64] θ;\nqubit[1] q;\nrx(θ) q[0];",
            QCircuit([Rx(θ, 0)]),
        ),
        (
            "OPENQASM 3.0;\ninclude 'stdgates.inc';\ninput float[64] θ;\n\ninput float[64] σ;\nqubit[1] q;\nrx(θ*σ) q[0];",
            QCircuit([Rx(θ * σ, 0)]),  # pyright: ignore[reportOperatorIssue]
        ),
        (
            "OPENQASM 3.0;\ninclude 'stdgates.inc';\ninput float[64] θ;\n\ninput float[64] σ;\nqubit[2] q;\nrx(6*σ) q[1]; \nrx(2*σ) q[0];",
            QCircuit(
                [Rx(6 * σ, 1), Rx(2 * σ, 0)]  # pyright: ignore[reportOperatorIssue]
            ),
        ),
    ],
)
def test_parametrized_circuit(qasm_code: str, expected: QCircuit):
    c = QCircuit.from_other_language(qasm_code)
    assert c == expected


@pytest.mark.parametrize(
    "qasm_code", ["OPENQASM 3.0;\ninclude 'stdgates.inc';\nqubit[1] q;\nrx(θ) q[0];"]
)
def test_parametrized_circuit_not_declared(qasm_code: str):
    with pytest.raises(
        ValueError,
        match="Variable: θ not found",
    ):
        QCircuit.from_other_language(qasm_code)
