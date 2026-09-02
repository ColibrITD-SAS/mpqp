from typing import TYPE_CHECKING

import numpy as np
import pytest

from mpqp import CNOT, CP, BasisMeasure, CustomGate, H, Language, Rx
from mpqp.translation.qasm.qasm_to_mpqp import qasm2_parse
from mpqp.tools.circuit import random_circuit


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
                BasisMeasure([0], [0]),
                BasisMeasure([1], [1]),
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
                BasisMeasure([0], [0]),
                BasisMeasure([1], [1]),
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
                BasisMeasure([0], [2]),
                BasisMeasure([1], [1]),
                BasisMeasure([2], [0]),
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
            measure q[1] -> c[1];""",
            [H(0), CNOT(0, 1), BasisMeasure([1], [0]), BasisMeasure([1], [1])],
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
                BasisMeasure([0], [2]),
                BasisMeasure([1], [1]),
                BasisMeasure([2], [0]),
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
    with pytest.raises(SyntaxError):
        qasm2_parse(qasm_code)


def test_invalid_qasm_header():
    with pytest.raises(SyntaxError, match="must start with OPENQASM 2.0"):
        qasm2_parse("NOTQASM 2.0; qreg q[1];")


def test_qasm_numeric_expression_uses_power_operator():
    circuit = qasm2_parse("OPENQASM 2.0; qreg q[1]; rx(2^3) q[0];")
    assert circuit.instructions == [Rx(8, 0)]


def test_custom_gate_pragma_accepts_only_literals():
    circuit = qasm2_parse(
        'OPENQASM 2.0; qreg q[1];\n'
        '#pragma mpqp CustomGate(array([[0, 1], [1, 0]]), [0], "X")\n'
    )
    assert circuit.instructions == [CustomGate(np.array([[0, 1], [1, 0]]), [0], "X")]


def test_custom_gate_pragma_rejects_function_calls(monkeypatch: pytest.MonkeyPatch):
    def unexpected_load(*args: object, **kwargs: object):
        pytest.fail("The custom-gate pragma executed an arbitrary NumPy function")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(SyntaxError, match="Invalid MPQP custom gate pragma"):
        qasm2_parse('OPENQASM 2.0; qreg q[1];\n#pragma mpqp np.load("untrusted.npy")\n')


def test_random_qasm_code():
    for _ in range(15):
        qcircuit = random_circuit(nb_qubits=6, nb_gates=20)
        qasm_code = qcircuit.to_other_language(Language.QASM2)
        if TYPE_CHECKING:
            assert isinstance(qasm_code, str)
        assert qcircuit.is_equivalent(qasm2_parse(qasm_code))
