import pytest

from mpqp.qasm.qasm_to_cirq import qasm2_to_cirq_Circuit
from cirq.circuits.circuit import Circuit


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
                "I",
                "I",
                "H",
                "CNOT",
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
            [
                "I",
                "I",
                "H",
                "CNOT",
            ],
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
            [
                "I",
                "I",
                "I",
                "H",
                "CNOT",
            ],
        ),
    ],
)
def test_qasm2_to_Cirq_Circuit(qasm_code: str, gate_names: list[str]):
    circ = qasm2_to_cirq_Circuit(qasm_code)
    assert isinstance(circ, Circuit)
    for operations, expected_gate in zip(circ.all_operations(), gate_names):
        assert str(operations.gate) == expected_gate
