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
