import pytest

from mpqp.qasm.qasm_to_myqlm import qasm2_to_myqlm_Circuit
from qat.core.wrappers.circuit import Circuit


@pytest.mark.parametrize(
    "qasm_code, gate_names",
    [
        (
            """OPENQASM 3.0;""",
            [],
        ),
        (
            """OPENQASM 2.0;

            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];

            measure q[0] -> c[0];
            measure q[1] -> c[1];""",
            ["H", "CNOT"],
        ),
    ],
)
def test_qasm2_to_myqlm_Circuit(qasm_code: str, gate_names: list[str]):
    circ = qasm2_to_myqlm_Circuit(qasm_code)
    assert isinstance(circ, Circuit)
    for op, expected_gate in zip(circ.ops, gate_names):
        assert op.gate == expected_gate
