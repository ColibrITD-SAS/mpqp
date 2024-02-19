from braket.circuits import Operator, Circuit
from braket.circuits.gates import H, CNot
import pytest

from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit


@pytest.mark.parametrize(
    "qasm_code, braket_operators",
    [
        (
            """OPENQASM 3.0;""",
            [],
        ),
        (
            """OPENQASM 3.0;
            include 'stdgates.inc';

            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0],q[1];

            c[0] = measure q[0];
            c[1] = measure q[1];""",
            [H(), CNot()],
        ),
    ],
)
def test_qasm3_to_braket_Circuit(qasm_code: str, braket_operators: list[Operator]):
    circ = qasm3_to_braket_Circuit(qasm_code)
    assert isinstance(circ, Circuit)
    for circ_instr, expected_operator in zip(circ.instructions, braket_operators):
        assert circ_instr.operator == expected_operator
