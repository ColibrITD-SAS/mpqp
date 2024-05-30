import pytest
from braket.circuits import Circuit, Operator
from braket.circuits.gates import CNot, H

from mpqp.qasm.qasm_to_braket import qasm3_to_braket_Circuit
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning


@pytest.mark.parametrize(
    "qasm_code, braket_operators",
    [
        (
            """OPENQASM 3.0;""",
            [],
        ),
    ],
)
def test_qasm3_to_braket_Circuit(qasm_code: str, braket_operators: list[Operator]):
    circ = qasm3_to_braket_Circuit(qasm_code)

    assert isinstance(circ, Circuit)
    for circ_instr, expected_operator in zip(circ.instructions, braket_operators):
        assert circ_instr.operator == expected_operator


@pytest.mark.parametrize(
    "qasm_code, braket_operators",
    [
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
def test_qasm3_to_braket_Circuit_warning(
    qasm_code: str, braket_operators: list[Operator]
):
    warning = (
        "This program uses OpenQASM language features that may not be supported"
        " on QPUs or on-demand simulators."
    )
    with pytest.warns(UnsupportedBraketFeaturesWarning, match=warning):
        circ = qasm3_to_braket_Circuit(qasm_code)

    assert isinstance(circ, Circuit)
    for circ_instr, expected_operator in zip(circ.instructions, braket_operators):
        assert circ_instr.operator == expected_operator
