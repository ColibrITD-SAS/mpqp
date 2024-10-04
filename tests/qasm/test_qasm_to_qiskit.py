import pytest

from mpqp.qasm.qasm_to_qiskit import qasm2_to_Qiskit_Circuit


@pytest.mark.parametrize(
    "qasm_code, gate_names",
    [
        (
            """OPENQASM 2.0;""",
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
                "h",
                "cx",
            ],
        ),
    ],
)
def test_qasm2_to_Qiskit_Circuit(qasm_code: str, gate_names: list[str]):
    from qiskit import QuantumCircuit

    circ = qasm2_to_Qiskit_Circuit(qasm_code)
    assert isinstance(circ, QuantumCircuit)
    for instr, expected_gate in zip(circ.data, gate_names):
        assert instr.operation.name == expected_gate
