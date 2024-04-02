"""File regrouping all features for translating QASM code to cirq objects """

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cirq.circuits.circuit import Circuit as cirq_circuit

from typeguard import typechecked


@typechecked
def qasm2_to_cirq_Circuit(qasm_str: str) -> "cirq_circuit":
    """
    Converting a OpenQASM 2.0 code into a cirq Circuit

    Args:
        qasm_str: a string representing the OpenQASM 2.0 code

    Returns:
        a Circuit equivalent to the QASM code in parameter
    """
    import numpy as np
    from cirq.contrib.qasm_import._parser import QasmParser, QasmGateStatement
    from cirq.circuits.qasm_output import QasmUGate
    from cirq.ops.controlled_gate import ControlledGate
    from cirq.ops.common_gates import rz
    from cirq.ops.common_channels import ResetChannel

    lines = qasm_str.split("\n")

    # Remove the line containing the barrier keyword
    modified_lines = [line for line in lines if "barrier" not in line]
    qasm_str = "\n".join(modified_lines)

    qasm_parser = QasmParser()
    qs_dict = {
        "cu1": QasmGateStatement(
            qasm_gate="cu1",
            num_params=1,
            num_args=2,
            cirq_gate=(
                lambda params: ControlledGate(QasmUGate(0, 0, params[0] / np.pi))
            ),
        ),
        "crz": QasmGateStatement(
            qasm_gate="crz",
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ControlledGate(rz(params[0]))),
        ),
        "cu3": QasmGateStatement(
            qasm_gate="cu3",
            num_params=3,
            num_args=2,
            cirq_gate=(
                lambda params: ControlledGate(QasmUGate(*[p / np.pi for p in params]))
            ),
        ),
        "reset": QasmGateStatement(
            qasm_gate="reset", num_params=0, num_args=1, cirq_gate=ResetChannel()
        ),
        "u0": QasmGateStatement(
            qasm_gate="u0", num_params=1, num_args=1, cirq_gate=QasmUGate(0, 0, 0)
        ),
    }
    qasm_parser.all_gates |= qs_dict

    return qasm_parser.parse(qasm_str).circuit
