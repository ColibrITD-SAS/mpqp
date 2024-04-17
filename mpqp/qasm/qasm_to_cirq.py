"""File regrouping all features for translating QASM code to cirq objects """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cirq.circuits.circuit import Circuit

from typeguard import typechecked



@typechecked
def qasm2_to_cirq_Circuit(qasm_str: str) -> "Circuit":
    """
    Converting a OpenQASM 2.0 code into a cirq Circuit

    Args:
        qasm_str: a string representing the OpenQASM 2.0 code

    Returns:
        a Circuit equivalent to the QASM code in parameter
    """
    from cirq.contrib.qasm_import.qasm import circuit_from_qasm

    if "include \"qelib1.inc\";" not in qasm_str:
        qasm_str = "include \"qelib1.inc\";\n" + qasm_str

    # NOTE: the cu1 gate is not supported by cirq
    return circuit_from_qasm(qasm_str)
