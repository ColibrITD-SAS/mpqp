"""File regrouping all features for translating QASM code to cirq objects """
from cirq import Circuit
from cirq.contrib.qasm_import import circuit_from_qasm
from typeguard import typechecked



@typechecked
def qasm2_to_cirq_Circuit(qasm_str: str) -> Circuit:
    """
    Converting a OpenQASM 2.0 code into a cirq Circuit

    Args:
        qasm_str: a string representing the OpenQASM 2.0 code

    Returns:
        a Circuit equivalent to the QASM code in parameter
    """
    return circuit_from_qasm(qasm_str)
