"""File regrouping all features for translating QASM code to Qiskit objects """
from qiskit import QuantumCircuit
from typeguard import typechecked


@typechecked
def qasm2_to_QuantumCircuit(qasm_str: str) -> QuantumCircuit:
    """
    Converting a OpenQASM 2.0 code into a Qiskit QuantumCircuit.

    Args:
        qasm_str: A string representing the OpenQASM 2.0 code.

    Returns:
        A QuantumCircuit equivalent to the QASM code in parameter.
    """
    return QuantumCircuit.from_qasm_str(qasm_str)
