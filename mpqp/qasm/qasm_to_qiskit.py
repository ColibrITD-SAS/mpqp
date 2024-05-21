"""The main object used to perform quantum computations in Qiskit is the
``QuantumCircuit``. Qiskit naturally supports OpenQASM 2.0 to instantiate a
circuit. One can remark that few remote devices also support OpenQASM 3.0 code,
this is not generalized yet to the whole library and device. We call the
function :func:`qasm2_to_Qiskit_Circuit` to generate the circuit from the qasm
code.
"""

from typing import TYPE_CHECKING

from typeguard import typechecked

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


@typechecked
def qasm2_to_Qiskit_Circuit(qasm_str: str) -> "QuantumCircuit":
    """Converting a OpenQASM 2.0 code into a Qiskit QuantumCircuit.

    Args:
        qasm_str: A string representing the OpenQASM 2.0 code.

    Returns:
        A QuantumCircuit equivalent to the QASM code in parameter.
    """
    from qiskit import QuantumCircuit

    return QuantumCircuit.from_qasm_str(qasm_str)
