"""File regrouping all features for translating QASM code to myQLM objects """
from qat.core.wrappers.circuit import Circuit
from qat.interop.openqasm import OqasmParser  # type: ignore
from typeguard import typechecked

from mpqp.qasm.open_qasm_2_and_3 import open_qasm_hard_includes


@typechecked
def qasm2_to_myqlm_Circuit(qasm_str: str) -> Circuit:
    """
    Converting a OpenQASM 2.0 code into a QLM Circuit.

    Args:
        qasm_str: A string representing the OpenQASM 2.0 code.

    Returns:
        A Circuit equivalent to the QASM code in parameter.
    """
    parser = OqasmParser(gates={"p": "PH", "u": "U"})  # requires myqlm-interop-1.9.3
    circuit = parser.compile(open_qasm_hard_includes(qasm_str, set()))
    return circuit
