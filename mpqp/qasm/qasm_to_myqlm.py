"""The myQLM library allows the user to instantiate a myQLM ``Circuit`` from an
OpenQASM 2.0 code. MyQLM is able to parse most of the standard gates, and allows
us to complete the missing gates by linking them to already defined ones. We
call the function :func:`qasm2_to_myqlm_Circuit` to generate the circuit from
the qasm code."""

import re
from typing import TYPE_CHECKING

from mpqp.qasm.open_qasm_2_and_3 import open_qasm_hard_includes
from typeguard import typechecked

if TYPE_CHECKING:
    from qat.core.wrappers.circuit import Circuit


@typechecked
def qasm2_to_myqlm_Circuit(qasm_str: str) -> "Circuit":
    """Converting a OpenQASM 2.0 code into a QLM Circuit.

    Args:
        qasm_str: A string representing the OpenQASM 2.0 code.

    Returns:
        A Circuit equivalent to the QASM code in parameter.

    Example:
        >>> qasm_code = '''
        ... OPENQASM 2.0;
        ... qreg q[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... '''
        >>> circuit = qasm2_to_myqlm_Circuit(qasm_code)
        >>> circuit.display(batchmode=True) # doctest: +NORMALIZE_WHITESPACE
          ┌─┐
         ─┤H├─●─
          └─┘ │
              │
             ┌┴┐
         ────┤X├
             └─┘

    """
    from qat.interop.openqasm import OqasmParser

    parser = OqasmParser(gates={"p": "PH", "u": "U"})  # requires myqlm-interop-1.9.3

    # We replace 'sdg' for S_dagger gate by 'u1(pi/2)', because of problem on myqlm side
    pattern = re.compile(r'(?<!gate\s)sdg\s+(\S+)\s*;')
    updated_qasm = pattern.sub(r'u1(-pi/2) \1;', qasm_str)

    circuit = parser.compile(open_qasm_hard_includes(updated_qasm, set()))
    return circuit
