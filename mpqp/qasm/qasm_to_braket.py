"""Amazon Braket made the choice to directly support a subset of OpenQASM 3.0 
for gate-based devices and simulators. In fact, Braket supports a set of data
types, statements and pragmas (specific to Braket) for OpenQASM 3.0, sometimes
with a different syntax.

Braket Circuit parser does not support for the moment the OpenQASM 3.0 native
operations (``U`` and ``gphase``) but allows to define custom gates using a
combination of supported standard gates (``rx``, ``ry``, ``rz``, ``cnot``,
``phaseshift`` for instance). Besides, the inclusion of files is not yet handled
by Braket library meaning we use a mechanism of *hard* includes (see
:func:`~mpqp.qasm.qasm_to_myqlm.hard-open_qasm_hard_includes`)
directly in the OpenQASM 3.0 code, to be sure the parser and interpreter have
all definitions in there. We also hard-include all included files in the
OpenQASM 3.0 code inputted for conversion.

.. note::
    In the custom hard-imported file for native and standard gate redefinitions, 
    we use ``ggphase`` to define the global phase, instead of the OpenQASM 3.0 
    keyword ``gphase``, which is already used and protected by Braket.

Braket ``Circuit``s are created using :func:`qasm3_to_braket_Circuit`. If
needed, you can also generate a Braket ``Program`` from an OpenQASM 3.0 input
string using the :func:`qasm3_to_braket_Program`. However, in this case, the
program parser does not need to redefine the native gates, and thus only
performing a hard import of standard gates and other included file is
sufficient. However, note that a ``Program`` cannot be used to retrieve the
statevector and expectation value in Braket.
"""

import io
import warnings
from logging import StreamHandler, getLogger
from typing import TYPE_CHECKING

from typeguard import typechecked

if TYPE_CHECKING:
    from braket.ir.openqasm import Program
    from braket.circuits import Circuit

from mpqp.qasm.open_qasm_2_and_3 import open_qasm_hard_includes
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning


@typechecked
def qasm3_to_braket_Program(qasm3_str: str) -> "Program":
    r"""Converting a OpenQASM 3.0 code into a Braket Program.

    Args:
        qasm3_str: A string representing the OpenQASM 3.0 code.

    Returns:
        A Program equivalent to the QASM code in parameter.

    Example:
        >>> qasm_code = '''
        ... OPENQASM 3.0;
        ... qubit[2] q;
        ... h q[0];
        ... '''
        >>> program = qasm3_to_braket_Program(qasm_code)
        >>> print(program)
        braketSchemaHeader=BraketSchemaHeader(name='braket.ir.openqasm.program', version='1') source='\nOPENQASM 3.0;\nqubit[2] q;\nh q[0];\n' inputs=None

    """
    from braket.ir.openqasm import Program

    # PROBLEM: import and standard gates are not supported by Braket
    # NOTE: however custom OpenQASM 3 gates declaration is supported by Braket,
    # the idea is then to hard import the standard lib and other files into the qasm string's header before
    # giving it to the Program.
    after_stdgates_included = open_qasm_hard_includes(qasm3_str, set())

    program = Program(source=after_stdgates_included, inputs=None)
    return program


@typechecked
def qasm3_to_braket_Circuit(qasm3_str: str) -> "Circuit":
    """Converting a OpenQASM 3.0 code into a Braket Circuit.

    Args:
        qasm3_str: A string representing the OpenQASM 3.0 code.

    Returns:
        A Circuit equivalent to the QASM code in parameter.

    Example:
        >>> qasm_code = '''
        ... OPENQASM 3.0;
        ... qubit[2] q;
        ... h q[0];
        ... '''
        >>> circuit = qasm3_to_braket_Circuit(qasm_code)
        >>> print(circuit) # doctest: +NORMALIZE_WHITESPACE
        T  : │  0  │
              ┌───┐
        q0 : ─┤ H ├─
              └───┘
        T  : │  0  │

    """
    # PROBLEM: import and standard gates are not supported by Braket
    # NOTE: however custom OpenQASM 3 gates declaration is supported by Braket,
    # SOLUTION: the idea is then to hard import the standard lib and other files into the qasm string's header before
    # giving it to the Circuit.
    # PROBLEM2: Braket doesn't support NATIVE freaking gates U and gphase, so the trick may not work for the moment
    # for circuit, only for program
    # SOLUTION: import a specific qasm file with U and gphase redefined with the supported Braket SDK gates, and by
    # removing from this import file the already handled gates
    from braket.circuits import Circuit

    qasm3_str = qasm3_str.replace("stdgates.inc", "braket_custom_include.inc")

    after_stdgates_included = open_qasm_hard_includes(qasm3_str, set())

    braket_warning_message = (
        "This program uses OpenQASM language features that may not be supported"
        " on QPUs or on-demand simulators."
    )

    braket_logger = getLogger()
    logger_output_stream = io.StringIO()
    stream_handler = StreamHandler(logger_output_stream)
    braket_logger.addHandler(stream_handler)

    circuit = Circuit.from_ir(after_stdgates_included)

    braket_logger.removeHandler(stream_handler)
    log_lines = logger_output_stream.getvalue().split("\n")
    for message in log_lines:
        if message == braket_warning_message:
            warnings.warn(
                "\n" + braket_warning_message, UnsupportedBraketFeaturesWarning
            )
        else:
            if message != "":
                braket_logger.warning(message)

    return circuit
