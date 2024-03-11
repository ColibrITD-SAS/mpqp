"""File regrouping all features for translating QASM code to Amazon Braket objects."""

import io
import warnings
from logging import StreamHandler, getLogger

from braket.circuits import Circuit
from braket.ir.openqasm import Program
from typeguard import typechecked

from mpqp.qasm.open_qasm_2_and_3 import open_qasm_hard_includes
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning


@typechecked
def qasm3_to_braket_Program(qasm3_str: str) -> Program:
    """Converting a OpenQASM 3.0 code into a Braket Program.

    Args:
        qasm3_str: A string representing the OpenQASM 3.0 code.

    Returns:
        A Program equivalent to the QASM code in parameter.
    """

    # PROBLEM: import and standard gates are not supported by Braket
    # NOTE: however custom OpenQASM 3 gates declaration is supported by Braket,
    # the idea is then to hard import the standard lib and other files into the qasm string's header before
    # giving it to the Program.
    after_stdgates_included = open_qasm_hard_includes(qasm3_str, set())

    program = Program(source=after_stdgates_included, inputs=None)
    return program


@typechecked
def qasm3_to_braket_Circuit(qasm3_str: str) -> Circuit:
    """Converting a OpenQASM 3.0 code into a Braket Circuit.

    Args:
        qasm3_str: A string representing the OpenQASM 3.0 code.

    Returns:
        A Circuit equivalent to the QASM code in parameter.
    """
    # PROBLEM: import and standard gates are not supported by Braket
    # NOTE: however custom OpenQASM 3 gates declaration is supported by Braket,
    # SOLUTION: the idea is then to hard import the standard lib and other files into the qasm string's header before
    # giving it to the Circuit.
    # PROBLEM2: Braket doesn't support NATIVE freaking gates U and gphase, so the trick may not work for the moment
    # for circuit, only for program
    # SOLUTION: import a specific qasm file with U and gphase redefined with the supported Braket SDK gates, and by
    # removing from this import file the already handled gates

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
            braket_logger.warning(message)

    return circuit
