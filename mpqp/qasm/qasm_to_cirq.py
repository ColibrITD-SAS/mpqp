"""File regrouping all features for translating QASM code to cirq objects """

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cirq.circuits.circuit import Circuit as cirq_circuit

from typeguard import typechecked
from mpqp.qasm.qasm_remplace_custom_gate import replace_custom_gates

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
    from cirq.ops.common_gates import rz, ry
    from cirq.ops.common_channels import ResetChannel
    from cirq.ops.wait_gate import WaitGate
    from cirq.value.duration import Duration
    from cirq.ops.global_phase_op import GlobalPhaseGate
    from cirq.ops.raw_types import Gate

    qasm_str = replace_custom_gates(qasm_str)
    

    """Define a custom single-qubit gate."""

    class PhaseGate(Gate):
        def __init__(self, theta: complex):
            super(PhaseGate, self)
            self.theta = theta

        def _num_qubits_(self):
            return 1

        def _unitary_(self):
            return np.array(
                [[np.exp(1j * self.theta), 0], 
                 [0, np.exp(1j * self.theta)]]
            )

        def _circuit_diagram_info_(self, args):
            return f"P({self.theta})"

    class Rxx(Gate):
        def __init__(self, theta: complex):
            super(Rxx, self)
            self.theta = theta

        def _num_qubits_(self):
            return 2

        def _unitary_(self):
            return np.array(
                [
                    [np.cos(self.theta / 2), 0, 0, -1j * np.sin(self.theta / 2)],
                    [0, np.cos(self.theta / 2), -1j * np.sin(self.theta / 2), 0],
                    [0, -1j * np.sin(self.theta / 2), np.cos(self.theta / 2), 0],
                    [-1j * np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)],
                ]
            )

        def _circuit_diagram_info_(self, args):
            return f"Rxx({self.theta})"
    
    class Rzz(Gate):
        def __init__(self, theta: complex):
            super(Rzz, self)
            self.theta = theta

        def _num_qubits_(self):
            return 2

        def _unitary_(self):
            return np.array(
                [
                    [np.exp(-1j*self.theta / 2), 0, 0, 0],
                    [0, np.exp(1j*self.theta / 2), 0, 0],
                    [0, 0, np.exp(1j*self.theta / 2), 0],
                    [0, 0, 0, np.exp(-1j*self.theta / 2)],
                ]
            )

        def _circuit_diagram_info_(self, args):
            return f"Rzz({self.theta})"

    class MyQasmUGate(QasmUGate):
        def _decompose_(self, qubits):
            q = qubits[0]
            return [
                GlobalPhaseGate(np.exp(1j * (self.lmda + self.phi) / 2)).on(),
                rz(self.lmda).on(q),
                ry(self.theta).on(q),
                rz(self.phi).on(q),
            ]

    # Remove the line containing the barrier keyword
    modified_lines = [line for line in qasm_str.split("\n") if "barrier" not in line]
    qasm_str = "\n".join(modified_lines)

    qasm_parser = QasmParser()
    qs_dict = {
        "cu1": QasmGateStatement(
            qasm_gate="cu1",
            num_params=1,
            num_args=2,
            cirq_gate=(
                lambda params: ControlledGate(MyQasmUGate(0, 0, params[0] / np.pi))
            ),
        ),
        "cu3": QasmGateStatement(
            qasm_gate="cu3",
            num_params=3,
            num_args=2,
            cirq_gate=(
                lambda params: ControlledGate(MyQasmUGate(*[p / np.pi for p in params]))
            ),
        ),
        "crz": QasmGateStatement(
            qasm_gate="crz",
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ControlledGate(rz(params[0]))),
        ),
        "reset": QasmGateStatement(
            qasm_gate="reset", num_params=0, num_args=1, cirq_gate=ResetChannel()
        ),
        "u0": QasmGateStatement(
            qasm_gate="u0",
            num_params=1,
            num_args=1,
            cirq_gate=(lambda params: WaitGate(Duration(micros=params[0]))),
        ),
        "p": QasmGateStatement(
            qasm_gate="p",
            num_params=1,
            num_args=1,
            cirq_gate=(lambda params: PhaseGate(params[0])),
        ),
        "cp": QasmGateStatement(
            qasm_gate="cp",
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ControlledGate(PhaseGate(params[0]))),
        ),
        "u": QasmGateStatement(
            qasm_gate="u3",
            num_params=3,
            num_args=1,
            cirq_gate=(lambda params: MyQasmUGate(*[p / np.pi for p in params])),
        ),
        "rxx": QasmGateStatement(
            qasm_gate="rxx",
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: Rxx(params[0])),
        ),
        "rzz": QasmGateStatement(
            qasm_gate="rzz",
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: Rzz(params[0])),
        ),
    }
    qasm_parser.all_gates |= qs_dict

    return qasm_parser.parse(qasm_str).circuit
