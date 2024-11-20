"""The Cirq library allows the user to instantiate a Cirq ``Circuit`` from an
OpenQASM 2.0 code.

The Cirq parser lacks native support for certain OpenQASM 2.0 operations such as
``cu1``, ``crz``, ``cu3``, ``reset``, ``u0``, ``p``, ``cp``, ``u``, ``rzz``,
``rxx`` and custom ``gate``. To address this limitation, we are redefining these
gates so you can use them on Cirq devices even though Cirq doesn't support it (a
behavior sometimes called *polyfill*, especially in the browser world). These
features are handled by :func:`qasm2_to_cirq_Circuit`.

In addition, Cirq does not handle user defined gates. So an important part of
:func:`qasm2_to_cirq_Circuit` is also a function which might be useful to you:
:func:`remove_user_gates`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cirq.circuits.circuit import Circuit as cirq_circuit

from typeguard import typechecked

from mpqp.qasm.open_qasm_2_and_3 import remove_user_gates


@typechecked
def qasm2_to_cirq_Circuit(qasm_str: str) -> "cirq_circuit":
    """
    Converting a OpenQASM 2.0 code into a cirq Circuit

    Args:
        qasm_str: a string representing the OpenQASM 2.0 code

    Returns:
        a Circuit equivalent to the QASM code in parameter

     Example:
        >>> qasm_code = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... '''
        >>> circuit = qasm2_to_cirq_Circuit(qasm_code)
        >>> print(circuit) # doctest: +NORMALIZE_WHITESPACE
        q_0: ───I───H───@───
                        │
        q_1: ───I───────X───

    """
    import numpy as np
    from cirq.circuits.qasm_output import QasmUGate
    from cirq.contrib.qasm_import._parser import QasmGateStatement, QasmParser
    from cirq.ops.common_channels import ResetChannel
    from cirq.ops.common_gates import ry, rz
    from cirq.ops.controlled_gate import ControlledGate
    from cirq.ops.global_phase_op import GlobalPhaseGate
    from cirq.ops.raw_types import Gate, Qid
    from cirq.ops.wait_gate import WaitGate
    from cirq.protocols.circuit_diagram_info_protocol import CircuitDiagramInfoArgs
    from cirq.value.duration import Duration

    qasm_str = remove_user_gates(qasm_str, skip_qelib1=True)

    class PhaseGate(Gate):
        def __init__(self, theta: complex):
            super(PhaseGate, self)
            self.theta = theta

        def _num_qubits_(self):
            return 1

        def _unitary_(self):
            return np.array([[1, 0], [0, np.exp(1j * self.theta)]])

        def _circuit_diagram_info_(self, args: CircuitDiagramInfoArgs):
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

        def _circuit_diagram_info_(self, args: CircuitDiagramInfoArgs):
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
                    [np.exp(-1j * self.theta / 2), 0, 0, 0],
                    [0, np.exp(1j * self.theta / 2), 0, 0],
                    [0, 0, np.exp(1j * self.theta / 2), 0],
                    [0, 0, 0, np.exp(-1j * self.theta / 2)],
                ]
            )

        def _circuit_diagram_info_(self, args: CircuitDiagramInfoArgs):
            return f"Rzz({self.theta})"

    class MyQasmUGate(QasmUGate):  # pyright: ignore[reportUntypedBaseClass]
        def __init__(
            self, theta, phi, lmda  # pyright: ignore[reportMissingParameterType]
        ) -> None:
            self.lmda = lmda
            self.theta = theta
            self.phi = phi

        def __repr__(self) -> str:
            return (
                f'U('
                f'theta={self.theta !r}, '
                f'phi={self.phi!r}, '
                f'lmda={self.lmda})'
            )

        def _decompose_(self, qubits: tuple[Qid, ...]):
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
            cirq_gate=(lambda params: ControlledGate(MyQasmUGate(0, 0, params[0]))),
        ),
        "cu3": QasmGateStatement(
            qasm_gate="cu3",
            num_params=3,
            num_args=2,
            cirq_gate=(
                lambda params: ControlledGate(MyQasmUGate(*[p for p in params]))
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
            cirq_gate=(lambda params: MyQasmUGate(*[p for p in params])),
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

    def p_new_reg2(self, p):  # pyright: ignore[reportMissingParameterType]
        """new_reg : QREG ID '[' NATURAL_NUMBER ']' ';'
        | CREG ID '[' NATURAL_NUMBER ']' ';'"""
        from cirq.ops.named_qubit import NamedQubit
        from cirq.contrib.qasm_import.exception import QasmException

        name, length = p[2], p[4]
        if name in self.qregs.keys() or name in self.cregs.keys():
            raise QasmException(f"{name} is already defined at line {p.lineno(2)}")
        if length == 0:
            raise QasmException(
                f"Illegal, zero-length register '{name}' at line {p.lineno(4)}"
            )
        if p[1] == "qreg":
            self.qregs[name] = length
            for idx in range(self.qregs[name]):
                arg_name = self.make_name(idx, name)
                if arg_name not in self.qubits.keys():
                    self.qubits[arg_name] = NamedQubit(arg_name)
                from cirq.ops.identity import I

                self.circuit.append(I(NamedQubit(arg_name)))
        else:
            self.cregs[name] = length
        p[0] = (name, length)

    qasm_parser.p_new_reg = p_new_reg2.__get__(qasm_parser)

    from ply import yacc

    qasm_parser.parser = yacc.yacc(module=qasm_parser, debug=False, write_tables=False)
    return qasm_parser.parse(qasm_str).circuit
