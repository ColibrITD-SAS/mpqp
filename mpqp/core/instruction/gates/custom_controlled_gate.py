from typing import TYPE_CHECKING, Any, Optional, Union

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.languages import Language

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

    from mpqp.core.instruction.gates.gate import Gate


class CustomControlledGate(ControlledGate):
    """Class used to define a custom controlled gate.

    Args:
        controls: Indices referring to the qubits used to control the gate.
        gate: The original, non controlled, instance of the gate.
        label: Label used to identify the gate.

    Examples:
        >>> circuit = QCircuit(2)
        >>> circuit.add(CustomControlledGate(0, Y(1)))
        >>> pprint(circuit.to_matrix())
        [[1, 0, 0 , 0  ],
         [0, 1, 0 , 0  ],
         [0, 0, 0 , -1j],
         [0, 0, 1j, 0  ]]
        >>> print(circuit)  # doctest: +NORMALIZE_WHITESPACE
        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ Y ├
             └───┘
        >>> circuit = QCircuit(3)
        >>> circuit.add(CustomControlledGate([0,2], CustomGate(np.array([[1,0],[0,-1]]),[1])))
        >>> print(circuit)  # doctest: +NORMALIZE_WHITESPACE
        q_0: ─────■─────
             ┌────┴────┐
        q_1: ┤ Unitary ├
             └────┬────┘
        q_2: ─────■─────

    """

    def __init__(
        self,
        controls: Union[list[int], int],
        gate: "Gate",
        label: Optional[str] = None,
    ):
        if isinstance(gate, ControlledGate):
            if isinstance(controls, int):
                controls = [controls]
            controls += gate.controls
            ControlledGate.__init__(
                self, controls, gate.targets, gate.non_controlled_gate, label
            )
        else:
            ControlledGate.__init__(self, controls, gate.targets, gate, label)

    def __repr__(self) -> str:
        return f"CustomControlledGate({self.controls}, {self.non_controlled_gate.__repr__()})"

    def to_canonical_matrix(self):
        import numpy as np

        l = 2 ** len(self.targets)
        m = np.identity(2 ** (len(self.controls) + len(self.targets)), dtype=complex)
        m[-l:, -l:] = self.non_controlled_gate.to_matrix()
        return m

    def inverse(self) -> "CustomControlledGate":
        return CustomControlledGate(self.controls, self.non_controlled_gate.inverse())

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ) -> Any:
        if language == Language.QISKIT:
            from qiskit.quantum_info import Operator

            gate = self.non_controlled_gate.to_other_language()
            if isinstance(gate, Operator):
                gate = gate.to_instruction()
            gate = gate.control(len(self.controls))
            return gate
        elif language == Language.QASM2:
            from qiskit import QuantumCircuit, qasm2

            nb_qubits = max(max(self.targets), max(self.controls)) + 1

            qiskit_circ = QuantumCircuit(nb_qubits)

            if isinstance(self.non_controlled_gate, CustomGate):
                targets = self.targets + self.controls
                targets.sort()
                gate = CustomGate(self.to_matrix(), targets)

                return gate.to_other_language(Language.QASM2)

            else:
                qiskit_circ.append(
                    self.to_other_language(Language.QISKIT),
                    self.controls + self.targets,
                )
            qasm_str = qasm2.dumps(qiskit_circ)
            qasm_lines = qasm_str.splitlines()
            if isinstance(self.non_controlled_gate, CustomGate):
                return qasm_str, 0
            instructions_only = [
                line
                for line in qasm_lines
                if not (
                    line.startswith("qreg")
                    or line.startswith("include")
                    or line.startswith("creg")
                    or line.startswith("OPENQASM")
                )
            ]

            return "\n".join(instructions_only), 0
        else:
            raise NotImplementedError(f"Error: {language} is not supported")
