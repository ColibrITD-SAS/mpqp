from typing import Any, Optional, Union

from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.languages import Language
from mpqp.tools.generics import Matrix


class CustomControlledGate(ControlledGate):
    """
    Class used to define a custom controlled gate.
    It can be either a native gate with any numbers of control qubits or a custom gate with control qubits.

    Args:
        controls: List of indices referring to the qubits used to control the gate.
        targets: List of indices referring to the qubits on which the gate will be applied.
        non_controlled_gate: The original, non controlled, gate or a matrix.
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
    >>> circuit.add(CustomControlledGate([0, 2], CustomGate(np.array([[1, 0], [0, -1]]),[1])))
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
        gate: Union[NativeGate, CustomGate],
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

    def to_matrix(self, desired_gate_size: int = 0) -> Matrix:
        from functools import reduce

        import numpy as np

        if len(self.controls) != 1 or len(self.targets) != 1:
            from mpqp.core.instruction.gates.native_gates import SWAP

            controls, targets = self.controls, self.targets
            min_qubit, max_qubit = min(self.connections()), max(self.connections())

            # If nb_qubits is not provided, calculate the necessary number of minimal qubits
            if desired_gate_size == 0:
                desired_gate_size = max_qubit - min_qubit + 1
                controls = [x - min_qubit for x in controls]
                targets = [x - min_qubit for x in targets]
            elif desired_gate_size < max_qubit + 1:
                raise ValueError(f"nb_qubits must be at least {max_qubit + 1}")

            canonical_matrix = np.kron(
                self.to_canonical_matrix(),
                np.eye(2 ** (desired_gate_size - self.nb_qubits)),
            )

            permutations = set(
                tuple(sorted(idx))
                for idx in enumerate(controls + targets)
                if idx[0] != idx[1] and not set(idx).issubset(controls)
            )

            swaps = [
                SWAP(canonical_index, actual_index).to_matrix(desired_gate_size)
                for canonical_index, actual_index in permutations
            ]

            return reduce(np.dot, swaps[::-1] + [canonical_matrix] + swaps)

        control, target = self.controls[0], self.targets[0]

        if desired_gate_size != 0:
            max_qubit = max(control, target) + 1
            if desired_gate_size < max_qubit:
                raise ValueError(f"nb_qubits must be at least {max_qubit}")
        else:
            min_qubit = min(control, target)
            control -= min_qubit
            target -= min_qubit
            desired_gate_size = abs(control - target) + 1

        zero = np.diag([1, 0]).astype(np.complex64)
        one = np.diag([0, 1]).astype(np.complex64)
        non_controlled_gate = self.non_controlled_gate.to_matrix()
        I2 = np.eye(2, dtype=np.complex64)

        control_matrix = zero if control == 0 else I2
        target_matrix = (
            one if control == 0 else (non_controlled_gate if target == 0 else I2)
        )

        for i in range(1, desired_gate_size):
            if i == control:
                target_matrix = np.kron(target_matrix, one)
                control_matrix = np.kron(control_matrix, zero)
            elif i == target:
                target_matrix = np.kron(target_matrix, non_controlled_gate)
                control_matrix = np.kron(control_matrix, I2)
            else:
                target_matrix = np.kron(target_matrix, I2)
                control_matrix = np.kron(control_matrix, I2)

        return control_matrix + target_matrix

    def to_canonical_matrix(self):
        import numpy as np

        l = 2 ** len(self.targets)
        m = np.identity(2 ** (len(self.controls) + len(self.targets)), dtype=complex)
        m[-l:, -l:] = self.non_controlled_gate.to_matrix()
        return m

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
