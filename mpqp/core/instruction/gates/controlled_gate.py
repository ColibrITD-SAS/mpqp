from __future__ import annotations

from abc import ABC
from typing import Optional

from typeguard import typechecked

from mpqp.tools.generics import Matrix

from .gate import Gate


@typechecked
class ControlledGate(Gate, ABC):
    """Abstract class representing a controlled gate, that can be controlled by
    one or several qubits.

    Args:
        controls: List of indices referring to the qubits used to control the gate.
        targets: List of indices referring to the qubits on which the gate will be applied.
        non_controlled_gate: The original, non controlled, gate.
        label: Label used to identify the gate.
    """

    def __init__(
        self,
        controls: list[int],
        targets: list[int],
        non_controlled_gate: Gate,
        label: Optional[str] = None,
    ):
        if len(set(controls)) != len(controls):
            raise ValueError(f"Duplicate registers in controls: {controls}")
        if len(set(controls).intersection(set(targets))):
            raise ValueError(
                f"Common registers between targets {targets} and controls {controls}"
            )
        self.controls = controls
        """See parameter description."""
        self.non_controlled_gate = non_controlled_gate
        """See parameter description."""

        Gate.__init__(self, targets, label)

    def controlled_gate_to_matrix(self) -> Matrix:
        """
        Constructs the matrix representation of a controlled gate.

        Returns:
            The matrix representation of the controlled gate.
        """
        import numpy as np

        if len(self.controls) != 1 or len(self.targets) != 1:
            return self._multi_control_gate_to_matrix()
        control = self.controls[0]
        target = self.targets[0]

        nb_qubit = abs(control - target) + 1
        zero = np.array([[1, 0], [0, 0]])
        one = np.array([[0, 0], [0, 1]])
        I2 = np.eye(2)

        control_matrix = zero if control == 0 else I2
        target_matrix = (
            one
            if control == 0
            else (self.non_controlled_gate.to_matrix() if target == 0 else I2)
        )

        for i in range(1, nb_qubit):
            if i == control:
                target_matrix = np.kron(target_matrix, one)
            elif i == target:
                target_matrix = np.kron(
                    target_matrix, self.non_controlled_gate.to_matrix()
                )
            else:
                target_matrix = np.kron(target_matrix, I2)

        for i in range(1, nb_qubit):
            if i == control:
                control_matrix = np.kron(control_matrix, zero)
            else:
                control_matrix = np.kron(control_matrix, I2)

        return control_matrix + target_matrix

    def _multi_control_gate_to_matrix(self) -> Matrix:
        import numpy as np
        import math
        from mpqp.core.instruction.gates.native_gates import SWAP

        I2 = np.eye(2)

        min_qubit = min(min(self.controls), min(self.targets))
        max_qubit = max(max(self.controls), max(self.targets))
        nb_qubits = max_qubit - min_qubit + 1

        canonical_matrix = self.to_canonical_matrix()
        while canonical_matrix.shape[0] < 2**nb_qubits:
            canonical_matrix = np.kron(canonical_matrix, I2)

        matrix = np.eye(2**nb_qubits, dtype=np.complex64)
        qubit_types = {i: "None" for i in range(nb_qubits)}

        for i, _ in enumerate(self.controls):
            qubit_types[i] = "control"
        for i, _ in enumerate(self.targets, start=len(self.controls)):
            qubit_types[i] = "target"

        def swap_and_update(matrix: Matrix, idx_a: int, idx_b: int):
            swap_matrix = SWAP(idx_a, idx_b).to_matrix()
            extended_swap_matrix = np.eye(2**nb_qubits)
            start = min(idx_b, idx_a)
            if start == 0:
                extended_swap_matrix = swap_matrix
            if start != 0:
                extended_swap_matrix = I2
                for _ in range(1, start - 1):
                    extended_swap_matrix = np.kron(extended_swap_matrix, I2)
                extended_swap_matrix = np.kron(extended_swap_matrix, swap_matrix)
            for _ in range(int(math.log2(extended_swap_matrix.shape[0])), nb_qubits):
                extended_swap_matrix = np.kron(extended_swap_matrix, I2)
            return np.dot(matrix, extended_swap_matrix)

        for control in sorted(self.controls):
            control -= min_qubit
            if qubit_types[control] != "control":
                target_idx = next(
                    i
                    for i in range(nb_qubits)
                    if qubit_types[i] == "control" and i not in self.controls
                )
                print("control:", control, target_idx)
                matrix = swap_and_update(matrix, control, target_idx)
                qubit_types[control], qubit_types[target_idx] = (
                    qubit_types[target_idx],
                    qubit_types[control],
                )

        for target in sorted(self.targets):
            target -= min_qubit
            if qubit_types[target] != "target":
                target_idx = next(
                    i
                    for i in range(nb_qubits)
                    if qubit_types[i] == "target" and i not in self.controls
                )
                matrix = swap_and_update(matrix, target, target_idx)
                qubit_types[target], qubit_types[target_idx] = (
                    qubit_types[target_idx],
                    qubit_types[target],
                )

        return matrix.dot(canonical_matrix).dot(matrix)
