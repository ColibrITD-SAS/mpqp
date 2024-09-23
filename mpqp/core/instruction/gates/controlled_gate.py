from __future__ import annotations

from abc import ABC
from functools import reduce
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

    def to_matrix(self, desired_gate_size: int = 0) -> Matrix:
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

    def __repr__(self) -> str:
        c = self.controls if len(self.controls) > 1 else self.controls[0]
        t = self.targets if len(self.targets) > 1 else self.targets[0]
        return f"{type(self).__name__}({c}, {t})"
