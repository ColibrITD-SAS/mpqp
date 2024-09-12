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

    def to_matrix(self, nb_qubits: int = 0) -> Matrix:
        """
        Constructs the matrix representation of a controlled gate.

        Args:
            nb_qubits: The total number of qubits in the system. If not provided,
                        the minimum number of qubits required to generate the matrix
                        will be used.

        Returns:
            The matrix representation of the controlled gate.
        """
        import numpy as np

        if len(self.controls) != 1 or len(self.targets) != 1:
            return self._multi_control_gate_to_matrix(nb_qubits)

        control, target = self.controls[0], self.targets[0]

        if nb_qubits != 0:
            max_qubit = max(control, target) + 1
            if nb_qubits < max_qubit:
                raise ValueError(f"nb_qubits must be at least {max_qubit}")
        else:
            min_qubit = min(control, target)
            control -= min_qubit
            target -= min_qubit
            nb_qubits = abs(control - target) + 1

        zero = np.diag([1, 0])
        one = np.diag([0, 1])
        non_controlled_gate = self.non_controlled_gate.to_matrix()
        I2 = np.eye(2)

        control_matrix = zero if control == 0 else I2
        target_matrix = (
            one if control == 0 else (non_controlled_gate if target == 0 else I2)
        )

        for i in range(1, nb_qubits):
            if i == control:
                target_matrix = np.kron(target_matrix, one)
                control_matrix = np.kron(control_matrix, zero)
            elif i == target:
                target_matrix = np.kron(target_matrix, non_controlled_gate)
                control_matrix = np.kron(control_matrix, I2)
            else:
                target_matrix = np.kron(target_matrix, I2)
                control_matrix = np.kron(control_matrix, I2)

        return control_matrix + target_matrix  # pyright: ignore[reportReturnType]

    def _multi_control_gate_to_matrix(self, nb_qubits: int = 0) -> Matrix:
        import numpy as np
        from mpqp.core.instruction.gates.native_gates import SWAP

        controls, targets = self.controls, self.targets
        min_qubit, max_qubit = min(self.connections()), max(self.connections())

        # If nb_qubits is not provided, calculate the necessary number of minimal qubits
        if nb_qubits == 0:
            nb_qubits = max_qubit - min_qubit + 1
            controls = [x - min_qubit for x in controls]
            targets = [x - min_qubit for x in targets]
        elif nb_qubits < max_qubit + 1:
            raise ValueError(f"nb_qubits must be at least {max_qubit + 1}")

        # Get the canonical matrix and extend it to the correct size
        canonical_matrix = self.to_canonical_matrix()
        if canonical_matrix.shape[0] < 2**nb_qubits:
            canonical_matrix = np.kron(
                canonical_matrix, np.eye(2**nb_qubits // canonical_matrix.shape[0])
            )

        swaps = []
        # qubit_types if a representation of the all qubits to follow target and control with swap
        # assuming that canonical_matrix start with control and then target
        qubit_types = {i: "None" for i in range(nb_qubits)}
        qubit_types.update({i: "control" for i in range(len(controls))})
        qubit_types.update({i + len(controls): "target" for i in range(len(targets))})

        def swap_and_update(qubits: list[int], target_type: str):
            for qubit in sorted(qubits):
                if qubit_types[qubit] != target_type:
                    # Find a qubit of the target type that is not in the current set of qubits
                    target_idx = next(
                        i
                        for i in range(nb_qubits)
                        if qubit_types[i] == target_type and i not in qubits
                    )
                    swaps.append(SWAP(qubit, target_idx).to_matrix(nb_qubits))
                    qubit_types[qubit], qubit_types[target_idx] = (
                        qubit_types[target_idx],
                        qubit_types[qubit],
                    )

        swap_and_update(controls, "control")
        swap_and_update(targets, "target")

        return reduce(np.dot, swaps[::-1] + [canonical_matrix] + swaps)


# pseudo algo for multi control g to m
# input: gate, nb_qubits

# blank_qubits = nb_qubits - gate.nb_qubits
# result = gate.canonical_matrix() @ np.eye(2^blank_qubits)
# swaps = []
# for canonical_index, actual_index in enumerate(gate.controls + gate.targets):
#     swaps.append(SWAP(canonical_index, actual_index).to_matrix(nb_qubits))
# return reduce(np.dot, swaps + [result] + swaps[::-1])
