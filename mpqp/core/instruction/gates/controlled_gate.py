from __future__ import annotations

from abc import ABC
from typing import Optional

from typeguard import typechecked

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
        non_controlled_gate: Optional[Gate] = None,
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
