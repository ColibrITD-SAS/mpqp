from __future__ import annotations
from typing import Optional

from qiskit.circuit import Parameter

from mpqp.core.instruction import Instruction
from mpqp.core.languages import Language


class Breakpoint(Instruction):
    """A breakpoint is a special instruction that will display the state of the
    circuit at the desired state.

    Args:
        draw_circuit: If ``True`` in addition of displaying the current state
            vector, it will draw the circuit at this step.
        enabled: A breakpoint can be kept in the circuit but temporarily
            disabled using this argument.
        label: A breakpoint can be given a label for easier traceability.
    """

    def __init__(
        self,
        draw_circuit: bool = False,
        enabled: bool = True,
        label: str | None = None,
    ):
        self.draw_circuit = draw_circuit
        self.enabled = enabled
        super().__init__([0], label)

    def __repr__(self) -> str:
        return (
            f"Breakpoint(targets={self.targets}, draw_circuit={self.draw_circuit},"
            f" enabled={self.enabled}, label={self.label})"
        )

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ) -> str:
        raise NotImplementedError(f"Error: {language} is not supported")
