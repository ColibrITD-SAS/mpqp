"""In some cases, you might need to debug the circuit you just created. We
simplify this by adding a breakpoint. As in other languages, a breakpoint is a
special instruction that stops the execution of the program and allows you to
dig into the state of the program. In order to enable this, a run with
breakpoints is in fact once per breakpoint, and for each run the circuit is
truncated up to the breakpoint."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from mpqp.core.instruction import Instruction
from mpqp.core.languages import Language

if TYPE_CHECKING:
    from qiskit.circuit import Parameter


class Breakpoint(Instruction):
    """A breakpoint is a special instruction that will display the state of the
    circuit at the desired state.

    Args:
        draw_circuit: If ``True`` in addition of displaying the current state
            vector, it will draw the circuit at this step.
        enabled: A breakpoint can be kept in the circuit but temporarily
            disabled using this argument.
        label: A breakpoint can be given a label for easier traceability.

    Example:
        >>> r = run(
        ...     QCircuit([H(0), Breakpoint(), CNOT(0,1), Breakpoint(draw_circuit=True, label="final")]),
        ...     IBMDevice.AER_SIMULATOR,
        ... )  # doctest: +NORMALIZE_WHITESPACE
        DEBUG: After instruction 1, state is
               0.707|00⟩ + 0.707|10⟩
        DEBUG: After instruction 2, at breakpoint `final`, state is
               0.707|00⟩ + 0.707|11⟩
               and circuit is
                    ┌───┐
               q_0: ┤ H ├──■──
                    └───┘┌─┴─┐
               q_1: ─────┤ X ├
                         └───┘

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
        args = []
        if self.draw_circuit is not False:
            args.append(f"draw_circuit={self.draw_circuit}")
        if self.enabled is not True:
            args.append(f"enabled={self.enabled}")
        if self.label is not None:
            args.append(f"label='{self.label}'")
        return f"Breakpoint({', '.join(args)})"

    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ) -> str:
        raise NotImplementedError(f"Error: {language} is not supported")

    def __eq__(self, value: object) -> bool:
        return isinstance(value, type(self)) and self.to_dict() == value.to_dict()

    def to_dict(self):
        """
        Serialize the Breakpoint to a dictionary.

        Returns:
            dict: A dictionary representation of the Breakpoint.
        """
        return {
            attr_name: getattr(self, attr_name)
            for attr_name in dir(self)
            if (
                not attr_name.startswith("_abc_")
                and not attr_name.startswith("__")
                and not callable(getattr(self, attr_name))
            )
        }
