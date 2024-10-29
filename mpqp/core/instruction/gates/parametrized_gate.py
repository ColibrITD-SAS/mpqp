"""Some gate (such as CNOT for instance) do not need any parameters, but in
order to have a universal set of gates, one needs at least one parametrized
gate. This module defines the abstract class needed to define these gates as
well as a way to handle symbolic variables.

More on the topic of symbolic variable can be found in the `VQA <VQA.html>`_
page."""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from numbers import Complex
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sympy import Expr

from typeguard import typechecked

from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.instruction.gates.gate_definition import GateDefinition

# 3M-TODO: there might be a conception problem: to initialize a gate we need a gate
#  definition, the easiest for a definition is to compute the matrix, the
#  computation of the matrix requires self.parameters which only exist after the
#  initialization of ParametrizedGate. For now this problem is manually tackled
#  by instantiating self.parameters before the computation of the gates's
#  definition but this solution looks like a conception problem...


@typechecked
class ParametrizedGate(Gate, ABC):
    """Abstract class to factorize behavior of parametrized gate.

    Args:
        definition: Provide a definition of the gate (matrix, gate combination,
            ...).
        targets: List of indices referring to the qubits on which the gate will
            be applied.
        parameters: List of parameters used to define the gate.
        label: Label used to identify the measurement.

    """

    def __init__(
        self,
        definition: GateDefinition,
        targets: list[int],
        parameters: list[Expr | float],
        label: Optional[str] = None,
    ):
        Gate.__init__(self, targets, label)
        self.definition = definition
        """See parameter description."""
        self.parameters = parameters
        """See parameter description."""

    def subs(
        self, values: dict[Expr | str, Complex], remove_symbolic: bool = False
    ) -> ParametrizedGate:
        from sympy import Expr

        concrete_gate = deepcopy(self)
        options = getattr(self, "native_gate_options", {})
        concrete_gate.definition = concrete_gate.definition.subs(
            values, remove_symbolic, **options
        )
        caster = lambda v: float(v) if remove_symbolic else v
        concrete_gate.parameters = [
            caster(param.subs(values)) if isinstance(param, Expr) else param
            for param in self.parameters
        ]

        return concrete_gate

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.parameters},{self.targets})"
