"""An :class:`Instruction` is the base element for circuits elements, containing
common methods to all of them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional
from numbers import Complex

from sympy import Expr
from qiskit.circuit import Parameter
from typeguard import typechecked

from mpqp.core.languages import Language
from mpqp.tools.generics import flatten


@typechecked
class Instruction(ABC):
    """Abstract class defining an instruction of a quantum circuit.

    An Instruction is the elementary component of a
    :class:`QCircuit<mpqp.core.circuit>`. It consists in a manipulation of one
    (or several) qubit(s) of the quantum circuit. It may involve classical bits
    as well, for defining or retrieving the result of the instruction.

    It can be of type:

        - :class:`Gate<mpqp.core.instruction.gates.gate.Gate>`
        - :class:`Measure<mpqp.core.instruction.measurement.measure.Measure>`
        - :class:`Barrier<mpqp.core.instruction.barrier.Barrier>`

    Args:
        targets: List of indices referring to the qubits on which the
            instruction will be applied
        label: label used to identify the instruction
    """

    def __init__(
        self,
        targets: list[int],
        label: Optional[str] = None,
    ):
        if len(set(targets)) != len(targets):
            raise ValueError(f"Duplicate registers in targets: {targets}")
        if not all([t >= 0 for t in targets]):
            raise ValueError(f"Negative index in targets: {targets}")
        self.targets = targets
        """See parameter description."""
        self.label = label
        """See parameter description."""

    @property
    def nb_qubits(self) -> int:
        """Number of qubits of this instruction"""
        return len(self.connections())

    @property
    def nb_cbits(self) -> int:
        """Number of cbits of this instruction"""
        from mpqp.core.instruction.measurement.basis_measure import BasisMeasure

        if isinstance(self, BasisMeasure):
            return len(flatten([self.c_targets]))
        else:
            return 0

    @abstractmethod
    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ) -> Any:
        """Transforms this instruction into the corresponding object in the
        language specified in the ``language`` arg.

        By default, the instruction is translated to the corresponding one in
        Qiskit, since it is the interface we use to generate the OpenQASM code.

        In the future, we will generate the OpenQASM code on our own, and this
        method will be used only for complex objects that are not tractable by
        OpenQASM (like hybrid structures).

        Args:
            language: enum representing the target language.
            qiskit_parameters: We need to keep track of the parameters
                passed to qiskit in order not to define twice the same
                parameter. Defaults to ``set()``.

        Returns:
            The corresponding instruction (gate or measure) in the target
            language
        """
        pass

    def __str__(self) -> str:
        from mpqp.core.circuit import QCircuit

        c = QCircuit(
            (self.targets if isinstance(self.targets, int) else max(self.targets)) + 1
        )
        c.add(self)
        return str(c)

    def __repr__(self) -> str:
        from mpqp.core.instruction.gates import ControlledGate

        controls = str(self.controls) + "," if isinstance(self, ControlledGate) else ""
        return f"{type(self).__name__}({controls}{self.targets})"

    def connections(self) -> set[int]:
        """Returns the indices of the qubits connected to the instruction.

        Example:
            >>> CNOT(0,1).connections()
            [0, 1]

        Returns:
            The qubits ordered connected to instruction.
        """
        from mpqp.core.instruction.gates import ControlledGate

        return (
            set(self.controls).union(self.targets)
            if isinstance(self, ControlledGate)
            else set(self.targets)
        )

    def subs(
        self, values: dict[Expr | str, Complex], remove_symbolic: bool = False
    ) -> Instruction:
        r"""Substitutes the parameters of the instruction with complex values.
        Optionally also removes all symbolic variables such as `\pi` (needed for
        example for circuit execution).

        Since we use ``sympy`` for gates' parameters, ``values`` can in fact be
        anything the ``subs`` method from ``sympy`` would accept.

        Example:
            >>> theta = symbols("θ")
            >>> print(Rx(theta, 0).subs({theta: np.pi}))
               ┌───────┐
            q: ┤ Rx(π) ├
               └───────┘

        Args:
            values: Mapping between the variables and the replacing values.
            remove_symbolic: If symbolic values should be replaced by their
                numeric counterpart.

        Returns:
            The circuit with the replaced parameters.
        """
        return deepcopy(self)
