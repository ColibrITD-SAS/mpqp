"""An :class:`Instruction` is the base element of circuit elements, containing methods common to all of them."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from numbers import Complex
from pickle import dumps
from typing import TYPE_CHECKING, Any, Optional

from typeguard import typechecked

if TYPE_CHECKING:
    from sympy import Expr
    from qiskit.circuit import Parameter

from mpqp.core.languages import Language
from mpqp.tools.generics import SimpleClassReprABC, flatten


@typechecked
class Instruction(SimpleClassReprABC):
    """Abstract class defining an instruction of a quantum circuit.

    An Instruction is the elementary component of a
    :class:`~mpqp.core.circuit`. It consists of a manipulation of one
    (or several) qubit(s) of the quantum circuit. It may involve classical bits
    as well, for defining or retrieving the result of the instruction.

    It can be of type:

        - :class:`~mpqp.core.instruction.gates.gate.Gate`
        - :class:`~mpqp.core.instruction.measurement.measure.Measure`
        - :class:`~mpqp.core.instruction.barrier.Barrier`

    Args:
        targets: List of indices referring to the qubits to which the
            instruction will be applied.
        label: Label used to identify the instruction.
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
        self._dynamic = False

    @property
    def nb_qubits(self) -> int:
        """Number of qubits of this instruction."""
        return len(self.connections())

    @property
    def nb_cbits(self) -> int:
        """Number of cbits of this instruction."""
        from mpqp.core.instruction.measurement.basis_measure import BasisMeasure

        if isinstance(self, BasisMeasure):
            return len(flatten([self.c_targets]))
        else:
            return 0

    @abstractmethod
    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ) -> Any:
        """Transforms this instruction into the corresponding object in the
        language specified in the ``language`` arg.

        By default, the instruction is translated to the corresponding one in
        Qiskit, since it is the interface we use to generate the OpenQASM code.

        In the future, we will generate the OpenQASM code on our own, and this
        method will be used only for complex objects that are not tractable by
        OpenQASM (like hybrid structures).

        Args:
            language: Enum representing the target language.
            qiskit_parameters: We need to keep track of the parameters
                passed to qiskit in order not to define twice the same
                parameter. Defaults to ``set()``.

        Returns:
            The corresponding instruction (gate or measure) in the target
            language.
        """
        pass

    def __eq__(self, value: object) -> bool:
        return dumps(self) == dumps(value)

    def __str__(self) -> str:
        from mpqp.core.circuit import QCircuit

        connection = self.connections()
        circuit_size = max(connection) + 1 if connection else 1
        circuit = QCircuit(circuit_size)
        circuit.add(self)
        return str(circuit)

    def __repr__(self) -> str:
        from mpqp.core.instruction.gates import ControlledGate

        controls = str(self.controls) + "," if isinstance(self, ControlledGate) else ""
        return f"{type(self).__name__}({controls}{self.targets})"

    def connections(self) -> set[int]:
        """Returns the indices of the qubits connected to the instruction.

        Returns:
            The ordered qubits connected to instruction.

        Example:
            >>> CNOT(0,1).connections()
            {0, 1}

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
        Optionally, also removes all symbolic variables such as `\pi` (needed for
        circuit execution, for example).

        Since we use ``sympy`` for gate parameters, ``values`` can in fact be
        anything the ``subs`` method from ``sympy`` would accept.

        Args:
            values: Mapping between the variables and the replacing values.
            remove_symbolic: Whether symbolic values should be replaced by their
                numeric counterparts.

        Returns:
            The circuit with the replaced parameters.

        Example:
            >>> theta = symbols("θ")
            >>> print(Rx(theta, 0).subs({theta: np.pi}))
               ┌───────┐
            q: ┤ Rx(π) ├
               └───────┘

        """
        return deepcopy(self)
