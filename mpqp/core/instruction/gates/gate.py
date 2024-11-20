from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from typing import Optional
from warnings import warn

import numpy as np
import numpy.typing as npt
from scipy.linalg import fractional_matrix_power
from typeguard import typechecked

from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.instruction.instruction import Instruction
from mpqp.tools.errors import NumberQubitsWarning
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import matrix_eq


@typechecked
class Gate(Instruction, ABC):
    """Represent a unitary operator acting on qubit(s).

    A gate is an measurement and the main component of a circuit. The semantics
    of a gate is defined using
    :class:`~mpqp.core.instruction.gates.gate_definition.GateDefinition`.

    Args:
        targets: List of indices referring to the qubits on which the gate will
            be applied.
        label: Label used to identify the gate.
    """

    def __init__(
        self,
        targets: list[int],
        label: Optional[str] = None,
    ):

        if len(targets) == 0:
            raise ValueError("Expected non-empty target list")
        super().__init__(targets, label=label)

    def to_matrix(self, desired_gate_size: int = 0) -> Matrix:
        """Return the matricial semantics to this gate. Considering connections'
        order and position, in contrast with :meth:`~Gate.to_canonical_matrix`.

        Args:
            desired_gate_size: The total number for qubits needed for the gate
                representation. If not provided, the minimum number of qubits
                required to generate the matrix will be used.

        Returns:
            A numpy array representing the unitary matrix of the gate.

        Example:
            >>> m = UnitaryMatrix(
            ...     np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
            ... )
            >>> pprint(CustomGate(m, [1, 2]).to_matrix())
            [[0, 0, 0, 1],
             [0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 1, 0]]
            >>> pprint(SWAP(0, 1).to_matrix())
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
            >>> pprint(TOF([1,3], 2).to_matrix())
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 0, 0]]

        """
        from .native_gates import SWAP

        first_connection = min(self.connections())
        last_connection = max(self.connections())
        connections_offset = 0

        if desired_gate_size == 0:
            desired_gate_size = last_connection - first_connection + 1
            connections_offset = first_connection

        if connections_offset + desired_gate_size < last_connection:
            raise ValueError(f"`desired_gate_size` must be at least {last_connection}")

        preceding_eyes = np.eye(2 ** (first_connection - connections_offset))
        following_eyes = np.eye(
            2
            ** (
                desired_gate_size
                - len(self.targets)
                - (first_connection - connections_offset)
            )
        )
        result = np.kron(
            preceding_eyes, np.kron(self.to_canonical_matrix(), following_eyes)
        )

        permutations = set(
            tuple(sorted((origin + first_connection, destination)))
            for (origin, destination) in enumerate(self.targets)
            if origin + first_connection != destination
        )

        swaps = [
            SWAP(o_index - connections_offset, d_index - connections_offset).to_matrix(
                desired_gate_size
            )
            for o_index, d_index in permutations
        ]

        return reduce(np.dot, swaps[::-1] + [result] + swaps)

    @abstractmethod
    def to_canonical_matrix(self) -> Matrix:
        """Return the "base" matricial semantics to this gate. Without
        considering potential column and row permutations needed if the targets
        of the gate are not sorted.

        Returns:
            A numpy array representing the unitary matrix of the gate.

        Example:
            >>> m = UnitaryMatrix(
            ...     np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
            ... )
            >>> pprint(CustomGate(m, [1, 2]).to_canonical_matrix())
            [[0, 0, 0, 1],
             [0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 1, 0]]
            >>> pprint(SWAP(0,1).to_canonical_matrix())
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]

        """
        pass

    def inverse(self) -> Gate:
        """Computing the inverse of this gate.

        Returns:
            The gate corresponding to the inverse of this gate.

        Example:
            >>> Z(0).inverse()
            Z(0)
            >>> gate = CustomGate(UnitaryMatrix(np.diag([1,1j])),[0])
            >>> pprint(gate.inverse().to_matrix())
            [[1, 0  ],
             [0, -1j]]

        """
        # TODO: test
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        return CustomGate(
            UnitaryMatrix(self.to_matrix().transpose().conjugate()),
            self.targets,
            (
                None
                if self.label is None
                else (self.label[:-1] if self.label.endswith("†") else self.label + "†")
            ),
        )

    def is_equivalent(self, other: Gate) -> bool:
        """Determine if the gate in parameter is equivalent to this gate.

        The equivalence of two gate is only determined from their matricial
        semantics (and thus ignores all other aspects of the gate such as the
        target qubits, the label, etc....)

        Args:
            other: the gate to test if it is equivalent to this gate

        Returns:
            ``True`` if the two gates' matrix semantics are equal.

        Example:
            >>> X(0).is_equivalent(CustomGate(UnitaryMatrix(np.array([[0,1],[1,0]])),[1]))
            True

        """
        # TODO: test
        return matrix_eq(self.to_matrix(), other.to_matrix())

    def power(self, exponent: float) -> Gate:
        """Compute the exponentiation `G^{exponent}` of this gate G.

        Args:
            exponent: Number representing the exponent.

        Returns:
            The gate elevated to the exponent in parameter.

        Examples:
            >>> swap_gate = SWAP(0,1)
            >>> pprint((swap_gate.power(2)).to_matrix())
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
            >>> pprint((swap_gate.power(-1)).to_matrix())
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
            >>> pprint((swap_gate.power(0.75)).to_matrix())
            [[1, 0               , 0               , 0],
             [0, 0.14645+0.35355j, 0.85355-0.35355j, 0],
             [0, 0.85355-0.35355j, 0.14645+0.35355j, 0],
             [0, 0               , 0               , 1]]

        """
        # TODO: test
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        if exponent == 1:
            return deepcopy(self)
        if exponent == -1:
            return self.inverse()

        semantics: npt.NDArray[np.complex64] = fractional_matrix_power(
            self.to_matrix(), exponent
        )

        return CustomGate(
            definition=UnitaryMatrix(semantics / np.linalg.norm(semantics, ord=2)),
            targets=self.targets,
            label=None if self.label is None else self.label + f"^{exponent}",
        )

    def tensor_product(self, other: Gate, targets: Optional[list[int]] = None) -> Gate:
        """Compute the tensor product of the current gate.

        This operation is shorthanded by the ``@`` operator.

        Args:
            other: Second operand of the tensor product.
            targets: If need be, the targets of the gates can be overridden
                using this value. Leave it empty to use the default automatic
                inference.

        Returns:
            A Gate representing a tensor product of this gate with the gate in
            parameter.

        Example:
            >>> (X(0).tensor_product(Z(0))).to_matrix()
            array([[ 0,  0,  1,  0],
                   [ 0,  0,  0, -1],
                   [ 1,  0,  0,  0],
                   [ 0, -1,  0,  0]])

        # 3M-TODO: to be implemented, don't trust the code bellow, it's pure experiments
        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        if targets is None:
            if len(set(self.targets).intersection(other.targets)) != 0:
                warn(
                    f"""
Targets have to change because the two gates overlap on qubits 
{set(self.targets).intersection(other.targets)}
If need be, please use the optional argument `targets` to remove all ambiguity. 
Naive attribution will be used (targets start at 0 and of the right length)""",
                    NumberQubitsWarning,
                )
                targets = list(range(self.nb_qubits + other.nb_qubits))
            else:
                targets = self.targets + other.targets

        gd = UnitaryMatrix(np.kron(self.to_matrix(), other.to_matrix()))

        l1 = "g1" if self.label is None else self.label
        l2 = "g2" if self.label is None else self.label

        return CustomGate(definition=gd, targets=targets, label=f"{l1}⊗{l2}")

    def _mandatory_label(self, postfix: str = ""):
        return "g" + postfix if self.label is None else self.label

    def __matmul__(self, other: Gate):
        return self.tensor_product(other)

    def product(self, other: Gate, targets: Optional[list[int]] = None) -> Gate:
        """Compute the composition of self and the other gate.

        This operation is shorthanded by the ``*`` operator.

        Args:
            other: Rhs of the product.
            targets: Qubits on which this new gate will operate. If not given,
                the targets of the two gates multiplied must be the same and the
                resulting gate will have this same targets.

        Returns:
            The product of the two gates concerned.

        Example:
            >>> pprint((X(0).product(Z(0))).to_matrix())
            [[0, -1],
             [1, 0 ]]

        """
        # TODO: test
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        return CustomGate(
            definition=UnitaryMatrix(self.to_matrix().dot(other.to_matrix())),
            targets=self._check_targets_compatibility(other, targets),
            label=f"{self._mandatory_label('1')}×{other._mandatory_label('2')}",
        )

    def __mul__(self, other: Gate):
        return self.product(other)

    def scalar_product(self, scalar: complex) -> Gate:
        """Multiply this gate by a scalar. It normalizes the result to ensure it
        is unitary.

        Args:
            scalar: The number to multiply the gate's matrix by.

        Returns:
            The result of the multiplication, the targets of the resulting gate
            will be the same as the ones of the initial gate.

        Example:
            >>> pprint((X(0).scalar_product(1j)).to_matrix())
            [[0 , 1j],
             [1j, 0 ]]

        """
        # 3M-TODO: to test
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        return CustomGate(
            UnitaryMatrix(self.to_matrix() * scalar / abs(scalar)),
            targets=self.targets,
            label=f"{scalar}×{self._mandatory_label()}",
        )

    def minus(self, other: Gate, targets: Optional[list[int]] = None) -> Gate:
        """Compute the subtraction of two gates. It normalizes the subtraction
        to ensure it is unitary.

        This operation is shorthanded by the ``-`` operator.

        Args:
            other: The gate to subtract to this gate.
            targets: Qubits on which this new gate will operate. If not given,
                the targets of the two gates multiplied must be the same and the
                resulting gate will have this same targets.

        Returns:
            The subtraction of ``self`` and ``other``.

        Example:
            >>> (X(0).minus(Z(0))).to_matrix()
            array([[-0.70710678,  0.70710678],
                   [ 0.70710678,  0.70710678]])

        """
        # TODO: test
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        subtraction = self.to_matrix() - other.to_matrix()
        return CustomGate(
            definition=UnitaryMatrix(subtraction / np.linalg.norm(subtraction, 2)),
            targets=self._check_targets_compatibility(other, targets),
            label=f"{self._mandatory_label('1')}-{other._mandatory_label('2')}",
        )

    def plus(self, other: Gate, targets: Optional[list[int]] = None) -> Gate:
        """Compute the sum of two gates. It normalizes the result to ensure it
        is unitary.

        This operation is shorthanded by the ``+`` operator.

        Args:
            other: The gate to add to this gate.
            targets: Qubits on which this new gate will operate. If not given,
                the targets of the two gates multiplied must be the same and the
                resulting gate will have this same targets.

        Returns:
            The sum of ``self`` and ``other``.

        Example:
            >>> (X(0).plus(Z(0))).to_matrix()
            array([[ 0.70710678,  0.70710678],
                   [ 0.70710678, -0.70710678]])

        """
        # 3M-TODO: to test
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        addition = self.to_matrix() + other.to_matrix()
        return CustomGate(
            definition=UnitaryMatrix(addition / np.linalg.norm(addition, 2)),
            targets=self._check_targets_compatibility(other, targets),
            label=f"{self._mandatory_label('1')}+{other._mandatory_label('2')}",
        )

    def _check_targets_compatibility(self, other: Gate, targets: Optional[list[int]]):
        if targets is None:
            if self.targets != other.targets:
                raise ValueError(
                    "Cannot infer what targets to use, please specify them in the "
                    "`targets` argument."
                )
            targets = self.targets
        if self.nb_qubits != other.nb_qubits:
            raise ValueError(
                f"Incompatible shapes for gates: respectively {self.nb_qubits} "
                f"and {other.nb_qubits} qubits"
            )
        if len(targets) != self.nb_qubits:
            raise ValueError(
                f"Incorrect size for targets: size {len(targets)} while it "
                f"should be {self.nb_qubits}"
            )
        return targets

    def __add__(self, other: Gate) -> Gate:
        return self.plus(other)

    def __sub__(self, other: Gate) -> Gate:
        return self.minus(other)


@typechecked
class InvolutionGate(Gate, ABC):
    """Gate who's inverse is itself.

    Args:
        targets: List of indices referring to the qubits on which the gate will be applied.
        label: Label used to identify the gate.
    """

    def inverse(self) -> Gate:
        return deepcopy(self)


@typechecked
class SingleQubitGate(Gate, ABC):
    """Abstract class for gates operating on a single qubit.

    Args:
        target: Index or referring to the qubit on which the gate will be applied.
        label: Label used to identify the gate.
    """

    def __init__(self, target: int, label: Optional[str] = None):
        Gate.__init__(self, [target], label)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.targets[0]})"

    nb_qubits = (  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        1
    )

    @classmethod
    def range(cls, start_or_end: int, end: Optional[int] = None, step: int = 1):
        """Apply the gate to a range of qubits.

        Args:
            start_or_end: If ``end`` is not defined, this value is treated as
                the end value of the range, and the range starts from ``0``.
                Otherwise, it is treated as the start value.
            end: The upper bound of the range (exclusive).
            step: The step or increment between indices in the range.

        Returns:
            A list of gate instances applied to the qubits in the specified
            range.

        Examples:
            >>> H.range(3)
            [H(0), H(1), H(2)]
            >>> S.range(1, 4)
            [S(1), S(2), S(3)]
            >>> Z.range(7, step=2)
            [Z(0), Z(2), Z(4), Z(6)]

        """
        if end is None:
            start_or_end, end = 0, start_or_end
        return [cls(index) for index in range(start_or_end, end, step)]
