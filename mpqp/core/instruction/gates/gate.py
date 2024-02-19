from __future__ import annotations
from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.linalg import fractional_matrix_power
from typeguard import typechecked

from mpqp.core.instruction.instruction import Instruction
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import matrix_eq


@typechecked
class Gate(Instruction, ABC):
    """Represent a unitary operator acting on qubit(s).

    A gate is an measurement and the main component of a circuit. The semantics
    of a gate is defined using
    :class:`GateDefinition<mpqp.core.instruction.gates.gate_definition.GateDefinition>`.

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
        super().__init__(targets, label=label)

    @abstractmethod
    def to_matrix(self) -> Matrix:
        """Return the "base" matricial semantics to this gate. Without
        considering potential column and row permutations needed if the targets
        of the gate are not sorted.

        Example:
            >>> gd = UnitaryMatrix(
            ...     np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
            ... )
            >>> CustomGate(gd, [1, 2]).to_matrix()
            array([[0, 0, 0, 1],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 0]])
            >>> SWAP(0,1).to_matrix()
            array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])

        Returns:
            A numpy array representing the unitary matrix of the gate.
        """

    def inverse(self) -> Gate:
        """Computing the inverse of this gate.

        Example:
            >>> Z(0).inverse()
            Z(0)
            >>> CustomGate(UnitaryMatrix(np.diag([1,1j])),[0]).inverse().to_matrix()
            array([[1,  0 ],
                   [0, -1j]])

        Returns:
            The gate corresponding to the inverse of this gate.
        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        return CustomGate(
            UnitaryMatrix(self.to_matrix().transpose().conjugate()),
            self.targets,
            self.label,
        )

    def is_equivalent(self, other: Gate) -> bool:
        """Determine if the gate in parameter is equivalent to this gate.

        The equivalence of two gate is only determined from their matricial
        semantics (and thus ignores all other aspects of the gate such as the
        target qubits, the label, etc....)

        Example:
            >>> X(0).is_equivalent(CustomGate(UnitaryMatrix(np.array([[0,1],[1,0]])),[1]))
            True

        Args:
            other: the gate to test if it is equivalent to this gate

        Returns:
            ``True`` if the two gates' matrix semantics are equal.
        """
        return matrix_eq(self.to_matrix(), other.to_matrix())

    def power(self, exponent: float) -> Gate:
        """Compute the exponentiation `G^{exponent}` of this gate G.

        Examples:
            >>> swap_gate = SWAP(0,1)
            >>> (swap_gate.power(2)).to_matrix()
            array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
            >>> (swap_gate.power(-1)).to_matrix()
            array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])
            >>> (swap_gate.power(0.75)).to_matrix() # not implemented yet
            array([[1.        +0.j        , 0.        +0.j        ,
                    0.        +0.j        , 0.        +0.j        ],
                   [0.        +0.j        , 0.14644661+0.35355339j,
                    0.85355339-0.35355339j, 0.        +0.j        ],
                   [0.        +0.j        , 0.85355339-0.35355339j,
                    0.14644661+0.35355339j, 0.        +0.j        ],
                   [0.        +0.j        , 0.        +0.j        ,
                    0.        +0.j        , 1.        +0.j        ]])

        Args:
            exponent: Number representing the exponent.

        Returns:
            The gate elevated to the exponent in parameter.
        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        return CustomGate(
            definition=UnitaryMatrix(
                fractional_matrix_power(self.to_matrix(), exponent)
            ),
            targets=self.targets,
            label=None if self.label is None else self.label + f"^{exponent}",
        )

    def tensor_product(self, other: Gate) -> Gate:
        """Compute the tensor product of the current gate.

        Example:
            >>> (X(0).tensor_product(Z(0))).to_matrix()
            array([[ 0,  0,  1,  0],
                   [ 0,  0,  0, -1],
                   [ 1,  0,  0,  0],
                   [ 0, -1,  0,  0]])

        Args:
            other: Second operand of the tensor product.

        Returns:
            A Gate representing a tensor product of this gate with the gate in
            parameter.

        # 3M-TODO: to be implemented, don't trust the code bellow, it's pure experiments
        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        gd = UnitaryMatrix(
            np.kron(self.to_matrix(), other.to_matrix())
        )  # self, gate, type="tensor"

        # compute the list of qubits that will be targeted by these gates
        ...

        # instantiate the definition

        l1 = "g1" if self.label is None else self.label
        l2 = "g2" if self.label is None else self.label

        return CustomGate(
            definition=gd,
            targets=[0],
            label=f"{l1}⊗{l2}",
        )

    def _mandatory_label(self, postfix: str = ""):
        return "g" + postfix if self.label is None else self.label

    def product(self, other: Gate, targets: Optional[list[int]] = None) -> Gate:
        """Compute the composition of self and the other gate.

        Example:
            >>> (X(0).product(Z(0))).to_matrix()
            array([[ 0, -1],
                   [ 1,  0]])

        Args:
            other: Rhs of the product.
            targets: Qubits on which this new gate will operate. If not given,
                the targets of the two gates multiplied must be the same and the
                resulting gate will have this same targets.

        Returns:
            The product of the two gates concerned.
        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        return CustomGate(
            definition=UnitaryMatrix(self.to_matrix().dot(other.to_matrix())),  # type: ignore
            targets=self._check_targets_compatibility(other, targets),
            label=f"{self._mandatory_label('1')}×{other._mandatory_label('2')}",
        )

    def scalar_product(self, scalar: complex) -> Gate:
        """Multiply this gate by a scalar. It normalizes the subtraction
        to ensure it is unitary.

        Example:
            >>> (X(0).scalar_product(1j)).to_matrix()
            array([[0, 1j],
                   [1j, 0]])

        Args:
            scalar: The number to multiply the gate's matrix by.

        Returns:
            The result of the multiplication, the targets of the resulting gate
            will be the same as the ones of the initial gate.
        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        return CustomGate(
            UnitaryMatrix(self.to_matrix() * scalar / abs(scalar)),
            targets=self.targets,
            label=f"{scalar}×{self._mandatory_label()}",
        )

    def minus(self, other: Gate, targets: Optional[list[int]] = None) -> Gate:
        """Compute the subtraction of two gates. It normalizes the subtraction
        to ensure it is unitary.

        Example:
            >>> (X(0).minus(Z(0))).to_matrix()
            array([[-0.70710678,  0.70710678],
                   [ 0.70710678,  0.70710678]])

        Args:
            other: The gate to subtract to this gate.
            targets: Qubits on which this new gate will operate. If not given,
                the targets of the two gates multiplied must be the same and the
                resulting gate will have this same targets.

        Returns:
            The subtraction of ``self`` and ``other``.

        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        subtraction = self.to_matrix() - other.to_matrix()
        return CustomGate(
            definition=UnitaryMatrix(subtraction / np.linalg.norm(subtraction, 2)),  # type: ignore
            targets=self._check_targets_compatibility(other, targets),
            label=f"{self._mandatory_label('1')}-{other._mandatory_label('2')}",
        )

    def plus(self, other: Gate, targets: Optional[list[int]] = None) -> Gate:
        """Compute the sum of two gates. It normalizes the subtraction
        to ensure it is unitary.

        Example:
            >>> (X(0).plus(Z(0))).to_matrix()
            array([[ 0.70710678,  0.70710678],
                   [ 0.70710678, -0.70710678]])

        Args:
            other: The gate to add to this gate.
            targets: Qubits on which this new gate will operate. If not given, the targets of the two gates multiplied
            must be the same and the resulting gate will have this same targets.

        Returns:
            The sum of ``self`` and ``other``.
        """
        from mpqp.core.instruction.gates.custom_gate import CustomGate

        addition = self.to_matrix() + other.to_matrix()
        return CustomGate(
            definition=UnitaryMatrix(addition / np.linalg.norm(addition, 2)),  # type: ignore
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
    """Gates operating on a single qubit.

    Args:
        target: Index or referring to the qubit on which the gate will be applied.
        label: Label used to identify the gate.
    """

    def __init__(self, target: int, label: Optional[str] = None):
        Gate.__init__(self, [target], label)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.targets[0]})"
