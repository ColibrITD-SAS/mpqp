from __future__ import annotations

from abc import abstractmethod, ABC
from warnings import warn

import numpy as np
from sympy import Expr
from typeguard import typechecked
from numbers import Complex

from mpqp.tools.maths import is_unitary, matrix_eq
from mpqp.tools.generics import Matrix, one_lined_repr


@typechecked
class GateDefinition(ABC):
    """A class used to handle the definition of a Gate.

    A quantum gate can be defined in several ways, and this class allows us to
    define it as we prefer. It also handles the translation from one definition
    to another.

    This said, for now only one way of defining the gates is supported, using
    their matricial semantics.

    Example:
        >>> gate_matrix = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
        >>> gate_definition = UnitaryMatrix(gate_matrix)
        >>> custom_gate = CustomGate(gate_definition)
    """

    """ TODO: put this back once we implement the other definitions
    This class permit to define a gate in 4 potential ways:
        1. the unitary matrix defining the gate
        2. a combination of several other gates
        3. a combination of Kraus operators
        4. the decomposition of the gate in the Pauli basis (only possible for LU gates)
    """

    @abstractmethod
    def to_matrix(self) -> Matrix:
        """Returns the matrix corresponding to this gate definition."""

    @abstractmethod
    def subs(
        self,
        values: dict[Expr | str, Complex],
        remove_symbolic: bool = False,
        disable_symbol_warn: bool = False,
    ) -> GateDefinition:
        pass

    # @abstractmethod
    def to_kraus_representation(self) -> KrausRepresentation:
        """6M-TODO"""
        raise NotImplementedError()

    # @abstractmethod
    def to_pauli_decomposition(self) -> PauliDecomposition:
        """6M-TODO"""
        raise NotImplementedError()

    def is_equivalent(self, other: GateDefinition) -> bool:
        """Determines if this definition is equivalent to the other.

        Args:
            other: The definition we want to know if it is equivalent.

        Example:
            >>> d1 = UnitaryMatrix(np.array([[1, 0], [0, -1]]))
            >>> d2 = UnitaryMatrix(np.array([[2, 0], [0, -2.0]]) / 2)
            >>> d1.is_equivalent(d2)
            True
        """
        return matrix_eq(self.to_matrix(), other.to_matrix())

    def inverse(self) -> GateDefinition:
        """Compute the inverse of the gate.

        Returns:
            A GateDefinition representing the inverse of the gate defined.

        Example:
            >>> UnitaryMatrix(np.array([[1, 0], [0, -1]])).inverse()
            array([[ 1.,  0.],
                   [-0., -1.]])
        """
        mat = self.to_matrix()

        if not all(
            isinstance(elt.item(), Complex) for elt in np.nditer(mat, ["refs_ok"])  # type: ignore
        ):
            raise ValueError("Cannot invert arbitrary gates using symbolic variables")
        return UnitaryMatrix(np.linalg.inv(mat))  # type:ignore


class KrausRepresentation(GateDefinition):
    """# 6M-TODO : implement and comment"""

    def __init__(self):
        self.pp = 1


class PauliDecomposition(GateDefinition):
    """# 6M-TODO : implement and comment"""

    def __init__(self):
        self.pp = 1


@typechecked
class UnitaryMatrix(GateDefinition):
    """Definition of a gate using it's matrix.

    Args:
        definition: Matrix defining the unitary gate.
        disable_symbol_warn: Boolean used to enable/disable warning concerning
            unitary checking with symbolic variables.
    """

    def __init__(self, definition: Matrix, disable_symbol_warn: bool = False):
        if any(isinstance(elt, Expr) for _, elt in np.ndenumerate(definition)):
            if not disable_symbol_warn:
                # 3M-TODO: can we improve this situation ?
                warn("Cannot ensure that a operator defined with variables is unitary.")
        elif not is_unitary(definition):
            raise ValueError(
                "Matrices defining gates have to be unitary. It is not the case"
                f" for\n{definition}"
            )
        self.matrix = definition
        """See parameter :attr:`definition`'s description."""

    def to_matrix(self) -> Matrix:
        return self.matrix

    def to_kraus_representation(self) -> KrausRepresentation:
        """6M-TODO to implement"""
        ...

    def to_pauli_decomposition(self) -> PauliDecomposition:
        """6M-TODO to implement"""
        ...

    def subs(
        self,
        values: dict[Expr | str, Complex],
        remove_symbolic: bool = False,
        disable_symbol_warn: bool = False,
    ):
        """Substitute some symbolic variables in the definition by complex values.

        Args:
            values: Mapping between the symbolic variables and their complex
                attributions.
            remove_symbolic: Some values such as pi are kept symbolic during
                circuit manipulation for better precision, but must be replaced
                by their complex counterpart for circuit execution, this
                arguments fills that role. Defaults to False.
            disable_symbol_warn: This method returns a :class:`UnitaryMatrix`,
                which raises a warning in case the matrix used to build it has
                symbolic variables. This is because this class performs
                verifications on the matrix to ensure it is indeed unitary, but
                those verifications cannot be done on symbolic variables. This
                argument disables this check because in some contexts, it is
                undesired. Defaults to False.
        """

        def mapping(val: Expr | Complex) -> Expr | Complex:
            def caster(v: Expr | Complex) -> Expr | Complex:
                # problem between pyright and abstract numeric types ?
                # "complex" cannot be assigned to return type "Complex"
                return (
                    complex(v) if remove_symbolic else v
                )  # pyright: ignore[reportReturnType]

            # the types in sympy are relatively badly handled
            # Argument of type "Unknown | Basic | Expr" cannot be assigned to parameter "v" of type "Expr | Complex"
            return (
                caster(val.subs(values))  # pyright: ignore[reportArgumentType]
                if isinstance(val, Expr)
                else val
            )

        matrix = self.to_matrix()
        otype = (
            complex
            if remove_symbolic
            or not any(isinstance(val, Expr) for _, val in np.ndenumerate(matrix))
            else object
        )

        return UnitaryMatrix(
            np.vectorize(mapping, otypes=[otype])(matrix), disable_symbol_warn
        )

    def __repr__(self) -> str:
        return f"UnitaryMatrix({one_lined_repr(getattr(self, 'matrix', ''))})"
