from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Complex
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from sympy import Expr

import numpy as np
from typeguard import typechecked

from mpqp.tools.display import one_lined_repr
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import is_power_of_two, is_unitary, matrix_eq


@typechecked
class GateDefinition(ABC):
    """Abstract class used to handle the definition of a Gate.

    A quantum gate can be defined in several ways, and this class allows us to
    define it as we prefer. It also handles the translation from one definition
    to another.

    This said, for now only one way of defining the gates is supported, using
    their matricial semantics.

    Example:
        >>> gate_matrix = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
        >>> custom_gate = CustomGate(gate_matrix, [0,1])

    """

    # TODO: put this back once we implement the other definitions. Are those
    # definitions really useful in practice ?
    # This class permit to define a gate in 4 potential ways:
    #     1. the unitary matrix defining the gate
    #     2. a combination of several other gates
    #     3. a combination of Kraus operators
    #     4. the decomposition of the gate in the Pauli basis (only possible for LU gates)

    @abstractmethod
    def to_matrix(self) -> Matrix:
        """Returns the matrix corresponding to this gate definition. Considering
        connections' order and position, in contrast with
        :meth:`~Gate.to_canonical_matrix`.
        """

    @abstractmethod
    def to_canonical_matrix(self) -> Matrix:
        """Returns the matrix corresponding to this gate definition."""

    @abstractmethod
    def subs(
        self,
        values: dict[Expr | str, Complex],
        remove_symbolic: bool = False,
        disable_symbol_warn: bool = False,
    ) -> GateDefinition:
        pass

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
            UnitaryMatrix(array([[ 1., 0.], [-0., -1.]]))

        """
        mat = self.to_matrix()

        if not all(isinstance(elt, Complex) for elt in mat.flatten()):
            raise ValueError("Cannot invert arbitrary gates using symbolic variables")
        return UnitaryMatrix(
            np.linalg.inv(mat)  # pyright: ignore[reportCallIssue, reportArgumentType]
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        self_dict = self.to_dict()
        value_dict = value.to_dict()

        if self_dict.keys() != value_dict.keys():
            return False

        for key in self_dict:
            val1, val2 = self_dict[key], value_dict[key]

            if isinstance(val1, list) and isinstance(val2, list):
                try:
                    from sympy import N

                    # TODO: replace by matrix_eq
                    val1_numeric = np.array(
                        [[N(el) for el in row] for row in val1], dtype=complex
                    )
                    val2_numeric = np.array(
                        [[N(el) for el in row] for row in val2], dtype=complex
                    )
                    if not np.allclose(val1_numeric, val2_numeric, atol=1e-7):
                        return False
                except:
                    if val1 != val2:
                        return False
            else:
                if val1 != val2:
                    return False
        return True

    def to_dict(self):
        """
        Serialize the gate to a dictionary.

        Returns:
            dict: A dictionary representation of the circuit.
        """
        result = {}
        for attr_name in dir(self):
            if (
                attr_name not in {'_abc_impl', '_nb_qubits'}
                and not attr_name.startswith("__")
                and not callable(getattr(self, attr_name))
            ):
                value = getattr(self, attr_name)
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                result[attr_name] = value
        return result


@typechecked
class UnitaryMatrix(GateDefinition):
    """Definition of a gate using its matrix.

    Args:
        definition: Matrix defining the unitary gate.
        disable_symbol_warn: Boolean used to enable/disable warning concerning
            unitary checking with symbolic variables.

    Raises:
        ValueError: Matrices defining gates have to be unitary.
        ValueError: The unitary matrix of a gate acting on qubits must have
            dimensions that are power of two.
    """

    def __init__(self, definition: Matrix, disable_symbol_warn: bool = False):

        numeric = True
        for _, elt in np.ndenumerate(definition):
            # 3M-TODO: can we improve this situation ?
            try:
                complex(elt)
            except TypeError:
                if not disable_symbol_warn:
                    warn(
                        "Cannot ensure that a operator defined with symbolic "
                        "variables is unitary."
                    )
                numeric = False
                break
        if numeric and not is_unitary(definition):
            raise ValueError(
                "Matrices defining gates have to be unitary. It is not the case"
                f" for\n{definition}"
            )
        if not is_power_of_two(definition.shape[0]):
            raise ValueError(
                "The unitary matrix of a gate acting on qubits must have "
                f"dimensions that are power of two, but got {definition.shape[0]}."
            )
        self.matrix = definition
        """See parameter :attr:`definition`'s description."""
        self._nb_qubits = None

    def to_matrix(self) -> Matrix:
        return self.matrix

    def to_canonical_matrix(self):
        return self.matrix

    @property
    def nb_qubits(self) -> int:
        if self._nb_qubits is None:
            self._nb_qubits = int(np.log2(self.matrix.shape[0]))
        return self._nb_qubits

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
        from sympy import Expr

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
        return f"UnitaryMatrix({one_lined_repr(self.matrix)})"
