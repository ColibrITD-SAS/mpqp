from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from mpqp.measures import Observable


class Qubo:
    """Class defining a QUBO representation, used to represent decision problems.

    A QUBO is defined by a quadratic expression of boolean variables, hence, the
    available operators are: ``+``, ``-``, ``*``, ``&``, ``|`` and ``^``.

    The QUBO expression can be solved with the qaoa_solver function.

    Args:
        value: string holding the name of the monomial
        left: left child of the node
        right: right child of the node

    Examples:
        >>> x0 = QuboAtom('x0')
        >>> x1 = QuboAtom('x1')
        >>> expr = 3*x0*x1 + x0 - 2*x1
        >>> pprint(expr.to_cost_hamiltonian().matrix)
        [[0, 0 , 0, 0],
         [0, -2, 0, 0],
         [0, 0 , 1, 0],
         [0, 0 , 0, 2]]
    """

    def __init__(
        self, value: str, left: Optional["Qubo"] = None, right: Optional["Qubo"] = None
    ):
        self.left = left
        self.right = right
        self.value = value

    def __neg__(self) -> "Qubo":
        return UnaryOperation('-', self)

    def __add__(self, other: Union["Qubo", int, float]) -> "Qubo":
        if isinstance(other, (float, int)):
            other = QuboConstant(str(other))
        current = BinaryOperation("+", self, other)
        return current

    def __radd__(self, other: Union["Qubo", int, float]) -> "Qubo":
        return self + other

    def __sub__(self, other: Union["Qubo", int, float]) -> "Qubo":
        if isinstance(other, (float, int)):
            other = QuboConstant(str(other))
        current = BinaryOperation('-', self, other)
        return current

    def __rsub__(self, other: Union["Qubo", int, float]) -> "Qubo":
        return self - other

    def __mul__(self, other: Union["Qubo", int, float]) -> "Qubo":
        if isinstance(other, (float, int)):
            other = QuboConstant(str(other))
        if not isinstance(other, QuboConstant):
            degree = self._check_degree()
            degree += other._check_degree()
            if degree > 2:
                raise ValueError(
                    f"The degree of the Qubo shouldn't be more than 2 not {degree}"
                )
        current = BinaryOperation("*", self, other)
        return current

    def __rmul__(self, other: Union["Qubo", int, float]) -> "Qubo":
        return self * other

    def __and__(self, other: "Qubo") -> "Qubo":
        return self * other

    def __or__(self, other: "Qubo") -> "Qubo":
        return self + other - self * other

    def __xor__(self, other: "Qubo") -> "Qubo":
        return self + other - 2 * (self * other)

    def __eq__(self, other: object) -> bool:
        if other == None:
            return False
        return False

    def __len__(self) -> int:
        """Returns the number of unique boolean variables in the QUBO expression."""
        return len(self.get_variables())

    def __str__(self) -> str:
        return self._print()

    def __collapse_coeffs(
        self, left: list[tuple[int, list[str]]], right: list[tuple[int, list[str]]]
    ):
        if len(right) == 1:
            for i in range(len(left)):
                hold: int = left[i][0]
                hold *= right[0][0]
                left[i][1].extend(right[0][1])
                left[i] = (hold, left[i][1])
            return left
        else:
            for i in range(len(right)):
                hold: int = right[i][0]
                hold *= left[0][0]
                right[i][1].extend(left[0][1])
                right[i] = (hold, right[i][1])
            return right

    def _check_degree(self):
        if isinstance(self, QuboAtom):
            return 1
        degree = 0

        if self.value == "+":
            assert self.left != None and self.right != None

            degree += max(self.left._check_degree(), self.right._check_degree())
        else:
            if self.left != None:
                degree += self.left._check_degree()
            if self.right != None:
                degree += self.right._check_degree()
        return degree

    def get_coeffs(self) -> list[tuple[int, list[str]]]:
        """Creates a list of lists containing the coefficients of the monomials
        of the QUBO.

        Returns:
            The list of coefficients and variables in the QUBO expression

        Examples:
            >>> x0 = QuboAtom('x0')
            >>> x1 = QuboAtom('x1')

            >>> expr = 3*x0 - x1 + 1
            >>> expr.get_coeffs()
            [(3, ['x0']), (-1, ['x1']), (1, [])]

            >>> expr = 3*(x0 | x1)
            >>> expr.get_coeffs()
            [(3, ['x0']), (3, ['x1']), (-3, ['x0', 'x1'])]
        """
        coeffs = []

        if self.left == None and self.right == None:
            return (
                [(int(self.value), [])]
                if isinstance(self, QuboConstant)
                else [(1, [self.value])]
            )

        left = []
        right = []
        hold = self.left
        if self.left != None:
            left = self.left.get_coeffs()
        if self.right != None:
            right = self.right.get_coeffs()

        if self.value == "+":
            coeffs.extend(left)
            coeffs.extend(right)
        elif self.value == "-":
            coeffs.extend(left)
            for i in range(len(right)):
                hold = right[i][0]
                hold *= -1
                right[i] = (hold, right[i][1])
            coeffs.extend(right)
        elif self.value == "*":
            return self.__collapse_coeffs(left, right)

        return coeffs

    def get_variables(self) -> list[str]:
        """Returns a list of all of the unique boolean variables of the QUBO.
        They are ordered from the left of the expression to the right.

        Examples:
            >>> x0 = QuboAtom('x0')
            >>> x1 = QuboAtom('x1')

            >>> expr = 3*x0 - x1
            >>> expr.get_variables()
            ['x0', 'x1']

            >>> expr = 3*x1 - 10*x0
            >>> expr.get_variables()
            ['x1', 'x0']

            >>> expr = 3*x0*x1 - x1 + x0
            >>> expr.get_variables()
            ['x0', 'x1']
        """
        coeffs = self.get_coeffs()
        known_vars = []
        for coeff in coeffs:
            for variable in coeff[1]:
                if not variable in known_vars:
                    known_vars.append(variable)
        return known_vars

    def matrix(self) -> tuple[npt.NDArray[np.float64], int]:
        """Creates the weight matrix of this QUBO expression.

        Returns:
            A tuple composed of the weight matrix and a potential additive constant.

        Examples:
            >>> x0 = QuboAtom('x0')
            >>> x1 = QuboAtom('x1')
            >>> x2 = QuboAtom('x2')
            >>> x3 = QuboAtom('x3')
            >>> expr = 2 * x0 + 3 * x1 + 4 * x0 * x2 + x3
            >>> pprint(expr.matrix()[0])
            [[2, 0, 2, 0],
             [0, 3, 0, 0],
             [2, 0, 0, 0],
             [0, 0, 0, 1]]
        """
        coeffs = self.get_coeffs()
        vars = self.get_variables()
        size = len(vars)
        matrix = np.zeros(shape=(size, size))
        constant = 0

        for coeff in coeffs:
            if len(coeff[1]) == 0:
                constant += coeff[0]
            elif len(coeff[1]) == 1:
                coord = 0
                for j in range(size):
                    if coeff[1][0] == vars[j]:
                        coord = j
                matrix[coord][coord] += coeff[0]
            else:
                abs = 0
                ord = 0
                for j in range(size):
                    if coeff[1][0] == vars[j]:
                        abs = j
                    if coeff[1][1] == vars[j]:
                        ord = j

                matrix[abs][ord] += coeff[0] / 2
                matrix[ord][abs] += coeff[0] / 2

        return matrix, constant

    def to_cost_hamiltonian(self) -> Observable:
        """Converts the QUBO matrix into a cost Hamiltonian.

        Returns:
            Observable: The cost Hamiltonian.

        Examples:
            >>> x_0 = QuboAtom("x_0")
            >>> x_1 = QuboAtom("x_1")
            >>> expr = 3 * x_0 * x_1 - 4 * x_0 - 2 * x_1 + 1
            >>> pprint(expr.to_cost_hamiltonian().matrix)
            [[1, 0 , 0 , 0 ],
             [0, -1, 0 , 0 ],
             [0, 0 , -3, 0 ],
             [0, 0 , 0 , -2]]
        """
        matrix, constant = self.matrix()
        size = matrix.shape[0]

        resulting_cost = np.zeros(shape=(2**size,))

        hx_ns = np.array([self._h_xi(size, i) for i in range(size)])

        for i in range(size):
            for j in range(i):
                resulting_cost += matrix[i][j] * hx_ns[i] * hx_ns[j] * 2
            resulting_cost += matrix[i][i] * hx_ns[i]
        return Observable(
            np.diag(resulting_cost).astype(np.complex64) + np.eye(2**size) * constant
        )

    def _h_xi(self, size: int, i: int):
        r"""Calculates the cost Hamiltonian `H(x_i)` for a given i-th binary
        parameter.

        `H` is defined as:
        $$ H(x_i) = \frac{I^{\otimes n} - Z_i}{2} $$
        $$ \text{with } ~~ Z_i = \underbrace{I \otimes \cdots \otimes I}_{i} \otimes Z \otimes \underbrace{I \otimes \cdots \otimes I}_{n-i-1} $$
        """
        Z = np.array([1, -1])
        Z_i = Z

        if i != 0:
            Z_i = np.kron(np.ones(2**i), Z_i)

        if size - i - 1 != 0:
            Z_i = np.kron(Z_i, np.ones(2 ** (size - i - 1)))

        result = (np.ones(2**size) - Z_i) / 2
        return result

    def _print(self, level: int = 1):
        left = ''
        right = ''

        if self.left != None:
            left = self.left._print(level + 1)
        if self.right != None:

            right = self.right._print(level + 1)
        if self.value == "*":
            return right + self.value + left
        return left + self.value + right


class QuboAtom(Qubo):
    """Class defining a boolean variable for a QUBO problem.

    See class Qubo for full usage of this class.

    Arg:
        value: String holding the name of the variable.

    Example:
        >>> x = QuboAtom("x")
        >>> expr = 2 * x + 2
        >>> print(expr.get_variables())
        ['x']
    """

    def __init__(self, value: str):
        super().__init__(value, None, None)


class BinaryOperation(Qubo):
    """Class defining binary operations in a Qubo expression."""

    def __init__(self, value: str, left: Qubo, right: Qubo):
        super().__init__(value, left, right)


class UnaryOperation(Qubo):
    """Class defining a unary operation for a Qubo expression."""

    def __init__(self, value: str, right: Qubo):
        super().__init__(value, None, right)


class QuboConstant(Qubo):
    """Class defining constant terms in a Qubo expression."""

    def __init__(self, value: str):
        super().__init__(value)
