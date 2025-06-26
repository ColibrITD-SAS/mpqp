from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from mpqp.measures import Observable

from typing import TYPE_CHECKING

from mpqp.tools.generics import Matrix


class Qubo:
    """Class defining a QUBO representation, used to represent decision problems.
    This class is instantiated through the use of QuboAtoms, not directly (see examples below).

    A QUBO is defined by a quadratic expression of boolean variables, hence, the
    available operators are: ``+``, ``-``, ``*``, ``&``, ``|`` and ``^``.

    The QUBO expression can be solved with the :func:`~mpqp.execution.vqa.qaoa.qaoa_solver` function.

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

    Note:
        This class is a binary tree representing the whole operation of the Qubo with QuboAtoms or QuboConstants as its leafs.

        Meanwhile the classes BinaryOperation and UnaryOperation are used as nodes representing the Qubo's operators.
    """

    def __init__(
        self, value: str, left: Optional["Qubo"] = None, right: Optional["Qubo"] = None
    ):

        if isinstance(left, QuboConstant) and isinstance(right, QuboConstant):
            raise ValueError("Qubo is not meant to model constant functions")

        self.left = left
        self.right = right
        self.value = value

        self._inverted_variables = []

    def __neg__(self) -> "Qubo":
        if isinstance(self, UnaryOperation):
            if TYPE_CHECKING:
                assert self.right
            return self.right
        return UnaryOperation('-', self)

    def __invert__(self) -> "Qubo":
        if isinstance(self, QuboAtom):
            from copy import deepcopy

            copy = deepcopy(self)
            copy.value = '~' + self.value
            return copy
        return -self

    def __add__(self, other: Union["Qubo", int, float]) -> "Qubo":
        if isinstance(other, (float, int)):
            if other == 0:
                return self
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
            if other == 1:
                return self
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

    def size(self) -> int:
        """Returns the number of unique boolean variables in the QUBO expression."""
        return len(self.get_variables())

    def __str__(self) -> str:
        return self._print()

    def __repr__(self) -> str:
        return (
            "("
            + (self.left.__repr__() if self.left is not None else "")
            + self.value
            + (self.right.__repr__() if self.right is not None else "")
            + ")"
        )

    def _collapse_coeffs(
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
            assert self.left is not None and self.right is not None

            degree += max(self.left._check_degree(), self.right._check_degree())
        else:
            if self.left is not None:
                degree += self.left._check_degree()
            if self.right is not None:
                degree += self.right._check_degree()
        return degree

    def get_terms_and_coefs(self) -> list[tuple[int, list[str]]]:
        """Creates a list of lists containing the coefficients of the monomials
        of the QUBO.

        Returns:
            The list of coefficients and variables in the QUBO expression.

        Examples:
            >>> x0 = QuboAtom('x0')
            >>> x1 = QuboAtom('x1')

            >>> expr = 3*x0 - x1 + 1
            >>> expr.get_terms_and_coefs()
            [(3, ['x0']), (-1, ['x1']), (1, [])]

            >>> expr = 3*(x0 | x1)
            >>> expr.get_terms_and_coefs()
            [(3, ['x0']), (3, ['x1']), (-3, ['x0', 'x1'])]
        """
        coeffs = []

        if self.left is None and self.right is None:
            return (
                [(int(self.value), [])]
                if isinstance(self, QuboConstant)
                else [(1, [self.value])]
            )

        left = []
        right = []

        if self.left is not None:
            left = self.left.get_terms_and_coefs()
        if self.right is not None:
            right = self.right.get_terms_and_coefs()

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
            return self._collapse_coeffs(left, right)

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
        coeffs = self.get_terms_and_coefs()
        known_vars = []
        for coeff in coeffs:
            for variable in coeff[1]:
                if variable[0] == "~":
                    if variable[1:] not in known_vars:
                        known_vars.append(variable[1:])
                elif variable not in known_vars:
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
        coeffs = self.get_terms_and_coefs()
        variables = self.get_variables()
        size = len(variables)
        matrix = np.zeros(shape=(size, size))
        constant = 0

        for coeff in coeffs:
            coef_names = coeff[1]
            if len(coef_names) == 0:
                constant += coeff[0]
            elif len(coef_names) == 1:
                coord = 0
                for j in range(size):
                    if coef_names[0][0] == '~':
                        if coef_names[0][1:] == variables[j]:
                            self._inverted_variables.extend([j, -1, 1])
                            coord = j
                    elif coef_names[0] == variables[j]:
                        coord = j
                matrix[coord][coord] += coeff[0]
            else:
                x_axis = 0
                y_axis = 0
                inv_variable = 0  # variable used to know which one of the two variables is inverted

                if coef_names[0][0] == "~":
                    x_axis = variables.index(coef_names[0][1:])
                    inv_variable += 1
                else:
                    x_axis = variables.index(coef_names[0])

                if coef_names[1][0] == "~":
                    y_axis = variables.index(coef_names[1][1:])
                    inv_variable += 2
                else:
                    y_axis = variables.index(coef_names[1])

                if coef_names[0][0] == "~" or coef_names[1][0] == "~":
                    self._inverted_variables.extend([x_axis, y_axis, inv_variable])

                matrix[x_axis][y_axis] += coeff[0] / 2
                matrix[y_axis][x_axis] += coeff[0] / 2

        return matrix, constant

    def to_cost_hamiltonian(self) -> Observable:
        """Converts this Qubo into a cost Hamiltonian, represented by an
        :class:`~mpqp.core.instruction.measurement.expectation_value.Observable`,
        that can typically be used in the QAOA algorithm.

        Returns:
            The cost Hamiltonian representing this Qubo.

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
        inv_variables = self._inverted_variables

        resulting_cost = _build_cost_hamiltonian(matrix, inv_variables, size)

        return Observable(
            np.diag(resulting_cost).astype(np.complex64) + np.eye(2**size) * constant
        )

    def _print(self, level: int = 1):
        left = self.left._print(level + 1) if self.left is not None else ""
        right = self.right._print(level + 1) if self.right is not None else ""
        if self.value == "*":
            if isinstance(self.right, QuboConstant):
                tmp = right
                right = left
                left = tmp
            if (isinstance(self.left, BinaryOperation) and self.left.value != "*") or (
                isinstance(self.right, BinaryOperation) and self.right.value != "*"
            ):
                return f"{left}{self.value}({right})"
        elif isinstance(self, UnaryOperation):
            if self.right and isinstance(
                self.right, Union[UnaryOperation | BinaryOperation]
            ):
                return f"{self.value}({right})"
        return left + self.value + right

    def simplify(self) -> "Qubo":
        """Returns the simplified form of the given Qubo.

        Notes:
            In the case of all the coefficients of an atom cancel each other
             then the simplified form will not declare the atom.

            The resulting cost hamiltonian hence would be changed.

        Examples:
            >>> x0 = QuboAtom("x0")
            >>> x1 = QuboAtom("x1")
            >>> expr = 3*x0 - x1 - 2*(x0^x1)
            >>> simplified = expr.simplify()
            >>> print(simplified)
            x0-3*x1+4*x0*x1
            >>> print(matrix_eq(expr.to_cost_hamiltonian().matrix, simplified.to_cost_hamiltonian().matrix))
            True
        """
        coefficients = {var: 0 for var in self.get_variables()}
        coeffs = self.get_terms_and_coefs()
        for coeff in coeffs:
            coef, var = coeff
            if len(var) == 1:
                coefficients.update({var[0]: coef + coefficients[var[0]]})
            else:
                var_name = f"{var[0]}*{var[1]}"
                reversed_name = f"{var[1]}*{var[0]}"
                keys = coefficients.keys()
                if keys.__contains__(var_name):
                    coefficients.update({var_name: coef + coefficients[var_name]})
                elif keys.__contains__(reversed_name):
                    coefficients.update(
                        {reversed_name: coef + coefficients[reversed_name]}
                    )
                else:
                    coefficients.update({var_name: coef})
        variables = coefficients.keys()
        result = 0
        for var in variables:
            if coefficients[var] == 0:
                continue
            if var.count('*') == 1:
                vars = var.split('*')
                if coefficients[var] > 0:
                    result += coefficients[var] * QuboAtom(vars[0]) * QuboAtom(vars[1])
                else:
                    result -= -coefficients[var] * QuboAtom(vars[0]) * QuboAtom(vars[1])
            else:
                if coefficients[var] > 0:
                    result += coefficients[var] * QuboAtom(var)
                else:
                    result -= -coefficients[var] * QuboAtom(var)
        assert isinstance(result, Qubo)
        return result


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
        >>> y = QuboAtom("y")
        >>> expr = 2 * x + 2 - 5 * y + 4 * (x ^ y)
        >>> print(expr.get_variables())
        ['x', 'y']
    """

    def __init__(self, value: str):
        import re

        if re.search("^[A-Z|a-z]", value) == None:
            raise ValueError("QuboAtoms have to be named using a letter at the start.")
        super().__init__(value, None, None)

    def __repr__(self):
        return 'QuboAtom("' + self.value + '")'


class BinaryOperation(Qubo):
    """Class defining binary operations in a Qubo expression.

    This class should be exclusively used by other classes and not by the user.

    Available binary operations : `+`, `-`, `*`
    Technically boolean operations (`|`, `&`, `^`) are available but they are decomposed
    into the previous operations.
    """

    def __init__(self, value: str, left: Qubo, right: Qubo):
        if value != "+" and value != "-" and value != "*":
            raise ValueError("Not an available binary operation")
        super().__init__(value, left, right)

    def __repr__(self) -> str:
        return (
            "("
            + self.left.__repr__()
            + " "
            + self.value
            + " "
            + self.right.__repr__()
            + ")"
        )


class UnaryOperation(Qubo):
    """Class defining a unary operation for a Qubo expression.

    This class should be exclusively used by other classes and not by the user.

    Unary operations supported : `-`
    """

    def __init__(self, value: str, right: Qubo):
        super().__init__(value, None, right)

    def __repr__(self) -> str:
        return self.value + self.right.__repr__()


class QuboConstant(Qubo):
    """Class defining constant terms in a Qubo expression.

    This class should be exclusively used by other classes and not by the user.

    Args:
        value: String hold the value of the int or float of the node.
    """

    def __init__(self, value: str):
        super().__init__(value)

    def __repr__(self) -> str:
        return self.value


def _build_cost_hamiltonian(matrix: Matrix, inv_variables: list[int], size: int):
    resulting_cost = np.zeros(shape=(2**size,))
    # Avoid recomputing several time the same hamiltonian
    hx_ns = [_generate_ith_Hamiltonian(size, i) for i in range(size)]
    for index in range(size):
        for j in range(index):
            if matrix[index][j] == 0:
                continue
            found = False  # check if one of the variables is inverted or not

            for k in range(0, len(inv_variables), 3):
                if (index == inv_variables[k] or index == inv_variables[k + 1]) and (
                    j == inv_variables[k] or j == inv_variables[k + 1]
                ):
                    local_cost = 1
                    if inv_variables[k + 2] == 0:
                        break
                    if inv_variables[k + 2] != 2:
                        local_cost *= _generate_ith_Hamiltonian(size, index, True)
                    else:
                        local_cost *= hx_ns[index]

                    if inv_variables[k + 2] >= 2:
                        local_cost *= _generate_ith_Hamiltonian(size, j, True)
                    else:
                        local_cost *= hx_ns[j]
                    resulting_cost += local_cost * matrix[index][j] * 2
                    found = True
                    break
            if not found:  # no inverted variables
                resulting_cost += matrix[index][j] * hx_ns[index] * hx_ns[j] * 2
        found = False
        for i in range(0, len(inv_variables), 3):
            if inv_variables[i] == index and inv_variables[i + 1] == -1:
                resulting_cost += matrix[index][index] * _generate_ith_Hamiltonian(
                    size, index, True
                )
                found = True
        if not found:
            resulting_cost += matrix[index][index] * hx_ns[index]
    return resulting_cost


def _generate_ith_Hamiltonian(size: int, i: int, neg: bool = False) -> Matrix:
    r"""Calculates the cost Hamiltonian `H(x_i)` for a given i-th binary parameter.
    This function has the purpose of being used with a QUBO object to generate the cost hamiltonian
    of a QUBO. see `~mpqp.execution.Qubo.to_cost_hamiltonian`

    `H(x_i)` is defined as:
    $$ H(x_i) = \frac{I^{\otimes n} - Z_i}{2} $$
    $$ \text{with } ~~ Z_i = \underbrace{I \otimes \cdots \otimes I}_{i} \otimes Z \otimes \underbrace{I \otimes \cdots \otimes I}_{n-i-1} $$

    Since in this case the hamiltonian will only be a diagonal matrix this function only returns a list of 1s and 0s.

    Args:
        Size: The total size of the hamiltonian in the context of QUBO it's the number of total variables in the expression.
        i: The index of the variable.
        neg: Boolean if the boolean variable is reversed.

    Example:
        >>> print(generate_ith_Hamiltonian(2,0))
        [0. 0. 1. 1.]
        >>> print(generate_ith_Hamiltonian(2,1))
        [0. 1. 0. 1.]
    """
    if i >= size:
        raise ValueError(
            "The index of the variable cannot be equal or higher than the total number of variables."
        )
    Z_i = np.array([1, -1])

    if i != 0:
        Z_i = np.kron(np.ones(2**i), Z_i)

    if size - i - 1 != 0:
        Z_i = np.kron(Z_i, np.ones(2 ** (size - i - 1)))

    result = (np.ones(2**size) - Z_i) / 2
    if neg:
        for j in range(len(result)):
            if result[j] == 0:
                result[j] = 1
            else:
                result[j] = 0
    return result
