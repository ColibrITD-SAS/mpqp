"""This module is designed to generate and manipulate Quadratic Unconstrained Binary Optimization
(Qubo) problems, which are widely use to describe various optimization problems.

We model :class:`~mpqp.execution.vqa.qubo.Qubo` expressions as binary trees, in the form of an Abstract Syntax Tree,
useful for taking into account the hierarchies and priorities between the operations appearing in the
:class:`~mpqp.execution.vqa.qubo.Qubo`. A node of this :class:`~mpqp.execution.vqa.qubo.Qubo` tree corresponds to an
operation while the leaves of the tree are boolean variables or constants on which the binary or unary operations are
applied. One can find below an example of such tree structure for the Qubo `-1 + 2x`::

          Addition
        /          \\
 UnaryMinus   Multiplication
    /           /       \\
   1           2    QuboAtom("x")

Most of the classes implemented in this module are not meant to be directly used by the user, but are instantiated
in the background when arithmetic or boolean operations are used. Every node in the tree is inheriting from the
:class:`~mpqp.execution.vqa.qubo.Qubo` class, regrouping all the common implementations for operations and simplifications.
The only class that is meant to be directly instantiated by the user is :class:`~mpqp.execution.vqa.qubo.QuboAtom`,
allowing one to define boolean variables that will be manipulated withing the :class:`~mpqp.execution.vqa.qubo.Qubo`.

In the context of the MPQP library, these classes are used in the QAOA module to encode the optimization problem solved
by the :func:`~mpqp.execution.vqa.qaoa.qaoa_solver`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
from mpqp.measures import Observable
from mpqp.tools.generics import Matrix
from mpqp.tools.operators import *
from typeguard import typechecked


class Qubo(ABC):
    """Abstract class defining an optimization problem following the Qubo representation. This class is instantiated
    through the use of :class:`~mpqp.execution.vqa.qubo.QuboAtom`, not directly (see examples below).

    A Qubo is defined as a quadratic expression of boolean variables, hence, the available operators are:
        - ``&`` : the logical AND operation between two :class:`~mpqp.execution.vqa.qubo.QuboAtom`
        - ``|`` : the logical OR operation between two :class:`~mpqp.execution.vqa.qubo.QuboAtom`
        - ``^`` : the logical XOR operation between two :class:`~mpqp.execution.vqa.qubo.QuboAtom`
        - ``+`` : the artihmetic addition operator between two :class:`~mpqp.execution.vqa.qubo.Qubo` expressions
        - ``-`` : the artihmetic subtraction operator between two :class:`~mpqp.execution.vqa.qubo.Qubo` expressions
        - ``*`` : the artihmetic multiplication operator between two :class:`~mpqp.execution.vqa.qubo.Qubo` expressions

    The evaluation of a Qubo expression can be minimized using the :func:`~mpqp.execution.vqa.qaoa.qaoa_solver` solver.

    Conceptually, a :class:`~mpqp.execution.vqa.qubo.Qubo` is the the root node of a binary tree representing the whole
    Qubo expression with :class:`~mpqp.execution.vqa.qubo.QuboAtom` or :class:`~mpqp.execution.vqa.qubo.QuboConstant` .
    as its leafs. The classes :class:`BinaryOperation` and :class:`UnaryOperation` are used as nodes representing the
    Qubo's operators.

    Args:
        value: Information concerning the root node, can be an ``Operator``, the name of the boolean variable or
            the value of the constant.
        left: Left child of this Qubo root node.
        right: Right child of this Qubo root node.

    Examples:
        >>> x0 = QuboAtom('x0')
        >>> x1 = QuboAtom('x1')
        >>> qubo1 = 3*x0*x1 + x0 - 2*x1
        >>> qubo2 = 0.7*x0 + 0.3*x1 - 0.02
        >>> print(qubo1)
        3*x0*x1+x0-2*x1
        >>> qubo2.__repr__()
        '(((QuboAtom("x0") * 0.7) + (QuboAtom("x1") * 0.3)) - 0.02)'
        >>> print(qubo1 + qubo2)
        3*x0*x1+x0-2*x1+0.7*x0+0.3*x1-0.02
        >>> print(qubo2 * qubo2)
        (0.7*x0+0.3*x1-0.02)*(0.7*x0+0.3*x1-0.02)
        >>> print(qubo1 - x0)
        3*x0*x1+x0-2*x1-x0
        >>> print((qubo1 + qubo2).simplify())
        1.7*x0-1.7*x1+3*x0*x1-0.02
        >>> print((qubo2 * x0).simplify())
        0.68*x0+0.3*x1*x0
        >>> print((qubo1 - x0).simplify())
        -(2*x1)+3*x0*x1
        >>> print(qubo1.to_cost_hamiltonian().pauli_string)
        0.25*I@I + 0.25*I@Z - 1.25*Z@I + 0.75*Z@Z
        >>> pprint(qubo1.to_cost_hamiltonian().matrix)
        [[0, 0 , 0, 0],
         [0, -2, 0, 0],
         [0, 0 , 1, 0],
         [0, 0 , 0, 2]]
        >>> qubo1.depth()
        4
        >>> (qubo1 -5).get_terms_and_coefs()
        [(3, ['x0', 'x1']), (1, ['x0']), (-2, ['x1']), (-5, [])]
        >>> qubo2.size()
        2
        >>> qubo2.weight_matrix()
        (array([[0.7, 0. ],
               [0. , 0.3]]), -0.02)
        >>> -(x0 - x1)
        -(QuboAtom("x0") - QuboAtom("x1"))
    """

    def __init__(
        self,
        value: Union[str, Operator, float],
        left: Optional["Qubo"] = None,
        right: Optional["Qubo"] = None,
    ):

        self.left = left
        self.right = right
        self.value = value

        # an inverted variable would be something like `~x0`
        # We keep track of them to perform a special encoding when computing weight_matrix
        self._inverted_variables = []

        # This integer is used to keep track of the degree of the Qubo polynomial.
        # It is needed because, by definition, the degree has to be <= 2
        self._degree = 0

    def __neg__(self) -> "Qubo":
        if isinstance(self, UnaryOperation):
            if TYPE_CHECKING:
                assert self.right
            return self.right
        return UnaryOperation(Minus(), self)

    def __add__(self, other: Union["Qubo", int, float]) -> "Qubo":
        if isinstance(self, QuboAtom) and isinstance(other, QuboAtom):
            inverted = False
            if self.value[0] == "~":
                name_self = self.value[1:]
                inverted = True
            else:
                name_self = self.value

            if other.value[0] == "~":
                other_self = other.value[1:]
                inverted = not inverted
            else:
                other_self = other.value

            if inverted:
                if name_self == other_self:
                    return QuboConstant(1)
        if isinstance(other, (float, int)):
            if other == 0:
                return self
            if other < 0:
                return self - (-other)
            other = QuboConstant(other)
        if isinstance(other, UnaryOperation) and isinstance(other.value, Minus):
            return self - (-other)
        current = BinaryOperation(Addition(), self, other)
        current._degree = max(self._degree, other._degree)
        return current

    def __radd__(self, other: Union["Qubo", int, float]) -> "Qubo":
        return self + other

    def __sub__(self, other: Union["Qubo", int, float]) -> "Qubo":
        if isinstance(other, (float, int)):
            if other == 0:
                return self
            if other < 0:
                return self + (-other)
            other = QuboConstant(other)
        if isinstance(other, UnaryOperation) and isinstance(other.value, Minus):
            return self + (-other)
        current = BinaryOperation(Subtraction(), self, other)
        current._degree = max(self._degree, other._degree)
        return current

    def __rsub__(self, other: Union["Qubo", int, float]) -> "Qubo":
        if other == 0:
            return -self
        return self - other

    def __mul__(self, other: Union["Qubo", int, float]) -> "Qubo":
        degree = self._degree
        if isinstance(other, (float, int)):
            if other == 1:
                return self
            if isinstance(self, QuboConstant):
                self.value *= other
                return self
            other = QuboConstant(other)
        elif isinstance(self, QuboAtom) and isinstance(other, QuboAtom):
            inverted = False
            if self.value[0] == "~":
                name_self = self.value[1:]
                inverted = True
            else:
                name_self = self.value

            if other.value[0] == "~":
                other_self = other.value[1:]
                inverted = not inverted
            else:
                other_self = other.value

            if inverted:
                if name_self == other_self:
                    return QuboConstant(0)

            if self.value == other.value:
                return self
        if not isinstance(other, QuboConstant):
            degree += other._degree
            if degree > 2:
                raise ValueError(
                    f"The degree of the Qubo shouldn't be more than 2 not {degree}"
                )

        current = BinaryOperation(Multiplication(), self, other)
        current._degree = degree
        return current

    def __rmul__(self, other: Union["Qubo", int, float]) -> "Qubo":
        return self * other

    def size(self) -> int:
        """Returns the number of unique boolean variables in the Qubo expression."""
        return len(self.get_variables())

    def __str__(self) -> str:
        return self._print()

    def __repr__(self) -> str:
        return (
            "("
            + (self.left.__repr__() if self.left is not None else "")
            + str(self.value)
            + (self.right.__repr__() if self.right is not None else "")
            + ")"
        )

    def evaluate(self, variables: dict[str, bool]) -> float:
        """Function used to evaluate the result of a Qubo expression.

        Args:
            variables: Mapping of the variables to their binary value.

        Returns:
            The value of the expression for the given values.

        Examples:
            >>> x0 = QuboAtom("x0")
            >>> x1 = QuboAtom("x1")
            >>> expr = 3*x0
            >>> expr.evaluate({"x0": True})
            3
            >>> expr.evaluate({"x0": False})
            0
            >>> expr = 3*(~x0)
            >>> expr.evaluate({"x0": False})
            3
            >>> expr = 3*x0*x1 - 2*x1
            >>> expr.evaluate({"x1": True, "x0": False})
            -2
        """
        terms = self.get_terms_and_coefs()
        result = 0
        for term in terms:
            coef, vars = term
            if len(vars) == 0:
                result += coef
            else:
                inverted_indexes = []
                for i in range(len(vars)):
                    if vars[i][0] == "~":
                        inverted_indexes.append(i)
                        vars[i] = vars[i][1:]

                if not all(variables.__contains__(var) for var in vars):
                    raise ValueError(
                        f"Variables {vars} were not found in the dictionary."
                    )
                local_result = 1
                for i in range(len(vars)):
                    if inverted_indexes.count(i) == 1:
                        local_result *= not variables[vars[i]]
                    else:
                        local_result *= variables[vars[i]]
                result += coef * local_result
        return result

    def get_terms_and_coefs(self) -> list[tuple[float, list[str]]]:
        """Creates a list of lists containing the coefficients of the monomials
        of the Qubo.

        Returns:
            The coefficients and variables in the Qubo expression.

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
            assert isinstance(self, (QuboConstant, QuboAtom))
            return (
                [(self.value, [])]
                if isinstance(self, QuboConstant)
                else [(1, [self.value])]
            )

        left = []
        right = []

        if self.left is not None:
            left = self.left.get_terms_and_coefs()
        if self.right is not None:
            right = self.right.get_terms_and_coefs()

        if isinstance(self.value, Addition):
            coeffs.extend(left)
            coeffs.extend(right)
        elif isinstance(self.value, (Subtraction, Minus)):
            coeffs.extend(left)
            for i in range(len(right)):
                hold = right[i][0]
                hold *= -1
                right[i] = (hold, right[i][1])
            coeffs.extend(right)
        elif isinstance(self.value, Multiplication):
            return _collapse_coeffs(left, right)

        return coeffs

    def depth(self) -> int:
        """Return the maximum depth of the tree representing the Qubo expression."""
        return self._depth()

    def _depth(self, level: int = 0) -> int:
        return max(
            level if self.right is None else self.right._depth(level + 1),
            level if self.left is None else self.left._depth(level + 1),
        )

    def get_variables(self) -> list[str]:
        """This function generates a list containing every unique variables
        used in the expression.
        They are ordered from the left of the expression to the right.

        Returns:
            A list of all of the unique boolean variables of the Qubo.

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

    def weight_matrix(self) -> tuple[npt.NDArray[np.float64], float]:
        r"""Generates the weight matrix corresponding to this Qubo expression.
        The weight matrix regroups the coefficients that appears in front of all
        possible combinations of quadratic binary monomials.

        The coefficient in front of QuboAtom `x_i` (which correspond to
        `x_i \cdot x_i`) gives us `i`-th the diagonal element of the weigh matrix.
        For instance, the Qubo written as `3x_0 + 2x_1`, is a two-by-two
        diagonal matrix with elements `[3,2]`.

        The off-diagonal element on the `i`-th line and the `j`-th column
        corresponds to the half of the coefficient in front of the binary
        monomial `x_i \cdot x_j`. One can notice that the weight matrix is
        indeed a symmetric matrix.

        If a constant is appearing in the Qubo, it will be stored aside of the
        weight matrix and returned by this function. This is useful in the
        context of the generation of the corresponding cost Hamiltonian.

        Returns:
            A tuple containing the weight matrix of this Qubo and the potential additive constant.

        Examples:
            >>> x0 = QuboAtom('x0')
            >>> x1 = QuboAtom('x1')
            >>> x2 = QuboAtom('x2')
            >>> x3 = QuboAtom('x3')
            >>> expr = 2 * x0 + 3 * x1 + 4 * x0 * x2 + x3 + 18
            >>> w_matrix, add_constant = expr.weight_matrix()
            >>> pprint(w_matrix)
            [[2, 0, 2, 0],
             [0, 3, 0, 0],
             [2, 0, 0, 0],
             [0, 0, 0, 1]]
            >>> print(add_constant)
            18.0
        """
        coeffs = self.get_terms_and_coefs()
        variables = self.get_variables()
        nb_vars = len(variables)
        matrix = np.zeros(shape=(nb_vars, nb_vars))
        constant = 0.0
        self._inverted_variables = []

        for coeff in coeffs:
            coef_names = coeff[1]
            if len(coef_names) == 0:
                constant += coeff[0]
            elif len(coef_names) == 1:
                coord = 0
                for j in range(nb_vars):
                    if coef_names[0][0] == '~':
                        if coef_names[0][1:] == variables[j]:
                            # This list is encoded with the coordinates of the variable's weight
                            self._inverted_variables.extend([j, j, 1])
                            coord = j
                    elif coef_names[0] == variables[j]:
                        coord = j
                matrix[coord][coord] += coeff[0]
            else:
                x_axis = 0
                y_axis = 0

                # variable used to know which one of the two variables is inverted
                inv_variable = 0  # 1 : first variables, 2 : second, 3 : both

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
                    # This list is encoded with the coordinates of the variables' weight
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
        optimized = self.simplify()
        matrix, constant = optimized.weight_matrix()
        size = matrix.shape[0]
        inv_variables = optimized._inverted_variables

        resulting_cost = _build_cost_hamiltonian(matrix, inv_variables, size)

        return Observable(
            np.diag(resulting_cost).astype(np.complex64) + np.eye(2**size) * constant
        )

    def _print(self):
        """Prints the expression of the Qubo including parenthesis for correct
        operation priority."""
        left_str = self.left._print() if self.left is not None else ""
        right_str = self.right._print() if self.right is not None else ""
        if isinstance(self.value, Multiplication):
            assert self.left and self.right
            if isinstance(self.right, QuboConstant):
                tmp = right_str
                right_str = left_str
                left_str = tmp
            result = ""
            if isinstance(self.left.value, (Addition, Subtraction)):
                result += f"({left_str})"
            else:
                result += f"{left_str}"
            result += f"{self.value}"
            if isinstance(self.right.value, (Addition, Subtraction)):
                result += f"({right_str})"
            else:
                result += f"{right_str}"
            return result
        elif isinstance(self, UnaryOperation):
            if self.right and isinstance(self.right, (UnaryOperation, BinaryOperation)):
                return f"{self.value}({right_str})"
        return left_str + str(self.value) + right_str

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
            >>> print(
            ...     matrix_eq(
            ...         expr.to_cost_hamiltonian().matrix,
            ...         simplified.to_cost_hamiltonian().matrix
            ...     )
            ... )
            True
        """
        coefficients: dict[str, float] = {var: 0 for var in self.get_variables()}
        constant = 0
        coeffs = self.get_terms_and_coefs()
        for coeff in coeffs:
            coef, var = coeff
            if len(var) == 0:
                constant += coef
            elif len(var) == 1:
                if var[0][0] == "~":
                    if coefficients.get(var[0]) is None:
                        coefficients.update({var[0]: coef})
                coefficients.update({var[0]: round(coef + coefficients[var[0]], 15)})
            else:
                var_name = f"{var[0]}*{var[1]}"
                reversed_name = f"{var[1]}*{var[0]}"
                keys = coefficients.keys()
                if keys.__contains__(var_name):
                    coefficients.update(
                        {var_name: round(coef + coefficients[var_name], 15)}
                    )
                elif keys.__contains__(reversed_name):
                    coefficients.update(
                        {reversed_name: round(coef + coefficients[reversed_name], 15)}
                    )
                else:
                    coefficients.update({var_name: coef})
        variables = coefficients.keys()
        result = 0
        for var in variables:
            if coefficients[var] == 0:
                continue
            if var.count('*') == 1:
                vars_split = var.split('*')
                minus = coefficients[var] < 0
                current = -coefficients[var] if minus else coefficients[var]

                for split in vars_split:
                    if split[0] == "~":
                        current *= ~QuboAtom(split[1:])
                    else:
                        current *= QuboAtom(split)

                if minus:
                    result -= current
                else:
                    result += current
            else:
                if coefficients[var] > 0:
                    if var[0] == "~":
                        result += coefficients[var] * ~QuboAtom(var[1:])
                    else:
                        result += coefficients[var] * QuboAtom(var)
                else:
                    if var[0] == "~":
                        result -= -(coefficients[var]) * ~QuboAtom(var)
                    else:
                        result -= -(coefficients[var]) * QuboAtom(var)
        if constant < 0:
            result -= -constant
        else:
            result += constant
        if not isinstance(result, Qubo):
            return QuboConstant(result)
        return result


@typechecked
class QuboAtom(Qubo):
    """Class defining a boolean variable for a Qubo problem.

    See class :class:`~mpqp.execution.vqa.qubo.Qubo` for full usage of this class.

    Arg:
        value: String holding the name of the variable.

    Examples:
        >>> x = QuboAtom("x")
        >>> expr = 2 * x + 2
        >>> print(expr.get_variables())
        ['x']
        >>> y = QuboAtom("y")
        >>> expr = 2 * x + 2 - 5 * y + 4 * (x ^ y)
        >>> print(expr.get_variables())
        ['x', 'y']
        >>> print(~y)
        ~y
        >>> ~(~y)
        QuboAtom("y")
        >>> print(x | y)
        x+y-x*y
        >>> print(x & y)
        x*y
        >>> print(x ^ y)
        x+y-2*x*y
    """

    def __init__(self, value: str):
        import re

        if re.search("^[A-Z|a-z]", value) is None:
            raise ValueError("QuboAtoms have to be named using a letter at the start.")
        if re.search("[*|^|+|-|&|~|(|)|=|%]", value) is not None:
            raise ValueError(
                "QuboAtoms cannot be named using operators or special characters."
            )
        super().__init__(value, None, None)
        self.value = value
        self._degree = 1

    def __invert__(self) -> "QuboAtom":
        from copy import deepcopy

        copy = deepcopy(self)
        if self.value[0] == "~":
            copy.value = self.value[1:]
            return copy
        copy.value = '~' + self.value
        return copy

    def __and__(self, other: "QuboAtom") -> "Qubo":
        inverted = False

        if self.value[0] == "~":
            name_self = self.value[1:]
            inverted = True
        else:
            name_self = self.value

        if other.value[0] == "~":
            other_self = other.value[1:]
            inverted = not inverted
        else:
            other_self = other.value

        if inverted:
            if name_self == other_self:
                return QuboConstant(0)

        if self.value == other.value:
            return self
        return self * other

    def __or__(self, other: "QuboAtom") -> "Qubo":
        inverted = False

        if self.value[0] == "~":
            name_self = self.value[1:]
            inverted = True
        else:
            name_self = self.value

        if other.value[0] == "~":
            other_self = other.value[1:]
            inverted = not inverted
        else:
            other_self = other.value

        if inverted:
            if name_self == other_self:
                return QuboConstant(1)

        if self.value == other.value:
            return self
        return self + other - self * other

    def __xor__(self, other: "QuboAtom") -> "Qubo":
        inverted = False

        if self.value[0] == "~":
            name_self = self.value[1:]
            inverted = True
        else:
            name_self = self.value

        if other.value[0] == "~":
            other_self = other.value[1:]
            inverted = not inverted
        else:
            other_self = other.value

        if inverted:
            if name_self == other_self:
                return QuboConstant(1)

        if self.value == other.value:
            return QuboConstant(0)

        return self + other - 2 * (self * other)

    def __repr__(self):
        if self.value[0] == "~":
            return f"~QuboAtom(\"{self.value[1:]}\")"
        return f"QuboAtom(\"{self.value}\")"


class BinaryOperation(Qubo):
    """Class defining binary operations in a Qubo expression. A binary operation is defined by an operator
    (`+`, `-` or `*`) applied on two operands (left and right), each of them potentially being:
        - a Qubo expression,
        - a QuboAtom variable,
        - a QuboConstant,
        - a BinaryOperation

    This class should be exclusively used by other classes and not by the user.

    Available binary operations: ``+``, ``-``, ``*``.
    (Technically boolean operations: ``|``, ``&``, ``^`` are available but they are
    decomposed into the previously mentioned operations.)

    Args:
        value: Operator corresponding to the modeled operation by this class.
        left: Left part of the equation, in term of a binary tree it's the left child.
        right: Right part of the equation, in term of a binary tree it's the right child.

    Examples:
        >>> BinaryOperation(Multiplication(), QuboConstant(3), QuboAtom("x"))
        (3 * QuboAtom("x"))
        >>> BinaryOperation(Subtraction(), QuboAtom("y"), QuboAtom("x"))
        (QuboAtom("y") - QuboAtom("x"))
    """

    def __init__(self, value: BinaryOperator, left: Qubo, right: Qubo):
        super().__init__(value, left, right)

    def __repr__(self) -> str:
        return f"({repr(self.left)} {str(self.value)} {repr(self.right)})"


class UnaryOperation(Qubo):
    """Class defining a unary operation for a Qubo expression.
    The value is the operator and the rest of the equation is store in the right child.

    This class should be exclusively used by other classes and not by the user.

    Unary operations supported: ``-``, ``~``.

    Args:
        value: The unary operator.
        right: The Qubo representing the operand.

    Examples:
        >>> UnaryOperation(Minus(), QuboAtom("x"))
        -QuboAtom("x")
        >>> UnaryOperation(Not(), QuboAtom("x"))
        ~QuboAtom("x")
    """

    def __init__(self, value: UnaryOperator, right: Qubo):
        super().__init__(value, None, right)

    def __repr__(self) -> str:
        if isinstance(self.right, UnaryOperation):
            return f"{str(self.value)}({repr(self.right)})"
        return str(self.value) + repr(self.right)


class QuboConstant(Qubo):
    """Class defining constant terms (real numbers) in a Qubo expression.
    In the context of the tree this node is always a leaf.

    This class should be exclusively used by other classes and not by the user.

    Args:
        value: The value of the constant in the expression.

    Examples:
        >>> QuboConstant(0)
        0
        >>> QuboConstant(3.2)
        3.2
        >>> (QuboConstant(3) - QuboConstant(4.0)).simplify()
        -1.0
    """

    def __init__(self, value: float):
        super().__init__(value, None, None)
        self.value = value

    def __repr__(self) -> str:
        return f"{self.value}"


def _build_cost_hamiltonian(matrix: Matrix, inv_variables: list[int], size: int):
    resulting_cost = np.zeros(shape=(2**size,))
    # Avoid recomputing several time the same hamiltonian
    hx_ns = [_generate_ith_Hamiltonian(size, i) for i in range(size)]
    for i in range(size):
        for j in range(i):
            if matrix[i][j] == 0:
                continue
            found = False  # check if one of the variables is inverted or not
            # inv_variables is encoded in the method matrix.
            for k in range(0, len(inv_variables), 3):
                if (i == inv_variables[k] or i == inv_variables[k + 1]) and (
                    j == inv_variables[k] or j == inv_variables[k + 1]
                ):
                    local_cost = 1
                    if inv_variables[k + 2] == 0:
                        break
                    if inv_variables[k + 2] != 2:
                        local_cost *= _generate_ith_Hamiltonian(size, j, True)
                    else:
                        local_cost *= hx_ns[j]

                    if inv_variables[k + 2] >= 2:
                        local_cost *= _generate_ith_Hamiltonian(size, i, True)
                    else:
                        local_cost *= hx_ns[i]
                    resulting_cost += local_cost * matrix[i][j] * 2
                    found = True
                    break
            if not found:  # no inverted variables
                resulting_cost += matrix[i][j] * hx_ns[i] * hx_ns[j] * 2
        found = False
        for index in range(0, len(inv_variables), 3):
            if inv_variables[index] == i and inv_variables[index + 1] == i:
                resulting_cost += matrix[i][i] * _generate_ith_Hamiltonian(
                    size, index, True
                )
                found = True
        if not found:
            resulting_cost += matrix[i][i] * hx_ns[i]
    return resulting_cost


def _generate_ith_Hamiltonian(size: int, i: int, neg: bool = False) -> Matrix:
    r"""Computes the cost Hamiltonian `H(x_i)` for a given i-th binary
    parameter. This function has the purpose of being used with a Qubo object to
    generate the cost hamiltonian of a Qubo. See
    `~mpqp.execution.Qubo.to_cost_hamiltonian`

    `H(x_i)` is defined as:
    $$ H(x_i) = \frac{I^{\otimes n} - Z_i}{2} $$
    $$ \text{with } ~~ Z_i = \underbrace{I \otimes \cdots \otimes I}_{i} \otimes Z \otimes \underbrace{I \otimes \cdots \otimes I}_{n-i-1} $$

    Since in this case the hamiltonian will only be a diagonal matrix this
    function only returns a list of 1s and 0s.

    Args:
        size: The total size of the hamiltonian in the context of Qubo it's the
            number of total variables in the expression.
        i: The index of the variable.
        neg: Boolean if the boolean variable is reversed.

    Returns:
        The diagonal of the cost hamiltonian
    """
    if i >= size:
        raise ValueError(
            "The index of the variable cannot be equal or higher than the total"
            " number of variables."
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


def _collapse_coeffs(
    lhs: list[tuple[float, list[str]]], rhs: list[tuple[float, list[str]]]
):
    """This function distribute the coefficient an expression in parenthesis.
    It is used in the creation of the Qubo to be able to accurately create the
    weight matrix. ``lhs`` and ``rhs`` respectively stand for left and right
    hand sides of the multiplication. It returns the multiplication with the
    coeff distributed.
    """
    if len(rhs) == 1:
        for i in range(len(lhs)):
            hold: float = lhs[i][0]
            hold *= rhs[0][0]
            if rhs[0][1] == lhs[i][1]:
                lhs[i] = (hold, lhs[i][1])
                continue
            lhs[i][1].extend(rhs[0][1])
            lhs[i] = (hold, lhs[i][1])
        return lhs
    elif len(lhs) == 1:
        for i in range(len(rhs)):
            hold: float = rhs[i][0]
            hold *= lhs[0][0]
            if rhs[i][1] == lhs[0][1]:
                rhs[i] = (hold, rhs[i][1])
                continue
            rhs[i][1].extend(lhs[0][1])
            rhs[i][1].reverse()
            rhs[i] = (hold, rhs[i][1])
        return rhs
    else:
        from copy import deepcopy

        result = []
        for i in range(len(rhs)):
            for j in range(len(lhs)):
                hold_j = lhs[j][0]
                hold_j *= rhs[i][0]
                if rhs[i][1] == lhs[j][1]:
                    result.append((hold_j, deepcopy(rhs[i][1])))
                    continue
                vars = deepcopy(lhs[j][1])
                vars.extend(rhs[i][1])
                result.append((hold_j, vars))

        return result
