import numpy as np
import numpy.typing as npt

from mpqp.measures import Observable


class Qubo:
    """
    Class defining a QUBO representation, used to represent decision problems.

    A QUBO is defined by a quadratic expression of boolean variables, hence, the available operators are : + , - , * , & , | , ^

    The QUBO expression can be solved with the qaoa_solver function.

    Examples:
        >>> x0 = Qubo('x0')
        >>> x1 = Qubo('x1')
        >>> expr = 3*x0*x1 + x0 - 2*x1
        >>> print(expr.to_cost_hamiltonian().matrix)
        [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j -2.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j]]

    """

    def __init__(
        self,
        value: str,
        left: "None | Qubo" = None,
        right: "Qubo | None" = None,
        flag: bool = False,
    ):
        self.left = left
        self.right = right
        self.value = value
        self.flag = flag

    def __neg__(self) -> "Qubo":
        return self * -1

    def __add__(self, other: "Qubo | int | float") -> "Qubo":
        if isinstance(other, int) or isinstance(other, float):
            other = Qubo(str(other), flag=True)
        current = Qubo("+", self, other)
        return current

    def __radd__(self, other: "Qubo | int | float") -> "Qubo":
        return self + other

    def __sub__(self, other: "Qubo | int | float") -> "Qubo":
        if isinstance(other, int) or isinstance(other, float):
            other = Qubo(str(other), flag=True)
        current = Qubo("-", self, other)
        return current

    def __rsub__(self, other: "Qubo | int | float") -> "Qubo":
        return self - other

    def __mul__(self, other: "Qubo | int | float") -> "Qubo":
        if isinstance(other, int) or isinstance(other, float):
            other = Qubo(str(other), flag=True)
        current = Qubo("*", self, other)
        return current

    def __rmul__(self, other: "Qubo | int | float") -> "Qubo":
        return self * other

    def __and__(self, other: "Qubo") -> "Qubo":
        return self * other

    def __or__(self, other: "Qubo") -> "Qubo":
        return self + other - self * other

    def __xor__(self, other: "Qubo") -> "Qubo":
        return self + other - self * other * 2

    def __eq__(self, other: object) -> bool:
        if other == None:
            return False
        return False

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

    def get_coeffs(
        self, coeffs: list[tuple[int, list[str]]]
    ) -> list[tuple[int, list[str]]]:
        """
        Creates a list of lists containing the coefficients of the monomials of the QUBO.

        Args:
            coeffs: An empty list

        Returns:
            List[tuple[int, list[str]]]: The list of coefficients and variables in the QUBO expression

        Examples:
            >>> x0 = Qubo('x0')
            >>> x1 = Qubo('x1')

            >>> expr = 3*x0 - x1 + 1
            >>> expr.get_coeffs([])
            [(3, ['x0']), (-1, ['x1']), (1, [])]

            >>> expr = 3*(x0 | x1)
            >>> expr.get_coeffs([])
            [(3, ['x0']), (3, ['x1']), (-3, ['x0', 'x1'])]
        """
        if self == None:
            return []

        if self.left == None and self.right == None:
            return [(int(self.value), [])] if self.flag else [(1, [self.value])]

        left = []
        right = []

        if self.left != None:
            left = self.left.get_coeffs([])
        if self.right != None:
            right = self.right.get_coeffs([])

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
        """
        Returns a list of all of the unique boolean variables of the QUBO. They are ordered from the left of the expression to the right.

        Examples:
            >>> x0 = Qubo('x0')
            >>> x1 = Qubo('x1')

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
        coeffs = self.get_coeffs([])
        known_vars = coeffs[0][1]
        index = 0
        seen = False
        for i in range(len(coeffs)):
            for j in range(len(coeffs[i][1])):
                seen = False
                for k in range(len(known_vars)):
                    if coeffs[i][1][j] == known_vars[k]:
                        seen = True
                    else:
                        index = j

            if not seen:
                if len(coeffs[i][1]) != 0:
                    known_vars.append(coeffs[i][1][index])

        return known_vars

    def get_size(self):
        """
        Returns the number of unique boolean variables in the QUBO expression.
        """
        return len(self.get_variables())

    def create_matrix(self) -> tuple[npt.NDArray[np.float64], int]:
        """
        Creates the weight matrix of this QUBO expression.

        Returns:
            A Tuple composed of the Weight Matrix and the eventual constants to add.

        Examples:
            >>> x0 = Qubo('x0')
            >>> x1 = Qubo('x1')
            >>> x2 = Qubo('x2')
            >>> x3 = Qubo('x3')
            >>> expr = 2 * x0 + 3 * x1 + 4 * x0 * x2 + x3
            >>> print(expr.create_matrix()[0])
            [[2. 0. 2. 0.]
             [0. 3. 0. 0.]
             [2. 0. 0. 0.]
             [0. 0. 0. 1.]]
        """
        coeffs = self.get_coeffs(coeffs=[])
        vars = self.get_variables()
        size = len(vars)
        matrix = np.zeros(shape=(size, size))
        constant = 0

        for i in range(len(coeffs)):
            if len(coeffs[i][1]) == 0:
                constant += coeffs[i][0]
            elif len(coeffs[i][1]) == 1:
                coord = 0
                for j in range(size):
                    if coeffs[i][1][0] == vars[j]:
                        coord = j
                matrix[coord][coord] += coeffs[i][0]
            else:
                abs = 0
                ord = 0
                for j in range(size):
                    if coeffs[i][1][0] == vars[j]:
                        abs = j
                    if coeffs[i][1][1] == vars[j]:
                        ord = j

                matrix[abs][ord] += coeffs[i][0] / 2
                matrix[ord][abs] += coeffs[i][0] / 2

        return matrix, constant

    def to_cost_hamiltonian(self) -> Observable:
        """
        Converts the QUBO matrix into a cost Hamiltonian.

        Returns:
            Observable: The cost Hamiltonian.

        Examples:
            >>> x_0 = Qubo("x_0")
            >>> x_1 = Qubo("x_1")
            >>> expr = 3 * x_0 * x_1 - 4 * x_0 - 2 * x_1 + 1
            >>> print(expr.to_cost_hamiltonian().matrix)
            [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j -1.+0.j  0.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j -3.+0.j  0.+0.j]
             [ 0.+0.j  0.+0.j  0.+0.j -2.+0.j]]
        """
        matrix, constant = self.create_matrix()
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
        r'''
        Calculates the cost Hamiltonian H(x_i) for a given i-th binary parameter.
        If follow this formula:
        $$ H(x_i) = \frac{I^{\otimes n} - Z_i}{2} $$
        $$ \text{with } ~~ Z_i = \underbrace{I \otimes \cdots \otimes I}_{i} \otimes Z \otimes \underbrace{I \otimes \cdots \otimes I}_{n-i-1} $$
        '''
        Z = np.array([1, -1])
        Z_i = Z

        if i != 0:
            Z_i = np.kron(np.ones(2**i), Z_i)

        if size - i - 1 != 0:
            Z_i = np.kron(Z_i, np.ones(2 ** (size - i - 1)))

        result = (np.ones(2**size) - Z_i) / 2
        return result

    def _print(self, level: int = 1, verbose: bool = False):
        left = ''
        right = ''
        if verbose:
            print(" " * level + "{")
            print(" " * (level + 1) + self.value + " FLAG:" + str(self.flag))

        if self.left != None:
            left = self.left._print(level + 1, verbose)
            if verbose:
                print(" " * (level + 1) + "}")
        if self.right != None:

            right = self.right._print(level + 1, verbose)
            if verbose:
                print(" " * (level + 1) + "}")
        if self.value == "*":
            return right + self.value + left
        return left + self.value + right

    def pprint(self, verbose: bool = False):
        """
        Prints the QUBO expression.
        Arg:
            verbose : A boolean value to print the tree representation of the expression. If False then it will only print the resulting expression.
        """
        print(self._print(1, verbose))
