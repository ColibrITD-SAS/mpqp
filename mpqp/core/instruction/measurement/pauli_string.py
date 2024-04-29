"""Represents Pauli strings, which is linear combinations of
:class:`PauliMonomial` which is a combination of :class:`PauliAtom`.
:class:`PauliString` objects can be added, subtracted, and multiplied by
scalars. They also support matrix multiplication with other :class:`PauliString`
objects.
"""

from __future__ import annotations

from copy import deepcopy
from functools import reduce
from numbers import Real
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

FixedReal = Union[Real, float]
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import atol, rtol


class PauliString:
    """Represents a Pauli string, a linear combination of Pauli monomials.

    Args:
        monomials : List of Pauli monomials.

    Example:
        >>> I @ Z + 2 * Y @ I + X @ Z
        1*I@Z + 2*Y@I + 1*X@Z
    """

    def __init__(self, monomials: Optional[list["PauliStringMonomial"]] = None):
        self._monomials: list[PauliStringMonomial] = []

        if monomials is not None:
            for mono in monomials:
                if isinstance(mono, PauliStringAtom):
                    mono = PauliStringMonomial(1, [mono])
                self._monomials.append(mono)

        for mono in self._monomials:
            if mono.nb_qubits != self.monomials[0].nb_qubits:
                raise ValueError(
                    f"Non homogeneous sizes for given PauliStrings: {monomials}"
                )

    @property
    def monomials(self) -> list[PauliStringMonomial]:
        """Gets the monomials of the PauliString.

        Returns:
            The list of monomials in the PauliString.
        """
        return self._monomials

    @property
    def nb_qubits(self) -> int:
        """Gets the number of qubits associated with the PauliString.

        Returns:
            The number of qubits associated with the PauliString.
        """
        return 0 if len(self._monomials) == 0 else self._monomials[0].nb_qubits

    def __str__(self):
        return " + ".join(map(str, self._monomials))

    def __repr__(self):
        return str(self)

    def __pos__(self) -> "PauliString":
        return deepcopy(self)

    def __neg__(self) -> "PauliString":
        return -1 * self

    def __iadd__(self, other: "PauliString") -> "PauliString":
        for mono in other.monomials:
            if (
                len(self._monomials) != 0
                and mono.nb_qubits != self._monomials[0].nb_qubits
            ):
                raise ValueError(
                    f"Non homogeneous sizes for given PauliStrings: {(self, other)}"
                )
        self._monomials.extend(deepcopy(other.monomials))
        return self

    def __add__(self, other: "PauliString") -> "PauliString":
        res = deepcopy(self)
        res += other
        return res

    def __isub__(self, other: "PauliString") -> "PauliString":
        self += -1 * other
        return self

    def __sub__(self, other: "PauliString") -> "PauliString":
        return self + (-1) * other

    def __imul__(self, other: FixedReal) -> "PauliString":
        for i, mono in enumerate(self._monomials):
            if isinstance(mono, PauliStringAtom):
                self.monomials[i] = PauliStringMonomial(atoms=[mono])
            self.monomials[i] *= other
        return self

    def __mul__(self, other: FixedReal) -> "PauliString":
        res = deepcopy(self)
        res *= other
        return res

    def __rmul__(self, other: FixedReal) -> "PauliString":
        return self * other

    def __itruediv__(self, other: FixedReal) -> "PauliString":
        self *= 1 / other  # pyright: ignore[reportOperatorIssue]
        return self

    def __truediv__(self, other: FixedReal) -> "PauliString":
        return self * (1 / other)  # pyright: ignore[reportOperatorIssue]

    def __imatmul__(self, other: "PauliString") -> "PauliString":
        self._monomials = [
            mono for s_mono in self.monomials for mono in (s_mono @ other).monomials
        ]
        return self

    def __matmul__(self, other: "PauliString") -> "PauliString":
        res = deepcopy(self)
        res @= other
        return res

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliString):
            return False
        return self.to_dict() == other.to_dict()

    def simplify(self, inplace: bool = False) -> PauliString:
        """Simplifies the PauliString by combining like terms and removing terms
        with zero coefficients.

        Args:
            inplace: Indicates if ``simplify`` should update self.

        Example:
            >>> ps = I @ I - 2 * I @ I + Z @ I - Z @ I
            >>> simplified_ps = ps.simplify()
            >>> print(simplified_ps)
            -1*I@I

        Returns:
            PauliString: A simplified version of the PauliString.

        """
        res = PauliString()
        for unique_mono_atoms in {tuple(mono.atoms) for mono in self.monomials}:
            coef = sum(
                [
                    mono.coef
                    for mono in self.monomials
                    if mono.atoms == list(unique_mono_atoms)
                ]
            )
            if coef != 0:
                res.monomials.append(PauliStringMonomial(coef, list(unique_mono_atoms)))
        if len(res.monomials) == 0:
            res.monomials.append(
                PauliStringMonomial(0, [I for _ in range(self.nb_qubits)])
            )
        if inplace:
            self._monomials = res.monomials
        return res

    def to_matrix(self) -> Matrix:
        """Converts the PauliString to a matrix representation.

        Example:
            >>> ps = I + Z
            >>> matrix_representation = ps.to_matrix()
            >>> print(matrix_representation)
            [[2.+0.j 0.+0.j]
            [0.+0.j 0.+0.j]]

        Returns:
            Matrix representation of the PauliString.
        """
        self = self.simplify()
        return sum(
            map(lambda m: m.to_matrix(), self.monomials),
            start=np.zeros((2**self.nb_qubits, 2**self.nb_qubits), dtype=np.complex64),
        )

    @classmethod
    def from_matrix(cls, matrix: Matrix) -> PauliString:
        """Constructs a PauliString from a matrix.

        Args:
            matrix: Matrix from which the PauliString is generated

        Example:
            >>> ps = PauliString.from_matrix(np.array([[0, 1], [1, 2]]))
            >>> print(ps)
            (1+0j)*I + (1+0j)*X + (-1+0j)*Z

        Returns:
            PauliString: Pauli string decomposition of the matrix in parameter.

        Raises:
            ValueError: If the input matrix is not square or its dimensions are not a power of 2.
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square.")

        num_qubits = int(np.log2(matrix.shape[0]))
        if 2**num_qubits != matrix.shape[0]:
            raise ValueError("Matrix dimensions must be a power of 2.")

        # Return the ordered Pauli basis for the n-qubit Pauli basis.
        pauli_1q = [PauliStringMonomial(1, [atom]) for atom in [I, X, Y, Z]]
        basis = pauli_1q
        for _ in range(num_qubits - 1):
            basis = [p1 @ p2 for p1 in basis for p2 in pauli_1q]

        pauli_list = PauliString()
        for i, mat in enumerate(basis):
            coeff = np.trace(mat.to_matrix().dot(matrix)) / (2**num_qubits)
            if not np.isclose(coeff, 0, atol=atol, rtol=rtol):
                mono = basis[i] * coeff
                pauli_list += mono

        if len(pauli_list.monomials) == 0:
            pauli_list.monomials.append(
                PauliStringMonomial(0, [I for _ in range(num_qubits)])
            )
        return pauli_list

    def to_dict(self) -> dict[str, float]:
        """Converts the PauliString object to a dictionary representation.

        Example:
            >>> ps = 1 * I @ Z + 2 * I @ I
            >>> print(ps.to_dict())
            {'II': 2, 'IZ': 1}

        Returns:
            Dictionary representation of the PauliString object.
        """
        self = self.simplify()
        dict = {}
        for mono in self.monomials:
            atom_str = ""
            for atom in mono.atoms:
                atom_str += str(atom)
            if atom_str not in dict:
                dict[atom_str] = mono.coef
            else:
                dict[atom_str] += mono.coef
        return dict

    def __hash__(self):
        monomials_as_tuples = tuple(
            tuple((atom.label for atom in mono.atoms) for mono in self.monomials)
        )
        return hash(monomials_as_tuples)


class PauliStringMonomial(PauliString):
    """Represents a monomial in a Pauli string, consisting of a coefficient and a list of PauliStringAtom objects.

    Args:
        coef: The coefficient of the monomial.
        atoms: The list of PauliStringAtom objects forming the monomial.
    """

    def __init__(
        self, coef: Real | float = 1, atoms: Optional[list["PauliStringAtom"]] = None
    ):
        self.coef = coef
        self.atoms = [] if atoms is None else atoms

    @property
    def nb_qubits(self) -> int:
        return len(self.atoms)

    @property
    def monomials(self) -> list["PauliStringMonomial"]:
        return [PauliStringMonomial(self.coef, self.atoms)]

    def __str__(self):
        return f"{self.coef}*{'@'.join(map(str,self.atoms))}"

    def __repr__(self):
        return str(self)

    def to_matrix(self) -> Matrix:
        return (
            reduce(
                np.kron,
                map(lambda a: a.to_matrix(), self.atoms),
                np.eye(1, dtype=np.complex64),
            )
            * self.coef
        )

    def __iadd__(self, other: "PauliString"):
        for mono in other.monomials:
            if (
                len(self.monomials) != 0
                and mono.nb_qubits != self.monomials[0].nb_qubits
            ):
                raise ValueError(
                    f"Non homogeneous sizes for given PauliStrings: {(self, other)}"
                )
        res = PauliString([self])
        res.monomials.extend(deepcopy(other.monomials))
        return res

    def __add__(self, other: "PauliString") -> PauliString:
        res = deepcopy(self)
        res += other
        return res

    def __imul__(self, other: FixedReal) -> PauliStringMonomial:
        self.coef *= other
        return self

    def __mul__(self, other: FixedReal) -> PauliStringMonomial:
        res = deepcopy(self)
        res *= other
        return res

    def __itruediv__(self, other: FixedReal) -> PauliStringMonomial:
        self.coef /= other
        return self

    def __truediv__(self, other: FixedReal) -> PauliStringMonomial:
        res = deepcopy(self)
        res /= other
        return res

    def __imatmul__(self, other: PauliString) -> PauliString:
        if isinstance(other, PauliStringAtom):
            self.atoms.append(other)
            return self
        elif isinstance(other, PauliStringMonomial):
            self.coef *= other.coef
            self.atoms.extend(other.atoms)
            return self
        else:
            res = deepcopy(other)
            res._monomials = [
                mono
                for s_mono in self.monomials
                for mono in (other @ s_mono)._monomials
            ]
            return res

    def __matmul__(self, other: PauliString):
        res = deepcopy(self)
        res @= other
        return res

    def simplify(self, inplace: bool = False):
        return deepcopy(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PauliStringMonomial):
            for a1, a2 in zip(self.atoms, other.atoms):
                if a1 != a2:
                    return False
            return self.coef == other.coef
        return super().__eq__(other)

    def __hash__(self):
        atoms_as_tuples = tuple((atom.label for atom in self.atoms))
        return hash(atoms_as_tuples)


class PauliStringAtom(PauliStringMonomial):
    """Represents a single Pauli operator acting on a qubit in a Pauli string.

    Args:
        label: The label representing the Pauli operator (e.g., 'I', 'X', 'Y', 'Z').
        matrix: The matrix representation of the Pauli operator.

    Raises:
        AttributeError: new atoms cannot be created

    Note:
        All the atoms are already initialized. Available atoms are ('I', 'X', 'Y', 'Z').
    """

    __is_mutable = True

    def __init__(self, label: str, matrix: npt.NDArray[np.complex64]):
        if _allow_atom_creation:
            self.label = label
            self.matrix = matrix
            self.__is_mutable = False
        else:
            raise AttributeError(
                "New atoms cannot be created, just use the given I, X, Y and Z"
            )

    @property
    def nb_qubits(self) -> int:
        return 1

    @property
    def atoms(self):
        return [self]

    @property
    def coef(self):
        return 1

    @property
    def monomials(self):
        return [PauliStringMonomial(self.coef, [a for a in self.atoms])]

    def __setattr__(self, name: str, value: Any):
        if self.__is_mutable:
            super().__setattr__(name, value)
        else:
            raise AttributeError("This object is immutable")

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)

    def __truediv__(self, other: FixedReal) -> PauliStringMonomial:
        return PauliStringMonomial(
            1 / other, [self]
        )  # pyright: ignore[reportArgumentType]

    def __imul__(self, other: FixedReal) -> PauliStringMonomial:
        return self * other

    def __mul__(self, other: FixedReal) -> PauliStringMonomial:
        return PauliStringMonomial(other, [self])

    def __rmul__(self, other: FixedReal) -> PauliStringMonomial:
        return PauliStringMonomial(other, [self])

    def __matmul__(self, other: PauliString) -> PauliString:
        res = (
            PauliStringMonomial(1, [other])
            if isinstance(other, PauliStringAtom)
            else deepcopy(other)
        )
        if isinstance(res, PauliStringMonomial):
            res.atoms.insert(0, self)
        else:
            for i, mono in enumerate(res.monomials):
                res.monomials[i] = PauliStringMonomial(mono.coef, mono.atoms)
                res.monomials[i].atoms.insert(0, self)
        return res

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        return self.matrix

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PauliStringAtom):
            return self.label == other.label
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash(self.label)


_allow_atom_creation = True

I = PauliStringAtom("I", np.eye(2, dtype=np.complex64))
r"""Pauli-I atom representing the identity operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}1&0\\0&1\end{pmatrix}`
"""
X = PauliStringAtom("X", 1 - np.eye(2, dtype=np.complex64))
r"""Pauli-X atom representing the X operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}0&1\\1&0\end{pmatrix}`

"""
Y = PauliStringAtom("Y", np.fliplr(np.diag([1j, -1j])))
r"""Pauli-Y atom representing the Y operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}0&-i\\i&0\end{pmatrix}`
"""
Z = PauliStringAtom("Z", np.diag([1, -1]))
r"""Pauli-Z atom representing the Z operator in a Pauli monomial or string.
Matrix representation:
`\begin{pmatrix}1&0\\0&-1\end{pmatrix}`
"""
_allow_atom_creation = False
