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
    def __init__(self, monomials: Optional[list["PauliStringMonomial"]] = None):
        self._monomials: list[PauliStringMonomial] = (
            [] if monomials is None else monomials
        )
        for mono in self._monomials:
            if mono.nb_qubits != self.monomials[0].nb_qubits:
                raise ValueError(
                    f"Non homogeneous sizes for given PauliStrings: {monomials}"
                )

    @property
    def monomials(self) -> list[PauliStringMonomial]:
        return self._monomials

    @property
    def nb_qubits(self):
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
        self *= 1 / other
        return self

    def __truediv__(self, other: FixedReal) -> "PauliString":
        return self * (1 / other)

    def __imatmul__(self, other: "PauliString") -> "PauliString":
        self._monomials = [
            mono for s_mono in self.monomials for mono in (s_mono @ other).monomials
        ]
        return self

    def __matmul__(self, other: "PauliString") -> "PauliString":
        res = deepcopy(self)
        res @= other
        return res

    def simplify(self):
        res = PauliString()
        for unique_mono in set(self._monomials):
            coef = sum([mono.coef for mono in self._monomials if mono == unique_mono])
            if coef != 0:
                res._monomials.append(
                    PauliStringMonomial(coef, deepcopy(unique_mono).atoms)
                )
        return res

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        return sum(
            map(lambda m: m.to_matrix(), self._monomials),
            start=np.zeros((2**self.nb_qubits, 2**self.nb_qubits), dtype=np.complex64),
        )

    @classmethod
    def from_matrix(matrix: Matrix) -> PauliString:
        """Construct a PauliString from a matrix.

        Args:
            matrix (npt.NDArray[np.complex64]): Matrix.

        Returns:
            PauliString: form class PauliString.

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
        if num_qubits == 1:
            return pauli_1q
        basis = pauli_1q
        for _ in range(num_qubits - 1):
            basis = [p1 @ p2 for p1 in basis for p2 in pauli_1q]

        pauli_list = PauliString()
        for i, mat in enumerate(basis):
            coeff = np.trace(mat.to_matrix().dot(matrix)) / (2**num_qubits)
            if not np.isclose(coeff, 0, atol=atol, rtol=rtol):
                mono = basis[i] * coeff
                pauli_list += mono

        return pauli_list


class PauliStringMonomial(PauliString):
    def __init__(
        self, coef: Real | float = 1, atoms: Optional[list["PauliStringAtom"]] = None
    ):
        self.coef = coef
        self.atoms = [] if atoms is None else atoms

    @property
    def nb_qubits(self):
        return len(self.atoms)

    @property
    def monomials(self):
        return [PauliStringMonomial(self.coef, self.atoms)]

    def __str__(self):
        return f"{self.coef}*{'@'.join(map(str,self.atoms))}"

    def __repr__(self):
        return str(self)

    def to_matrix(self) -> npt.NDArray[np.complex64]:
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
                for s_mono in self._monomials
                for mono in (s_mono @ other)._monomials
            ]
            return res

    def __matmul__(self, other: PauliString):
        res = deepcopy(self)
        res @= other
        return res


class PauliStringAtom(PauliStringMonomial):
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
    def nb_qubits(self):
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
        return PauliStringMonomial(1 / other, [self])

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


_allow_atom_creation = True

I = PauliStringAtom("I", np.eye(2, dtype=np.complex64))
X = PauliStringAtom("X", 1 - np.eye(2, dtype=np.complex64))
Y = PauliStringAtom("Y", np.fliplr(np.diag([1j, -1j])))
Z = PauliStringAtom("Z", np.diag([1, -1]))

_allow_atom_creation = False
