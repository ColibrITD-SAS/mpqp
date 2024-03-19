from typing import Optional, Any
from numbers import Real
import numpy.typing as npt
import numpy as np
from copy import deepcopy


class PauliString:
    def __init__(self, monomials: Optional[list["PauliStringMonomial"]] = None):
        self.monomials = [] if monomials is None else monomials
        self.nb_qubits = 0 if monomials is None else len(self.monomials[0].atoms)

    def __str__(self):
        return " + ".join(map(str, self.monomials))

    def __repr__(self):
        return str(self)

    def __iadd__(self, other: "PauliString"):
        # TODO: test for homogeneity
        if isinstance(other, PauliStringAtom):
            self.monomials.append(PauliStringMonomial(1, other))
        elif isinstance(other, PauliStringMonomial):
            self.monomials.append(deepcopy(other))
        else:
            self.monomials.extend(deepcopy(other.monomials))
        return self

    def __add__(self, other: "PauliString"):
        res = deepcopy(self)
        res += other
        return res

    def __imul__(self, other: Real):
        for mono in self.monomials:
            mono *= other
        return self

    def __mul__(self, other: Real):
        res = deepcopy(self)
        res *= other
        return res

    def __imatmul__(self, other: "PauliString"):
        for mono in self.monomials:
            coef @= other
        return self

    def __matmul__(self, other: "PauliString"):
        res = deepcopy(self)
        res @= other
        return res

    def simplify(self):
        # 3M-TODO
        res = PauliString()
        for unique_mono in set(self.monomials):
            coefs = [mono.coef for mono in self.monomials if mono == unique_mono]
            res.append(PauliStringMonomial(sum(coefs), deepcopy(unique_mono)))
        return res

    def to_other_language(self):
        pass

    def to_matrix(self):
        if len(self.monomials) == 0:
            return np.zeros((2**self.nb_qubits, 2**self.nb_qubits))

        return sum(monomial.to_matrix() for monomial in self.monomials)


class PauliStringMonomial(PauliString):
    def __init__(self, coef: Real = 1, atoms: Optional[list["PauliStringAtom"]] = None):
        self.coef = coef
        self.atoms = [] if atoms is None else atoms

    def __str__(self):
        return f"{self.coef}*{'@'.join(map(str,self.atoms))}"

    def __repr__(self):
        return str(self)

    def __iadd__(self, other: "PauliString"):
        # TODO: test for homogeneity
        res = PauliString([self])
        if isinstance(other, PauliStringAtom):
            res.monomials.append(PauliStringMonomial(1, [other]))
        elif isinstance(other, PauliStringMonomial):
            res.monomials.append(deepcopy(other))
        else:
            res.monomials.extend(deepcopy(other.monomials))
        return res

    def __add__(self, other: "PauliString"):
        res = deepcopy(self)
        res += other
        return res

    def __imul__(self, other: Real):
        self.coef *= other
        return self

    def __mul__(self, other: Real):
        res = deepcopy(self)
        res *= other
        return res

    def __imatmul__(self, other: PauliString):
        if isinstance(other, PauliStringAtom):
            self.atoms.append(other)
            return self
        elif isinstance(other, PauliStringMonomial):
            self.coef *= other.coef
            self.atoms.extend(other.atoms)
            return self
        else:
            res = deepcopy(other)
            for mono in res:
                mono = self @ mono
            return res

    def __matmul__(self, other: PauliString):
        res = deepcopy(self)
        res @= other
        return res

    def to_matrix(self):
        matrix = self.coef
        for atom in self.atoms:
            matrix = np.kron(matrix, atom.to_matrix())
        return matrix


ALLOW_ATOM_CREATION = True


class PauliStringAtom(PauliStringMonomial):
    __is_mutable = True

    def __init__(self, label: str, matrix: npt.NDArray[np.complex64]):
        if ALLOW_ATOM_CREATION:
            self.label = label
            self.matrix = matrix
            self.__is_mutable = False
        else:
            raise AttributeError(
                "New atoms cannot be created, just use the given I, X, Y and Z"
            )

    def __setattr__(self, name: str, value: Any):
        if self.__is_mutable:
            super().__setattr__(name, value)
        else:
            raise AttributeError("This object is immutable")

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)

    def __mul__(self, other: Real):
        return PauliStringMonomial(other, [self])

    def __rmul__(self, other: Real):
        return PauliStringMonomial(other, [self])

    def __matmul__(self, other: "PauliStringMonomial"):
        res = (
            PauliStringMonomial(1, [other])
            if isinstance(other, PauliStringAtom)
            else deepcopy(other)
        )
        res.atoms.insert(0, self)
        return res

    def to_matrix(self):
        return self.matrix


I = PauliStringAtom("I", np.eye(2))
X = PauliStringAtom("X", 1 - np.eye(2))
Y = PauliStringAtom("Y", np.diag([1j, -1j]))
Z = PauliStringAtom("Z", np.diag([1, -1]))

ALLOW_ATOM_CREATION = False
