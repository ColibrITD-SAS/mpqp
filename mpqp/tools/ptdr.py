"""
A tree-approach Pauli decomposition algorithm with application to quantum computing
Océane Koska, Marc Baboulin, Arnaud Gazda
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import numpy.typing as npt
from anytree import NodeMixin, RenderTree

from mpqp.core.instruction import Observable
from mpqp.core.instruction.measurement.pauli_string import (
    I,
    PauliString,
    PauliStringAtom,
    X,
    Y,
    Z,
    PauliStringMonomial,
)
from mpqp.tools import Matrix, is_hermitian, rand_hermitian_matrix

paulis = [I, X, Y, Z]


class PauliNode(NodeMixin):
    def __init__(self, atom: PauliStringAtom = None, parent: "PauliNode" = None):
        self.pauli = atom
        self.parent: PauliNode = parent
        self.children: list[PauliNode] = []
        self.coefficient = None

        if parent is None:
            self.nY = 0
        else:
            self.nY = parent.nY + 1 if atom is Y else parent.nY

    @property
    def childI(self):
        return self.children[0]

    @property
    def childX(self):
        return self.children[1]

    @property
    def childY(self):
        return self.children[2]

    @property
    def childZ(self):
        return self.children[3]

    def get_monomial(self):
        atoms = []
        node = self
        while node.parent is not None:
            atoms.append(node.pauli)
            node = node.parent
        return PauliStringMonomial(self.coefficient, atoms)


def compute_coefficients(k, m, current_node: PauliNode, matrix, monomial_list):
    """Algorithm 2: compute the coefficients for this node"""

    m_size = len(matrix)

    current_node.coefficient = (
        sum(
            matrix[k[j], j] * m[j] * (-1j) ** (current_node.nY % 4)
            for j in range(len(matrix))
        )
        / m_size  # This factor was forgotten in the article
    )

    monomial_list.append(current_node.get_monomial())


def update_tree(current_node: PauliNode, k, m):
    """Algorithm 3: updates k and m for the node based on its type"""
    l = current_node.depth - 1
    t_l = 2**l
    t_l_1 = 2 ** (l + 1)
    if current_node.pauli is I:
        for i in range(t_l):
            k[i + t_l] = k[i] + t_l
            m[i + t_l] = m[i]

    elif current_node.pauli is X:
        for i in range(t_l):
            k[i + t_l] -= t_l
            k[i] += t_l

    elif current_node.pauli is Y:
        for i in range(t_l, t_l_1):
            m[i] *= -1

    elif current_node.pauli is Z:
        for i in range(t_l):
            k[i + t_l] += t_l
            k[i] -= t_l


def generate_and_explore_node(k, m, current_node: PauliNode, matrix, n, monomials):
    """Algorithm 4: recursively explore tree, updating nodes"""
    if current_node.depth > 0:
        update_tree(current_node, k, m)

    if current_node.depth < n:

        children = [PauliNode(atom=a, parent=current_node) for a in paulis]
        current_node.children = children

        generate_and_explore_node(k, m, current_node.childI, matrix, n, monomials)
        generate_and_explore_node(k, m, current_node.childX, matrix, n, monomials)
        generate_and_explore_node(k, m, current_node.childY, matrix, n, monomials)
        generate_and_explore_node(k, m, current_node.childZ, matrix, n, monomials)

    else:
        compute_coefficients(k, m, current_node, matrix, monomials)


def decompose_hermitian_matrix_ptdr(matrix: Matrix) -> PauliString:
    """Decompose the observable represented by the hermitian matrix given in parameter into a PauliString.

    Args:
        matrix: Hermitian matrix representing the observable to decompose

    Returns:

    """

    if not is_hermitian(matrix):
        raise ValueError(
            "The matrix in parameter is not hermitian (cannot define an observable)."
        )

    monomials = []
    size = len(matrix)
    # TODO add all the necessary checks on the size
    nb_qubits = int(np.log2(size))
    root = PauliNode()
    i_k = [0] * size
    i_m = [0] * size
    i_m[0] = 1
    generate_and_explore_node(i_k, i_m, root, matrix, nb_qubits, monomials)

    return PauliString(monomials)


def decompose_diagonal_observable_ptdr(
    diag_elements: list[Real] | npt.NDArray[np.complex64],
) -> PauliString:
    """Decompose the observable represented by the hermitian matrix given in parameter into a PauliString.

    Args:
        matrix: Hermitian matrix representing the observable to decompose

    Returns:

    """

    ...
    # TODO plug the PTDR algorithm adapted to diagonal case, or Youcef trick to decompose


matrix_ex = rand_hermitian_matrix(2**7)
# matrix_ex = np.array([[2, 3], [3, 1]]) # SHOULD GIVE 1.5*I + 3*X + 0.5*Z
# matrix_ex = np.array(
#     [[-1, 1 - 1j], [1 + 1j, 0]]
# )  # SHOULDtrix_ex = np.array(
#     [[-1, 1 - 1j], [1 + 1j, 0]]
# )  # SHOULD GIVE -0.5 * I + X - Y - 0.5 * Z
# matrix_ex = np.diag([-2, 4, 5, 3])  # 2.5*II - IZ - 1.5*ZI + 2*ZZ


# initial_k = [0] * matrix_size
# initial_m = [0] * matrix_size
#
# initial_m[0] = 1
#
# root = PauliNode()
#
# generate_and_explore_node(initial_k, initial_m, root, matrix_ex, num_qubits)
#
# print("\nPTDR:")

print(decompose_hermitian_matrix_ptdr(matrix_ex))
print()
print(Observable(matrix_ex).pauli_string)
