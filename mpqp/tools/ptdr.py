"""
A tree-approach Pauli decomposition algorithm with application to quantum computing
Océane Koska, Marc Baboulin, Arnaud Gazda
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import numpy.typing as npt
from anytree import NodeMixin
from mpqp.core.instruction.measurement.expectation_value import Observable
from mpqp.core.instruction.measurement.pauli_string import (
    I,
    PauliString,
    PauliStringAtom,
    PauliStringMonomial,
    X,
    Y,
    Z,
)
from mpqp.tools import Matrix, is_hermitian, is_power_of_two, rand_hermitian_matrix

paulis = [I, X, Y, Z]


class PauliNode:
    def __init__(self, atom: PauliStringAtom = None, parent: "PauliNode" = None):
        self.pauli = atom
        self.parent: PauliNode = parent
        self.depth = parent.depth + 1 if parent is not None else 0
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


def compute_coefficients(
    k: list[int],
    m: list[int],
    current_node: PauliNode,
    matrix: Matrix,
    monomial_list: list[PauliStringMonomial],
):
    """Algorithm 2: compute the coefficients for this node"""

    m_size = len(matrix)

    current_node.coefficient = (
        sum(
            matrix[k[j], j] * m[j] * (-1j) ** (current_node.nY % 4)
            for j in range(m_size)
        )
        / m_size  # This factor was forgotten in the article
    )

    monomial_list.append(current_node.get_monomial())


def update_tree(current_node: PauliNode, k: list[int], m: list[int]):
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


def generate_and_explore_node(
    k: list[int],
    m: list[int],
    current_node: PauliNode,
    matrix: Matrix,
    n: int,
    monomials: list[PauliStringMonomial],
):
    """Algorithm 4: recursively explore tree, updating nodes"""
    if current_node.depth > 0:
        update_tree(current_node, k, m)

    if current_node.depth < n:

        current_node.children.extend(
            [PauliNode(atom=a, parent=current_node) for a in paulis]
        )

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


############################### DIAGONAL CASE ##########################################


class DiagPauliNode:
    def __init__(self, atom: PauliStringAtom = None, parent: "DiagPauliNode" = None):
        self.pauli = atom
        self.parent: DiagPauliNode = parent
        self.depth = self.parent.depth + 1 if self.parent is not None else 0
        self.children: list[DiagPauliNode] = []
        self.coefficient = None

    @property
    def childI(self):
        return self.children[0]

    @property
    def childZ(self):
        return self.children[1]

    def get_monomial(self):
        atoms = []
        node = self
        while node.parent is not None:
            atoms.append(node.pauli)
            node = node.parent
        return PauliStringMonomial(self.coefficient, atoms)


def compute_coefficients_diagonal_case(
    m: list[bool],
    current_node: DiagPauliNode,
    diag_elements: npt.NDArray[np.float64],
    monomial_list: list[PauliStringMonomial],
):
    """Algorithm 2: compute the coefficients for this node"""

    m_size = len(diag_elements)

    current_node.coefficient = (
        sum(diag_elements[j] * (-1 if m[j] else 1) for j in range(m_size))
        / m_size  # This factor was forgotten in the article
    )

    monomial_list.append(current_node.get_monomial())


def update_tree_diagonal_case(current_node: DiagPauliNode, m: list[bool]):
    """Algorithm 3: updates k and m for the node based on its type"""
    l = current_node.depth - 1
    t_l = 2**l
    if current_node.pauli is I:
        m[t_l : 2 * t_l] = m[0:t_l]

    elif current_node.pauli is Z:
        for i in range(t_l):
            m[i + t_l] = not m[i + t_l]


def generate_and_explore_node_diagonal_case(
    m: list[bool],
    current_node: DiagPauliNode,
    diag_elements: npt.NDArray[np.float64],
    n: int,
    monomials: list[PauliStringMonomial],
):
    """Algorithm 4: recursively explore tree, updating nodes"""
    if current_node.depth > 0:
        update_tree_diagonal_case(current_node, m)

    if current_node.depth < n:

        current_node.children.append(DiagPauliNode(atom=I, parent=current_node))
        current_node.children.append(DiagPauliNode(atom=Z, parent=current_node))

        generate_and_explore_node_diagonal_case(
            m, current_node.childI, diag_elements, n, monomials
        )
        generate_and_explore_node_diagonal_case(
            m, current_node.childZ, diag_elements, n, monomials
        )

    else:
        compute_coefficients_diagonal_case(m, current_node, diag_elements, monomials)


def decompose_diagonal_observable_ptdr(
    diag_elements: list[Real] | npt.NDArray[np.float64],
) -> PauliString:
    """Decompose the observable represented by the hermitian matrix given in parameter into a PauliString.

    Args:
        diag_elements:

    Returns:

    """

    diags = np.array(diag_elements)
    monomials = []
    size = len(diags)

    if not is_power_of_two(size):
        raise ValueError
    # TODO add all the necessary checks on the size
    nb_qubits = int(np.log2(size))
    root = DiagPauliNode()
    i_m = [False] * size
    i_m[0] = False
    generate_and_explore_node_diagonal_case(i_m, root, diags, nb_qubits, monomials)

    return PauliString(monomials)


diag = [-2, 4, 5, 3]
matrix_ex = np.diag(diag)
print(decompose_diagonal_observable_ptdr(diag))
print()
print(decompose_hermitian_matrix_ptdr(matrix_ex))
print()
print(Observable(matrix_ex).pauli_string)
