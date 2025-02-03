"""
A tree-approach Pauli decomposition algorithm with application to quantum computing
Océane Koska, Marc Baboulin, Arnaud Gazda
"""

from __future__ import annotations
from numbers import Real

import numpy as np
import numpy.typing as npt
from anytree import NodeMixin, RenderTree
from mpqp.core.instruction import PauliString
from mpqp.tools import Matrix, is_hermitian, rand_hermitian_matrix
from mpqp.core.instruction.measurement.pauli_string import (
    PauliStringAtom,
    I,
    X,
    Y,
    Z,
)

paulis = [I, X, Y, Z]


class PauliNode(NodeMixin):
    def __init__(self, atom: PauliStringAtom = None, parent: "PauliNode" = None):
        self.pauli = atom
        # self.monomial = atom @ parent.monomial
        self.name = atom.label if parent is not None else ""
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


def compute_coefficients(k, m, current_node: PauliNode, matrix):
    """Algorithm 2: compute the coefficients for this node"""

    current_node.coefficient = sum(
        matrix[k[j], j] * m[j] * (-1j) ** (current_node.nY % 4)
        for j in range(len(matrix))
    )


def update_tree(current_node: PauliNode, k, m, matrix):
    """Algorithm 3: updates k and m for the node based on its type"""
    l = current_node.depth - 1
    t_l = 2**l
    t_l_1 = 2 ** (l + 1)
    if current_node.pauli is I:
        for i in range(t_l):
            k[i + t_l] = k[i] + t_l
            m[i + t_l] = m[i]

    elif current_node.pauli is X:
        for i in range(t_l, t_l_1):
            k[i] -= t_l_1

    elif current_node.pauli is Y:
        for i in range(t_l, t_l_1):
            m[i] *= -1

    elif current_node.pauli is Z:
        for i in range(t_l, t_l_1):
            k[i] += t_l_1


def generate_and_explore_node(k, m, current_node: PauliNode, matrix, n):
    """Algorithm 4: recursively explore tree, updating nodes"""
    if current_node.depth > 0:
        update_tree(current_node, k, m, matrix)

    if current_node.depth < n:

        children = [PauliNode(atom=a, parent=current_node) for a in paulis]
        current_node.children = children

        generate_and_explore_node(k, m, current_node.childI, matrix, n)
        generate_and_explore_node(k, m, current_node.childX, matrix, n)
        generate_and_explore_node(k, m, current_node.childY, matrix, n)
        generate_and_explore_node(k, m, current_node.childZ, matrix, n)

    else:
        compute_coefficients(k, m, current_node, matrix)


def decompose_hermitian_matrix_ptdr(matrix: Matrix) -> PauliString:
    """Decompose the observable represented by the hermitian matrix given in parameter into a PauliString.

    Args:
        matrix: Hermitian matrix representing the observable to decompose

    Returns:

    """

    if not is_hermitian(matrix):
        raise ValueError(
            "The matrix in parameter is not hermitian (cannot define an observable)"
        )

    ...
    # TODO plug the PTDR algorithm here


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


num_qubits = 1
matrix_size = 2**num_qubits

# matrix_ex = rand_hermitian_matrix(matrix_size)
matrix_ex = np.array([[2, 3], [3, 1]])  # SHOULD GIVE 1.5*I + 3*X + 0.5*Z

# FIXME: The arrays k and m are not initialized like it is described in the paper. They are not initialized with
#  zero array.
initial_k = [0] * matrix_size
initial_m = [0] * matrix_size

initial_m[0] = 1

root = PauliNode()

generate_and_explore_node(initial_k, initial_m, root, matrix_ex, num_qubits)

print("\nPTDR:")
for pre, _, node in RenderTree(root):
    print(f"{pre}{node.name} (coeff: {node.coefficient})")
