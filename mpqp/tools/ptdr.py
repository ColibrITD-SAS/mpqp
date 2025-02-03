from numbers import Real

import numpy as np
import numpy.typing as npt
from anytree import NodeMixin, RenderTree
from mpqp.core.instruction import PauliString
from mpqp.tools import Matrix, is_hermitian

# FIXME: I would use the PauliAtoms defined in pauli_string.py
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

pauli_matrices = [I, X, Y, Z]


# FIXME: I would avoid passing k and m as an argument of the constructor, especially doing a copy, since all nodes
#  are supposed to share the same arrays k and m.
# FIXME: I wouldn't put the coefficient as an attribute for each Node, since we only need it for the leaf.
class PauliNode(NodeMixin):
    def __init__(self, name, k, m, matrix, depth, parent=None):
        self.name = name
        self.k = k.copy()
        self.m = m.copy()
        self.matrix = matrix
        self.depth = depth
        self.coefficient = None
        self.parent = parent

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value

    # FIXME: you compute the coefficient by computing the trace, which make the use of the whole algorithm useless.
    #  You should use the formula and algorithm given in the article.
    # FIXME: I would avoid putting this as a method, but rather an external function that we call on a Node, but this
    #  can be discussed.
    def compute_coefficients(self):
        """Algorithm 2: compute the coefficients for this node"""
        coeff = 0
        n = len(self.k)

        pauli_operator = np.eye(1)
        for i in range(n):

            pauli_matrix = pauli_matrices[self.k[i]]
            pauli_operator = np.kron(pauli_operator, pauli_matrix)

        phase = (-1j) ** (sum(self.m) % 4)  # phase for the current node
        coeff = phase * np.trace(np.dot(self.matrix, pauli_operator))

        self.coefficient = coeff.real

        return self.coefficient


def update_tree(node):
    """Algorithm 3: updates k and m for the node based on its type"""
    if node.depth == 0:
        node.compute_coefficients()
        return

    l = node.depth - 1

    if node.name == "I":
        node.k[l : l + 2] = [node.k[l] + 2**l] * 2
        node.m[l : l + 2] = [node.m[l]] * 2

    elif node.name == "X":
        node.k[l : l + 2] = [node.k[l] - 2**l] * 2

    elif node.name == "Y":
        node.m[l : l + 2] = [-node.m[l]] * 2

    elif node.name == "Z":
        node.k[l : l + 2] = [node.k[l] + 2**l] * 2


def explore_node(node, tree_depth):
    """Algorithm 4: recursively explore tree, updating nodes"""
    update_tree(node)

    if node.depth > 0:
        for pauli_type in ["I", "X", "Y", "Z"]:
            # FIXME: I wouldn't copy the arrays k and m, also because you already copy them in the constructor
            #  (see comment above).
            child = PauliNode(
                name=pauli_type,
                k=node.k.copy(),
                m=node.m.copy(),
                matrix=node.matrix,
                depth=node.depth - 1,
                parent=node,
            )
            explore_node(child, tree_depth)


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


num_qubits = 3
matrix_size = 2**num_qubits

# FIXME: In our case, we don't want to implement the decomposition of a general matrix in the Pauli basis
#  (even if this is possible, but with complex coefficients). We want to focus on decomposition of observables, which
#  are represented by Hermitian matrices. You can use the function mpqp.tools.maths.rand_hermitian_matrix().
matrix_ex = np.random.rand(matrix_size, matrix_size)

# FIXME: The size of k and m are not good. They should match the size of the matrix because they represent efficient
#  way of representing elements in the spare matrix of Pauli.
# FIXME: The arrays k and m are not initialized like it is described in the paper. They are not initialized with
#  zero array.
initial_k = [0] * num_qubits
initial_m = [0] * num_qubits

root = PauliNode(
    name="root", k=initial_k, m=initial_m, matrix=matrix_ex, depth=num_qubits
)

explore_node(root, num_qubits)

print("\nPTDR:")
for pre, _, node in RenderTree(root):
    print(f"{pre}{node.name} (coeff: {node.coefficient})")
