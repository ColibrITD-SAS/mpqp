"""Functions used for the decomposition of observables in the Pauli basis."""

from __future__ import annotations

from numbers import Real
from typing import Optional

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from typeguard import typechecked

from mpqp.core.instruction.measurement.pauli_string import (
    I,
    PauliString,
    PauliStringAtom,
    PauliStringMonomial,
    X,
    Y,
    Z,
)
from mpqp.tools import Matrix, is_hermitian, is_power_of_two

paulis: list[PauliStringAtom] = [I, X, Y, Z]


@typechecked
class PauliNode:
    """A class represents a node in the Pauli tree used for decomposing a Hermitian matrix into a PauliString."""

    def __init__(
        self,
        atom: Optional[PauliStringAtom] = None,
        parent: Optional["PauliNode"] = None,
    ):
        """Initializes a PauliNode.

        Args:
            atom: The Pauli atom (I, X, Y, Z) associated with this node. Defaults to None.
            parent: The parent node in the decomposition tree. Defaults to None.
        """
        self.pauli = atom
        self.parent: Optional[PauliNode] = parent
        self.depth = parent.depth + 1 if parent is not None else 0
        self.children: list[PauliNode] = []
        self.coefficient: float = 0.0

        if parent is None:
            self.nY = 0
        else:
            self.nY = parent.nY + 1 if atom is Y else parent.nY

    @property
    def childI(self) -> "PauliNode":
        """Returns the child node corresponding to the Pauli-I atom."""
        return self.children[0]

    @property
    def childX(self) -> "PauliNode":
        """Returns the child node corresponding to the Pauli-X atom."""
        return self.children[1]

    @property
    def childY(self) -> "PauliNode":
        """Returns the child node corresponding to the Pauli-Y atom."""
        return self.children[2]

    @property
    def childZ(self) -> "PauliNode":
        """Returns the child node corresponding to the Pauli-Z atom."""
        return self.children[3]

    def get_monomial(self) -> PauliStringMonomial:
        """Constructs and returns the PauliStringMonomial corresponding to the node.

        Returns:
            PauliStringMonomial: The monomial representation of the node.

        """

        atoms = []
        node = self
        while node.parent is not None:
            atoms.append(node.pauli)
            node = node.parent
        return PauliStringMonomial(self.coefficient, atoms)


@typechecked
def compute_coefficients(
    k: list[int],
    m: list[int],
    current_node: PauliNode,
    matrix: Matrix,
    monomial_list: list[PauliStringMonomial],
):
    """Computes coefficients for the current node in the pauli tree based on the given matrix.

    Args:
        k: A list of column indices where the non-zero elements of the matrix are located.
        m: A list of values corresponding to the non-zero coefficients of the matrix.
        current_node: The current node in the Pauli tree.
        matrix: The given Hermitian matrix to be decomposed.
        monomial_list: A list to store the computed monomials.

    """

    m_size = len(matrix)

    current_node.coefficient = (
        sum(
            matrix[j, k[j]] * m[j] * (-1j) ** (current_node.nY % 4)
            for j in range(m_size)
        ).real
        / m_size  # This factor was forgotten in the article
    )

    if current_node.coefficient != 0.0:
        monomial_list.append(current_node.get_monomial())


@typechecked
def update_tree(current_node: PauliNode, k: list[int], m: list[int]):
    """Updates k (indices) and m (values) based on the Pauli type of the current node,
    and computing coefficients.

    Args:
        current_node: The current node in the Pauli tree.
        k: A list of column indices where the non-zero elements of the matrix are located.
        m: A list of values corresponding to the non-zero coefficients of the matrix.

    """
    l = current_node.depth - 1
    t_l = 2**l
    if current_node.pauli is I:
        for i in range(t_l):
            k[i + t_l] = k[i] + t_l
            m[i + t_l] = m[i]

    elif current_node.pauli is X:
        for i in range(t_l):
            k[i + t_l] -= t_l
            k[i] += t_l

    elif current_node.pauli is Y:
        for i in range(t_l, 2 * t_l):
            m[i] *= -1

    elif current_node.pauli is Z:
        for i in range(t_l):
            k[i + t_l] += t_l
            k[i] -= t_l


@typechecked
def generate_and_explore_node(
    k: list[int],
    m: list[int],
    current_node: PauliNode,
    matrix: Matrix,
    n: int,
    monomials: list[PauliStringMonomial],
):
    """Recursively generates and explores nodes in the Pauli tree.

    Args:
        k: A list of column indices where the non-zero elements of the matrix are located.
        m: A list of values corresponding to the non-zero coefficients of the matrix.
        current_node: The current node in the Pauli tree.
        matrix: The given Hermitian matrix to be decomposed.
        monomials: A list to store the computed monomials.

    """
    if current_node.depth > 0:
        update_tree(current_node, k, m)

    if current_node.depth < n:

        current_node.children.extend(
            [PauliNode(atom=a, parent=current_node) for a in paulis]
        )

        for child in current_node.children:
            generate_and_explore_node(k, m, child, matrix, n, monomials)

    else:
        compute_coefficients(k, m, current_node, matrix, monomials)


@typechecked
def decompose_hermitian_matrix_ptdr(matrix: Matrix) -> PauliString:
    """Decompose the observable represented by the hermitian matrix given in parameter into a PauliString.

    A tree-approach Pauli decomposition algorithm with application to quantum computing
    Océane Koska, Marc Baboulin, Arnaud Gazda

    Args:
        matrix: Hermitian matrix representing the observable to be decomposed.

    Returns:
        PauliString: The resulting decomposition as a PauliString representation.

    Raises:
        ValueError: If the matrix is not Hermitian or its dimensions are not a power of 2.

    Reference:
        Oceane Koska, Marc Baboulin & Arnaud Gazda. (2024). A tree-approach Pauli decomposition algorithm with application to quantum computing.
        https://arxiv.org/pdf/2403.11644

    """

    if not is_hermitian(matrix):
        raise ValueError(
            "The matrix in parameter is not hermitian (cannot define an observable)."
        )

    size = len(matrix)

    if not is_power_of_two(size):
        raise ValueError("Matrix dimensions must be a power of 2.")

    monomials = []
    nb_qubits = int(np.log2(size))
    root = PauliNode()
    i_k = [0] * size
    i_m = [0] * size
    i_m[0] = 1
    generate_and_explore_node(i_k, i_m, root, matrix, nb_qubits, monomials)

    return PauliString(monomials)


############################### DIAGONAL CASE PTDR ##########################################


@typechecked
class DiagPauliNode:
    """A class represents a node in the Pauli tree used for decomposing a diagonal observable into a PauliString."""

    def __init__(
        self,
        atom: Optional[PauliStringAtom] = None,
        parent: Optional["DiagPauliNode"] = None,
    ):
        """Initializes a node in the Pauli tree.

        Args:
            atom: The Pauli atom (I, Z) associated with this node. Defaults to None.
            parent: The parent node. Defaults to None.

        """
        self.pauli = atom
        self.parent: Optional[DiagPauliNode] = parent
        self.depth = self.parent.depth + 1 if self.parent is not None else 0
        self.children: list[DiagPauliNode] = []
        self.coefficient: float = 0.0

    @property
    def childI(self):
        """Returns the child node corresponding to the Pauli-I atom."""
        return self.children[0]

    @property
    def childZ(self):
        """Returns the child node corresponding to the Pauli-Z atom."""
        return self.children[1]

    def get_monomial(self) -> PauliStringMonomial:
        """Constructs and returns the PauliStringMonomial corresponding to the node.

        Returns:
            PauliStringMonomial: The monomial representation of the node.

        """
        atoms = []
        node = self
        while node.parent is not None:
            atoms.append(node.pauli)
            node = node.parent
        return PauliStringMonomial(self.coefficient, atoms)


@typechecked
def compute_coefficients_diagonal_case(
    m: list[bool],
    current_node: DiagPauliNode,
    diag_elements: npt.NDArray[np.float64],
    monomial_list: list[PauliStringMonomial],
):
    """Computes coefficients for the current node in the pauli tree based on the diagonal elements.

    Args:
        m: A list of booleans indicating whether the Pauli operator element has a sign flip (True) or not (False).
        current_node: The current node in the Pauli tree.
        diag_elements: The diagonal elements of the observable.
        monomial_list: A list to store the computed monomials.

    """

    m_size = len(diag_elements)

    current_node.coefficient = (
        sum(diag_elements[j] * (-1 if m[j] else 1) for j in range(m_size))
        / m_size  # This factor was forgotten in the article
    )

    monomial_list.append(current_node.get_monomial())


@typechecked
def update_tree_diagonal_case(current_node: DiagPauliNode, m: list[bool]):
    """updates m based on the Pauli type of the current node.

    Args:
       current_node: The current node in the tree.
       m: A list of booleans to be updated representing the sign flip states for each Pauli operator element.

    """
    l = current_node.depth - 1
    t_l = 2**l
    if current_node.pauli is I:
        m[t_l : 2 * t_l] = m[0:t_l]

    elif current_node.pauli is Z:
        for i in range(t_l):
            m[i + t_l] = not m[i + t_l]


@typechecked
def generate_and_explore_node_diagonal_case(
    m: list[bool],
    current_node: DiagPauliNode,
    diag_elements: npt.NDArray[np.float64],
    n: int,
    monomials: list[PauliStringMonomial],
):
    """Recursively explores the Pauli tree and computes the required monomials.

    Args:
       m: A list of booleans representing the current transformation for the Pauli operators (sign flips).
       current_node: The current node in the Pauli tree.
       diag_elements: The diagonal elements of the observable.
       n: The number of qubits.
       monomials: A list to store the computed monomials.

    """
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


@typechecked
def decompose_diagonal_observable_ptdr(
    diag_elements: list[Real] | npt.NDArray[np.float64],
) -> PauliString:
    """Decomposes a diognal observable into a PauliString representation.

    Args:
        diag_elements: The diagonal elements of the observable.

    Returns:
        PauliString: The corresponding PauliString representation.

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
    generate_and_explore_node_diagonal_case(i_m, root, diags, nb_qubits, monomials)

    return PauliString(monomials)


############################### WALSH HADAMARD IDEA ##########################################

# TODO, to optimize


@njit
def numba_hadamard(n: int) -> npt.NDArray[np.int8]:
    """Generates a Hadamard matrix of size n x n using Numba.

    Args:
        n: The size of the Hadamard matrix, must be a power of 2.

    Returns:
        H_matrix: The generated Hadamard matrix.

    Raises:
        ValueError: If n is not a power of 2.

    """
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(np.log2(n))
    if 2**lg2 != n:
        raise ValueError("n must be a power of 2")
    H_matrix = np.array([[1]], dtype=np.int8)
    for _ in range(lg2):
        size = H_matrix.shape[0]
        new_H = np.empty((2 * size, 2 * size), dtype=np.int8)
        for i in prange(size):
            for j in range(size):
                new_H[i, j] = H_matrix[i, j]
                new_H[i + size, j] = H_matrix[i, j]
                new_H[i, j + size] = H_matrix[i, j]
                new_H[i + size, j + size] = -H_matrix[i, j]
        H_matrix = new_H
    return H_matrix


@njit
def compute_coefficients_walsh(
    H_matrix: npt.NDArray[np.int8], diagonal_elements: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Computes the coefficients using the Walsh-Hadamard transform.

    Args:
        H_matrix: The Hadamard matrix.
        diagonal_elements: The diagonal elements of the observable.

    Returns:
        coefs: The computed coefficients.
    """
    coefs = np.zeros(H_matrix.shape[0], dtype=np.float64)
    inv = 1.0 / H_matrix.shape[0]

    for i in prange(H_matrix.shape[0]):
        row_sum = 0.0
        for j in range(H_matrix.shape[1]):
            row_sum += H_matrix[i, j] * diagonal_elements[j]
        coefs[i] = row_sum * inv

    return coefs


@typechecked
def decompose_diagonal_observable_walsh_hadamard(
    diag_elements: list[Real] | npt.NDArray[np.float64],
) -> PauliString:
    """Decomposes the observable represented by the diagonal elements into a PauliString using the Walsh-Hadamard transform.

    Args:
        diag_elements: The diagonal elements of the observable.

    Returns:
        PauliString: The corresponding PauliString representation.

    """
    pauli_1q = [I, Z]
    basis = pauli_1q
    diags = np.array(diag_elements)

    size = len(diags)
    nb_qubits = int(np.log2(size))
    for _ in range(nb_qubits - 1):
        basis = [p1 @ p2 for p1 in basis for p2 in pauli_1q]

    H_matrix = numba_hadamard(size)
    coefs = compute_coefficients_walsh(H_matrix, diags)
    for m, c in zip(basis, coefs):
        m.coef = c

    return PauliString(basis)
