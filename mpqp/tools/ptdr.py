from copy import deepcopy

import numpy as np
from anytree import NodeMixin


class PauliNode(NodeMixin):
    def __init__(self, name, k: list[int], m: list[int], matrix=None, parent_node=None):
        self.name = name
        self.k = deepcopy(k)
        self.m = deepcopy(m)
        self.matrix = matrix
        self.parent = parent_node

    def update_tree(self):
        l = len(self.k)
        matrix_size = self.matrix.shape[0]

        if self.name == "I":
            self.k = [(idx + 2**l) % matrix_size for idx in self.k]
            self.k = [(idx - 2 ** (l + 1)) % matrix_size for idx in self.k]
            self.m = [val + 1 for val in self.m]
        elif self.name == "Y":
            self.m = [-val for val in self.m]
        elif self.name == "Z":
            self.k = [(idx + 2 ** (l + 1)) % matrix_size for idx in self.k]
            self.m = [val + 2 for val in self.m]

    def compute_coefficient(self) -> float:
        if self.matrix is None:
            raise ValueError("A matrix should be provided to compute coefficients.")

        coeff = 0
        matrix_size = self.matrix.shape[0]

        for j in range(len(self.k)):
            if self.k[j] >= matrix_size:
                raise IndexError(
                    f"Index {self.k[j]} is out of bounds for matrix of size {matrix_size}."
                )

            phase = (-j) ** (self.m[j] % 4)
            coeff += phase * self.matrix[self.k[j], j]

        return coeff.real

    def __repr__(self):
        return f"{self.name} Node: k={self.k}, m={self.m}"


def build_pauli_tree(tree_depth: int, matrix: np.ndarray):
    root_node = PauliNode(name="root", k=[0], m=[0], matrix=matrix)

    def explore_node(current_node, current_depth):
        if current_depth == 0:
            return

        for name in ["I", "X", "Y", "Z"]:
            child_node = PauliNode(
                name=name,
                k=current_node.k,
                m=current_node.m,
                matrix=current_node.matrix,
                parent_node=current_node,
            )
            child_node.update_tree()
            explore_node(child_node, current_depth - 1)

    explore_node(root_node, tree_depth)
    return root_node
