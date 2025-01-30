import numpy as np
from anytree import NodeMixin, RenderTree

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

pauli_matrices = [I, X, Y, Z]


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
            child = PauliNode(
                name=pauli_type,
                k=node.k.copy(),
                m=node.m.copy(),
                matrix=node.matrix,
                depth=node.depth - 1,
                parent=node,
            )
            explore_node(child, tree_depth)


num_qubits = 3
matrix_size = 2**num_qubits

matrix_ex = np.random.rand(matrix_size, matrix_size)

initial_k = [0] * num_qubits
initial_m = [0] * num_qubits

root = PauliNode(
    name="root", k=initial_k, m=initial_m, matrix=matrix_ex, depth=num_qubits
)

explore_node(root, num_qubits)

print("\nPTDR:")
for pre, _, node in RenderTree(root):
    print(f"{pre}{node.name} (coeff: {node.coefficient})")
