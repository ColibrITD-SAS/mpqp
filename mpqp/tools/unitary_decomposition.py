"""This module regroups functions used for decomposition of arbitrary
unitary operator into elementary gates regrouped in a quantum circuit."""

import math
from typing import Union

import numpy as np
from mpqp.core.circuit import QCircuit
from mpqp.gates import CNOT, Ry, Rz
from mpqp.tools import Matrix
from mpqp.tools.maths import is_power_of_two
from scipy.linalg import cossin

PRECISION = 1e-9


def _gray_code(n: int):
    return n ^ (n >> 1)


def _unitary_SVD(U: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    r"""
    Returns the decomposition of the unitary using eigenvalues decomposition.
    This decomposition is used to decompose matrices that are the result of a cosine-sine decomposition.

    Hence we have $$U = \begin{pmatrix}U_0 & 0\\ 0 & U_1 \end{pmatrix}$$ with $U_0$ and $U_1$ being unitary matrices themselves.
    The goal of this SVD is to have following result : $$U = (I \otimes V) \times D \times (I \otimes W)$$.
    D is in the form of a multiplexed Rz operator.
    The whole equation should be in the form :
    `U = \begin{pmatrix}V & 0 \\ 0 & V end{pmatrix} \times \begin{pmatrix}D & 0 \\ 0 & D^{\dagger} \end{pmatrix} \times \begin{pmatrix}W & 0 \\ 0 & W \end{pmatrix}`

    In the context of the quantum Shannon decomposition U is the result of a cosine sine decomposition, hence U is always in the form : `U =  \begin{pmatrix}U_0 & 0\\ 0 & U_1 \end{pmatrix}`

    Args:
        U : The unitary matrix to decompose.

    Returns:
        A tuple of 3 matrices (V,D,W) so that U = $$(I \otimes V) \times D \times (I \otimes W)$$

    Reference:
    .. [1] V. Shende, S. S. Bullock, and I. Markov, “Synthesis of quantum logic
        circuits,” Computer-Aided Design of Integrated Circuits and Systems,
        IEEE Transactions on, vol. 25, pp. 1000 – 1010, July 2006.
    """

    # Decompose U into it's 2 submatrices G0 and G1
    length = len(U)

    # U is a block diagonal matrix. Here we the two blocks in G0 and G1
    SU = U[0 : (length // 2)]
    SU2 = U[(length // 2) : length]
    g0 = []
    g1 = []
    for i in range(length // 2):
        g0.append(list(SU[i][0 : length // 2]))
        g1.append(list(SU2[i][length // 2 : length]))

    G0 = np.array(g0)
    G1 = np.array(g1)

    # Build G as G = V @ D² @ V†
    G = G0 @ G1.conj().T

    eigvals, V = np.linalg.eig(G)
    D = np.diag(np.sqrt(eigvals.astype(complex)))

    # W = D @ V† @ G1
    W = np.asarray(D @ V.conj().T @ G1, dtype=np.complex128)
    D_dagg = D.conj().T

    # Reconstruct the whole D matrix
    padding = np.zeros(length // 2)
    D_result = []

    for i in range(length // 2):
        D_result.append(list(D[0 : length // 2][i]) + list(padding))

    for i in range(length // 2):
        D_result.append(list(padding) + list(D_dagg[0 : length // 2][i]))

    return V, np.array(D_result), W


def _gray_code_decomposition(
    thetas: Matrix,
    circuit: QCircuit,
    position: int,
    rotation: Union[type[Rz], type[Ry]],
) -> QCircuit:
    """
    Returns the decomposition of a multiplexed Rz or Ry gate.
    The circuit is composed of a succession of CNOTs and rotations according to the original operator.

    This function is originally from cirq's implementation.
    Args:
        thetas : A list of floats that are the rotations to be applied on each qubits.
        circuit: The circuit in which the decomposition is stocked.
        position : On which qubit is the rotation is taking place.
        rotation : Either a Ry or a Rz rotation.

    Returns:
        The circuit containing the decomposed operator.

    Reference:
    .. [1] Mikko Möttönen, Juha J. Vartiainen, Ville Bergholm, and Martti M. Salomaa. 2004. Quantum circuits for general multi-qubit gates. American Physical Society (APS) : 93-13.
    """
    for i in range(len(thetas)):

        angle = sum(
            -angle if bin(_gray_code(i) & j).count('1') % 2 else angle
            for j, angle in enumerate(thetas)
        )

        angle *= 2 / len(thetas)
        # CNOT's control is the changed bit of two consecutive numbers in gray code
        changed = _gray_code(i) ^ _gray_code(i + 1)
        control = next(i for i in range(len(thetas)) if (changed >> i & 1))
        control = max(-control - position - 1 + circuit.nb_qubits, 1)
        if np.abs(angle) > PRECISION:  # Dodge unnecessary rotations
            circuit.add(rotation(angle, position))
        circuit.add(CNOT(control + position, position))
    return circuit


def _decompose(U: Matrix, circuit: QCircuit, position: int = 0) -> QCircuit:
    """
    This function recursively decompose the matrix U into the circuit then returns it.

    For 1 qubit operators it executes a ZYZ decomposition.
    For higher dimensions it does a Quantum Shannon decomposition.

    Reference:
    .. [1] Mikko Möttönen, Juha J. Vartiainen, Ville Bergholm, and Martti M. Salomaa. 2004. Quantum circuits for general multi-qubit gates. American Physical Society (APS) : 93-13.
    """
    if len(U) == 2:  # Decompose a 1 qubit operator
        delta = np.angle(np.linalg.det(np.asarray(U, dtype=np.complex128))) / len(U)

        # extract the global phase so that SU is a special unitary, note that if U is a special unitary then delta = 0
        SU = U / np.exp(1j * delta)
        beta = 2 * math.acos(np.abs(SU[0][0]))
        alpha = -np.angle(SU[0][0]) - np.angle(SU[1][0])
        gamma = -np.angle(SU[0][0]) + np.angle(SU[1][0])

        circuit.add(Rz(alpha, position))
        circuit.add(Ry(beta, position))
        circuit.add(Rz(gamma, position))
        circuit.gphase += delta  # Stores the gphase in the circuit
        return circuit
    else:  # 2 qubits or more
        length = len(U)
        U12, MuxRy, V12 = cossin(U, p=length // 2, q=length // 2, separate=False)

        # Extracts the rotations of the multiplexed Ry for later decomposition
        thetas = []
        for i in range(MuxRy.shape[0] // 2):
            thetas.append(np.arccos(MuxRy[i][i]))
        thetas = np.array(thetas)

        assert isinstance(U12, np.ndarray)
        assert isinstance(V12, np.ndarray)
        Vu, MuxRzu, Wu = _unitary_SVD(U12)
        Vv, MuxRzv, Wv = _unitary_SVD(V12)

        # Extracts the rotations of both multiplexed Rz for later decomposition
        du = np.angle(MuxRzu.diagonal())
        for i in range(len(du) // 2):
            du[i] *= -1

        dv = np.angle(MuxRzv.diagonal())
        for i in range(len(dv) // 2):
            dv[i] *= -1

        # Now recursively decompose every obtained matrices.
        circuit = _decompose(Wv, circuit, position + 1)
        circuit = _gray_code_decomposition(dv, circuit, position, Rz)
        circuit = _decompose(Vv, circuit, position + 1)

        circuit = _gray_code_decomposition(
            thetas,
            circuit,
            position,
            Ry,
        )

        circuit = _decompose(Wu, circuit, position + 1)
        circuit = _gray_code_decomposition(du, circuit, position, Rz)
        circuit = _decompose(Vu, circuit, position + 1)

        return circuit


def _optimize_circuit(circuit: QCircuit) -> QCircuit:
    """
    Optimize the circuit of the decomposition by removing useless CNOT gates.
    """
    length = len(circuit.instructions)
    i = 0
    while i < length - 2:
        if isinstance(circuit.instructions[i], CNOT):
            j = i + 1
            while isinstance(circuit.instructions[j], CNOT):
                if circuit.instructions[i] == circuit.instructions[j]:
                    circuit.instructions.pop(j)
                    circuit.instructions.pop(i)
                    length -= 2
                    i -= 1
                    break
                j += 1
        i += 1

    return circuit


def quantum_shannon_decomposition(U: Matrix) -> QCircuit:
    """
    Returns a circuit containing the decomposition of a unitary.
    The resulting circuit is composed of gates CNOT, Ry and Rz.

    The Quantum Shannon Decomposition works by splitting an unitary into 3 matrices using the cosine sine decomposition.
    Which decompose an unitary matrix into 2 n-sized matrices around a multiplexed Ry gate.

    The two other matrices are then decomposed into 2 n-1 sized unitaries surrounding a multiplexed Rz gate.

    This process repeats until the matrices are reduced to 1 qubit gates which can be easily split with the ZYZ decomposition.

    Args:
        U: The unitary matrix to be decomposed

    Returns:
        A quantum circuit equivalent to U.

    References:
    .. [1] Mikko Möttönen, Juha J. Vartiainen, Ville Bergholm, and Martti M. Salomaa. 2004. Quantum circuits for general multi-qubit gates. American Physical Society (APS) : 93-13.

    Examples:
        >>> U = np.array([[1,0],[0,1]])
        >>> circuit = quantum_shannon_decomposition(U)
        >>> print(matrix_eq(U, circuit.to_matrix()))
        True
    """
    if not is_power_of_two(len(U)):
        raise ValueError(
            f"The size of the unitary matrix should be of the form 2**n : got {len(U)}"
        )
    circuit = QCircuit(int(np.log2(len(U))))
    circuit = _decompose(U, circuit, 0)
    return _optimize_circuit(circuit)
