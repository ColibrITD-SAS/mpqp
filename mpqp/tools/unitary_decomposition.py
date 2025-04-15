from mpqp.core.circuit import QCircuit
from mpqp.tools import Matrix
import math
from mpqp.gates import CNOT, Ry, Rz
from scipy.linalg import cossin
import numpy as np


def gray_code(n: int):
    """
    Returns the gray code of n.
    """
    return n ^ (n >> 1)


def unitary_SVD(U: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """
    Returns the decomposition of the unitary using eigenvalues decomposition.
    The goal of this SVD is to have following result : U = (I ⊗ V) @ D @ (I ⊗ W).
    D is in the form of a multiplexed Rz operator.

    This function should be used only in the context of the Quantum Shannon Decomposition.

    Args:
        U : The unitary matrix to decompose.

    Returns:
        3 matrices
    """
    length = len(U)
    SU = U[0 : (length // 2)]
    SU2 = U[(length // 2) : length]
    g0 = []
    g1 = []
    for i in range(length // 2):
        g0.append(list(SU[i][0 : length // 2]))
        g1.append(list(SU2[i][length // 2 : length]))

    G0 = np.array(g0)
    G1 = np.array(g1)

    G = G0 @ G1.conj().T

    eigvals, V = np.linalg.eig(G)
    D = np.diag(np.sqrt(eigvals.astype(complex)))
    W = np.asarray(D @ V.conj().T @ G1, dtype=np.complex64)
    D_dagg = D.conj().T
    padding = np.zeros(length // 2)
    D_result = []

    for i in range(length // 2):
        D_result.append(list(D[0 : length // 2][i]) + list(padding))

    for i in range(length // 2):
        D_result.append(list(padding) + list(D_dagg[0 : length // 2][i]))

    return V, np.array(D_result), W


def gray_code_decomposition(
    thetas: Matrix, circuit: QCircuit, position: int, rotation: str
) -> QCircuit:
    """
    Returns the decomposition of a multiplexed Rz or Ry gate.
    The circuit is composed of a succession of CNOTs and rotations according to the original operator.

    Args:
        thetas : A list of floats that are the rotations to be applied on each qubits.
        circuit: The circuit in which the decomposition is stocked.
        position : On which qubit is the rotation is taking place.
        rotation : A string containing either Ry or Rz in the case of the Shannon decomposition.

    Returns:
        The circuit containing the decomposed operator.
    """
    for i in range(len(thetas)):

        angle = sum(
            -angle if bin(gray_code(i) & j).count('1') % 2 else angle
            for j, angle in enumerate(thetas)
        )

        angle *= 2 / len(thetas)

        control_1 = gray_code(i) ^ gray_code(
            i + 1
        )  # CNOT's control is the changed bit of two consecutive natural numbers in gray code
        control = next(i for i in range(len(thetas)) if (control_1 >> i & 1))
        control = max(-position - control - 1, -circuit.nb_qubits + 1)
        if np.abs(angle) > 1e-9:  # Dodge unnecessary rotations
            (
                circuit.add(Ry(angle, position))
                if rotation == "Ry"
                else circuit.add(Rz(angle, position))
            )
        circuit.add(CNOT(control + position, position))
    return circuit


def _decompose(U: Matrix, circuit: QCircuit, position: int = 0) -> QCircuit:
    """
    This function recursively decompose the matrix U into the circuit then returns it.

    For 1 qubit operators it executes a ZYZ decomposition.
    For higher dimensions it does a Quantum Shannon decomposition.
    """
    if len(U) == 2:  # Decompose a 1 qubit operator
        delta = np.angle(np.linalg.det(np.asarray(U, dtype=np.complex64))) / len(U)
        V = U / np.exp(1j * delta)  # extract the global phase so that V is SU
        beta = 2 * math.acos(np.abs(V[0][0]))
        alpha = -np.angle(V[0][0]) - np.angle(V[1][0])
        gamma = -np.angle(V[0][0]) + np.angle(V[1][0])

        circuit.add(Rz(alpha, position))
        circuit.add(Ry(beta, position))
        circuit.add(Rz(gamma, position))
        circuit.gphase += delta
        return circuit
    else:  # 2 qubits or more
        length = len(U)
        (U1, U2), MuxRy, (V1, V2) = cossin(
            U, p=length // 2, q=length // 2, separate=True
        )
        # Reconstruct the
        U12 = np.zeros((len(U1) * 2, len(U1) * 2), dtype=np.complex64)
        for i in range(len(U1)):
            for j in range(len(U1)):
                U12[i][j] = U1[i][j]
                U12[i + len(U12) // 2][j + len(U12) // 2] = U2[i][j]

        V12 = np.zeros((len(V1) * 2, len(V1) * 2), dtype=np.complex64)
        for i in range(len(V1)):
            for j in range(len(V1)):
                V12[i][j] = V1[i][j]
                V12[i + len(V12) // 2][j + len(V12) // 2] = V2[i][j]

        Vu, Du, Wu = unitary_SVD(U12)
        Vv, Dv, Wv = unitary_SVD(V12)

        du = np.angle(Du.diagonal())
        for i in range(len(du) // 2):
            du[i] *= -1

        dv = np.angle(Dv.diagonal())
        for i in range(len(dv) // 2):
            dv[i] *= -1

        circuit = _decompose(Wv, circuit, position + 1)
        circuit = gray_code_decomposition(dv, circuit, position, "Rz")
        circuit = _decompose(Vv, circuit, position + 1)

        circuit = gray_code_decomposition(
            MuxRy,
            circuit,
            position,
            "Ry",
        )

        circuit = _decompose(Wu, circuit, position + 1)
        circuit = gray_code_decomposition(du, circuit, position, "Rz")
        circuit = _decompose(Vu, circuit, position + 1)

        return circuit


def quantum_shannon_decomposition(U: Matrix) -> QCircuit:
    """
    Returns a circuit containing the decomposition of a unitary.
    The resulting circuit is composed of gates CNOT, Ry and Rz.

    The Quantum Shannon Decomposition works by splitting an unitary into 3 matrices using the CSD (cosine sine decomposition).
    Which decompose an unitary matrix into 2 matrices around a multiplexed Ry gate.

    The two other matrices are then decomposed into 2 unitaries surrounding a multiplexed Rz gate.

    This process repeats until the matrices are reduced to 1 qubit gates which can be easily split with the ZYZ decomposition.

    Args:
        U: The unitary matrix to be decomposed

    Returns:
        A quantum circuit containing the decomposition of U.
    """
    circuit = QCircuit(int(np.log2(len(U))))
    return _decompose(U, circuit, 0)
