from mpqp.tools import *
from mpqp import QCircuit
from scipy.linalg import cossin
import numpy as np


def gray_code(n: int):
    return n ^ (n >> 1)


def unitary_SVD(U: Matrix):
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
    W = D @ V.conj().T @ G1
    D_dagg = D.conj().T
    padding = np.zeros(length // 2)
    D_result = []

    for i in range(length // 2):
        D_result.append(list(D[0 : length // 2][i]) + list(padding))

    for i in range(length // 2):
        D_result.append(list(padding) + list(D_dagg[0 : length // 2][i]))

    return V, np.array(D_result), W


def gray_code_decomposition(
    thetas: list[float], circuit: QCircuit, position: int, rotation: str
):
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
        control = max(-control - 1, -circuit.nb_qubits + 1)
        if np.abs(angle) > 1e-9:  # Dodge unnecessary rotations
            (
                circuit.add(Ry(angle, position))
                if rotation == "Ry"
                else circuit.add(Rz(angle, position))
            )
        circuit.add(CNOT(control + position, position))
    return circuit


def _decompose(U, circuit: QCircuit, position: int = 0):
    if len(U) == 2:  # Decompose 1 qubit matrices
        delta = np.exp(1j * np.angle(np.linalg.det(U)) / len(U))
        V = U / delta
        beta = 2 * math.acos(np.abs(V[0][0]))
        alpha = -np.angle(V[0][0]) - np.angle(V[1][0])
        gamma = -np.angle(V[0][0]) + np.angle(V[1][0])

        circuit.add(Rz(alpha, position))
        circuit.add(Ry(beta, position))
        circuit.add(Rz(gamma, position))
        if circuit.gphase == 0:
            circuit.gphase = delta
        else:
            circuit.gphase *= delta
        return circuit
    else:  # More than 2
        length = len(U)
        (U1, U2), MuxRy, (V1, V2) = cossin(
            U, p=length // 2, q=length // 2, separate=True
        )
        _, debug, _ = cossin(U, p=length // 2, q=length // 2)
        U12 = np.zeros((len(U1) * 2, len(U1) * 2), dtype=np.complex128)
        for i in range(len(U1)):
            for j in range(len(U1)):
                U12[i][j] = U1[i][j]
                U12[i + len(U12) // 2][j + len(U12) // 2] = U2[i][j]

        V12 = np.zeros((len(V1) * 2, len(V1) * 2), dtype=np.complex128)
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


def decompose(U):
    circuit = QCircuit(int(np.log2(len(U))))
    return _decompose(U, circuit, 0)
