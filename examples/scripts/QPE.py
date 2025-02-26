import numpy as np

from mpqp import Barrier, QCircuit
from mpqp.execution import IBMDevice, Result, run
from mpqp.gates import *
from mpqp.measures import BasisMeasure

# U_op = rand_unitary_2x2_matrix()
U_gate = Ry(np.pi / 4, 0)
i2 = np.eye(2, dtype=np.complex64)

n = 5


def c_Uk(unitary: Gate, phase_precision: int) -> CustomGate:
    total_size = phase_precision + unitary.nb_qubits
    matrix = np.zeros((2**total_size, 2**total_size), dtype=np.complex64)

    N_phase = 2**phase_precision
    for k in range(N_phase):
        k_vec = np.zeros((N_phase, N_phase))
        k_vec[k, k] = 1
        matrix += np.kron(k_vec, unitary.power(k).to_matrix())
    return CustomGate(UnitaryMatrix(matrix), list(range(total_size)))


qft = QCircuit(
    sum(
        [
            [H(i)] + [CRk(j + 1, i + j, i) for j in range(1, n - i)] + [Barrier()]
            for i in range(n)
        ],
        start=[],
    )
)

precision = 3

qpe = (
    QCircuit(H.range(n) + [c_Uk(U_gate, precision)])
    + qft.inverse()
    + QCircuit([BasisMeasure(list(range(precision)))])
)

result = run(qpe, IBMDevice.AER_SIMULATOR)
assert isinstance(result, Result)

probabilities = list(result.probabilities)
most_likely_state = probabilities.index(max(probabilities))

print(int(format(most_likely_state, '0{}b'.format(n))[::-1], 2) / (2**n) * 2 * np.pi)
