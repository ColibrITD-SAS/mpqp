"""This script runs the QPE for the gate defined in ``U_gate``. The QPE
estimates the phase of the gate, up to a precision given by the variable
``precision``.

This script come in complement with a notebook also tackling Grover's
algorithm, but it is done differently in order to highlight different ways to
use MPQP."""

import numpy as np

from mpqp import Barrier, QCircuit
from mpqp.execution import IBMDevice, Result, run
from mpqp.gates import *
from mpqp.measures import BasisMeasure

U_gate = Ry(np.pi / 4, 0)
precision = 3
n = U_gate.nb_qubits + precision

i2 = np.eye(2, dtype=np.complex64)


def c_Uk(unitary: Gate, phase_precision: int) -> CustomGate:
    """Single gate representing the cascade of controlled U gates to the power
    of k used in the QPE."""
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
