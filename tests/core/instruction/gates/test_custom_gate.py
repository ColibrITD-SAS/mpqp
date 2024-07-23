import random

import pytest

import numpy as np

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.tools.maths import matrix_eq, rand_orthogonal_matrix, is_unitary
from mpqp.execution import run, ATOSDevice


# @pytest.mark.parametrize(
#     "circuit, state_vector",
#     [
#         (
#             QCircuit([CustomGate(UnitaryMatrix())]),
#             np.array()
#         )
#     ],
# )
# def test_custom_gate_state_vector(circuit: QCircuit, state_vector: np.ndarray):
#     assert matrix_eq(run(circuit, ATOSDevice.MYQLM_PYLINALG).state_vector.amplitudes, state_vector)

def test_custom_gate_is_unitary():
    definition = UnitaryMatrix(np.array([[1, 0], [0, 1j]]))
    assert is_unitary(CustomGate(definition, [0]).to_matrix())


def test_random_orthogonal_matrix():
    n_circ = random.randint(1, 8)
    n_gate = random.randint(1, n_circ)
    t = random.randint(0, n_circ)
    m = rand_orthogonal_matrix(2**n_gate)
    c = QCircuit([CustomGate(UnitaryMatrix(m), [(t+i) % n_gate for i in range(n_gate)])])
    exp_state_vector = m[:, 0]
    assert matrix_eq(run(c, ATOSDevice.MYQLM_PYLINALG).state_vector.amplitudes, exp_state_vector)
