import random

import pytest

import numpy as np

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.tools.maths import matrix_eq, rand_orthogonal_matrix, is_unitary
from mpqp.execution import run, ATOSDevice, IBMDevice, AWSDevice


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


@pytest.mark.parametrize(
    "n_circ",
    [
        (random.randint(1, 5)) for _ in range(10)
    ],
)
def test_random_orthogonal_matrix(n_circ: int):
    n_gate = random.randint(1, n_circ)
    t = random.randint(0, n_circ-n_gate)
    m = rand_orthogonal_matrix(2**n_gate)
    c = QCircuit([CustomGate(UnitaryMatrix(m), [(t+i) for i in range(n_gate)])])
    exp_state_vector = m[:, 0]
    for _ in range(0, t):
        exp_state_vector = np.kron(np.array([1, 0]), exp_state_vector)
    for _ in range(t+n_gate+1, n_circ):
        exp_state_vector = np.kron(exp_state_vector, np.array([1, 0]))

    print("n_circ", n_circ)
    print("n_gate", n_gate)
    print("t", t)
    print("m", m)
    c.pretty_print()
    print("exp_state_vector", exp_state_vector)

    execution_ibm_statevector = run(c, IBMDevice.AER_SIMULATOR).state_vector
    #execution_aws_statevector = run(c, AWSDevice.BRAKET_LOCAL_SIMULATOR).state_vector
    execution_qlm_statevector = run(c, ATOSDevice.MYQLM_PYLINALG).state_vector

    print("result", execution_ibm_statevector)

    # assert matrix_eq(execution_ibm_statevector.amplitudes, exp_state_vector)
    #assert matrix_eq(execution_aws_statevector.amplitudes, exp_state_vector)
    assert matrix_eq(execution_qlm_statevector.amplitudes, exp_state_vector)
