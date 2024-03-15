import numpy as np
import pytest


from mpqp import QCircuit
from mpqp.gates import *
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.execution import run, IBMDevice, AWSDevice, ATOSDevice
from mpqp.tools.maths import matrix_eq


def hae_3_qubit_circuit(t1, t2, t3, t4, t5, t6):
    return QCircuit([Ry(t1, 0), Ry(t2, 1), Ry(t3, 2), CNOT(0,1), CNOT(1,2),
                     Ry(t4, 0), Ry(t5, 1), Ry(t6, 2), CNOT(0,1), CNOT(1,2)])


@pytest.mark.parametrize(
    "parameters, expected_vector",
    [
        ([0, 0, 0, 0, 0, 0, 0], np.array([1, 0, 0, 0, 0, 0, 0, 0])),
    ],
)
def test_state_vector_result_HEA_ansatz(parameters, expected_vector):

    batch = run(hae_3_qubit_circuit(*parameters), [IBMDevice.AER_SIMULATOR_STATEVECTOR,
                          ATOSDevice.MYQLM_CLINALG,
                          ATOSDevice.MYQLM_PYLINALG,
                          ])
    #TODO: Julien , add cirq simulators, and statevector simulators here
    for result in batch:
        assert matrix_eq(result.amplitudes, expected_vector)
