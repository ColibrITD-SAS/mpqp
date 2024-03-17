import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.execution import run, IBMDevice, AWSDevice, ATOSDevice
from mpqp.tools import Matrix
from mpqp.tools.maths import matrix_eq

pi = np.pi

# TODO: Add Cirq simulators to the tests below, when available

def hae_3_qubit_circuit(t1, t2, t3, t4, t5, t6):
    return QCircuit([Ry(t1, 0), Ry(t2, 1), Ry(t3, 2), CNOT(0, 1), CNOT(1, 2),
                     Ry(t4, 0), Ry(t5, 1), Ry(t6, 2), CNOT(0, 1), CNOT(1, 2)])


@pytest.mark.parametrize(
    "parameters, expected_vector",
    [
        ([0, 0, 0, 0, 0, 0], np.array([1, 0, 0, 0, 0, 0, 0, 0])),
        ([pi / 2, 0, -pi, pi / 2, 0, 0], np.array([0, -0.5, 0, 0.5, -0.5, 0, -0.5, 0])),
        ([pi / 2, pi, -pi / 2, pi / 5, 0, -pi], np.array([0.5 * np.sin(pi / 10), 0.5 * np.sin(pi / 10),
                                                          0.5 * np.cos(pi / 10), 0.5 * np.cos(pi / 10),
                                                          0.5 * np.sin(pi / 10), 0.5 * np.sin(pi / 10),
                                                          -0.5 * np.cos(pi / 10), -0.5 * np.cos(pi / 10)])),
        ([0.34, 0.321, -0.7843, 1.2232, 4.2323, 6.66], np.array([0.3812531672, -0.05833733076,
                                                                 0.1494487426, -0.6633351291,
                                                                 -0.5508843680, 0.1989958354,
                                                                 0.1014433799, 0.1884958074])),
    ],
)
def test_state_vector_result_HEA_ansatz(parameters, expected_vector):
    batch = run(hae_3_qubit_circuit(*parameters),
                [IBMDevice.AER_SIMULATOR_STATEVECTOR,
                 ATOSDevice.MYQLM_CLINALG,
                 ATOSDevice.MYQLM_PYLINALG,
                 AWSDevice.BRAKET_LOCAL_SIMULATOR
                 ])
    for result in batch:
        assert matrix_eq(result.amplitudes, expected_vector)


@pytest.mark.parametrize(
    "instructions, expected_vector",
    [
        ([], np.array([])),
    ],
)
def test_state_vector_various_native_gates(gates: list[Gate], expected_vector: Matrix):
    batch = run(QCircuit(gates),
                [IBMDevice.AER_SIMULATOR_STATEVECTOR,
                 ATOSDevice.MYQLM_CLINALG,
                 ATOSDevice.MYQLM_PYLINALG,
                 AWSDevice.BRAKET_LOCAL_SIMULATOR
                 ])

    pass


def test_sample_basis_state_in_samples(gates: list[Gate], basis_states: list[str]):
    pass


def test_sample_counts_in_trust_interval(gates: list[Gate], counts_intervals: list[tuple[int, int]]):
    pass


def test_observable_ideal_case(gates: list[Gate], observable: Matrix, expected_value: float):
    pass
