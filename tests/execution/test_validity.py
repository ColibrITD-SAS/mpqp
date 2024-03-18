import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.gates import *
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.execution import run, IBMDevice, AWSDevice, ATOSDevice
from mpqp.tools import Matrix
from mpqp.tools.maths import matrix_eq

pi = np.pi
s = np.sqrt
e = np.exp

# TODO: Add Cirq simulators to the list, when available
devices = [IBMDevice.AER_SIMULATOR_STATEVECTOR, ATOSDevice.MYQLM_CLINALG, ATOSDevice.MYQLM_PYLINALG,
           AWSDevice.BRAKET_LOCAL_SIMULATOR]


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
    batch = run(hae_3_qubit_circuit(*parameters), devices)
    for result in batch:
        assert matrix_eq(result.amplitudes, expected_vector)


@pytest.mark.parametrize(
    "gates, expected_vector",
    [
        ([H(0), H(1), H(2), CNOT(0, 1), P(pi / 3, 2), T(0), CZ(1, 2)],
         np.array([s(2) / 4, e(1j * pi / 3) * s(2) / 4, s(2) / 4, -e(1j * pi / 3) * s(2) / 4,
                   (s(2) / 2 + 1j * s(2) / 2) * s(2) / 4, (s(2) / 2 + 1j * s(2) / 2) * e(1j * pi / 3) * s(2) / 4,
                   (s(2) / 2 + 1j * s(2) / 2) * s(2) / 4, (-s(2) / 2 - 1j * s(2) / 2) * e(1j * pi / 3) * s(2) / 4])),
        ([H(0), Rk(4,1), H(2), Rx(pi/3,0), CRk(6,1,2), CNOT(0,1), X(2)],
         np.array([s(3)/4 - 1j/4, s(3)/4 - 1j/4, 0, 0,
                   0, 0, s(3)/4 - 1j/4, s(3)/4 - 1j/4])),
    ],
)
def test_state_vector_various_native_gates(gates: list[Gate], expected_vector: Matrix):
    batch = run(QCircuit(gates), devices)
    for result in batch:
        assert matrix_eq(result.amplitudes, expected_vector)


@pytest.mark.parametrize(
    "gates, basis_states",
    [
        ([], ["000"]),
    ],
)
def test_sample_basis_state_in_samples(gates: list[Gate], basis_states: list[str]):
    pass


@pytest.mark.parametrize(
    "gates, counts_intervals",
    [
        ([], [(0, 0.0)]),
    ],
)
def test_sample_counts_in_trust_interval(gates: list[Gate], counts_intervals: list[tuple[int, int]]):
    pass


@pytest.mark.parametrize(
    "gates, observable, expected_value",
    [
        ([], np.array([]), 0.0),
    ],
)
def test_observable_ideal_case(gates: list[Gate], observable: Matrix, expected_value: float):
    pass


@pytest.mark.parametrize(
    "gates, observable, expected_interval",
    [
        ([], np.array([]), (0.0, 0.0)),
    ],
)
def test_observable_shot_noise(gates: list[Gate], observable: Matrix, expected_interval: tuple[float, float]):
    pass
