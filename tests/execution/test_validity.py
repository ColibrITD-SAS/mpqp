import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.gates import *
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.execution import run, IBMDevice, AWSDevice, ATOSDevice
from mpqp.tools import Matrix, rand_hermitian_matrix, atol, rtol
from mpqp.tools.maths import matrix_eq

pi = np.pi
s = np.sqrt
e = np.exp

# TODO: Add Cirq simulators to the list, when available
state_vector_devices = [IBMDevice.AER_SIMULATOR_STATEVECTOR, ATOSDevice.MYQLM_CLINALG, ATOSDevice.MYQLM_PYLINALG,
                        AWSDevice.BRAKET_LOCAL_SIMULATOR]

sampling_devices = [IBMDevice.AER_SIMULATOR, ATOSDevice.MYQLM_CLINALG, ATOSDevice.MYQLM_PYLINALG,
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
    batch = run(hae_3_qubit_circuit(*parameters), state_vector_devices)
    for result in batch:
        assert matrix_eq(result.amplitudes, expected_vector)


@pytest.mark.parametrize(
    "gates, expected_vector",
    [
        ([H(0), H(1), H(2), CNOT(0, 1), P(pi / 3, 2), T(0), CZ(1, 2)],
         np.array([s(2) / 4, e(1j * pi / 3) * s(2) / 4, s(2) / 4, -e(1j * pi / 3) * s(2) / 4,
                   (s(2) / 2 + 1j * s(2) / 2) * s(2) / 4, (s(2) / 2 + 1j * s(2) / 2) * e(1j * pi / 3) * s(2) / 4,
                   (s(2) / 2 + 1j * s(2) / 2) * s(2) / 4, (-s(2) / 2 - 1j * s(2) / 2) * e(1j * pi / 3) * s(2) / 4])),
        ([H(0), Rk(4, 1), H(2), Rx(pi / 3, 0), CRk(6, 1, 2), CNOT(0, 1), X(2)],
         np.array([s(3) / 4 - 1j / 4, s(3) / 4 - 1j / 4, 0, 0, 0, 0, s(3) / 4 - 1j / 4, s(3) / 4 - 1j / 4])),
        ([H(0), H(1), H(2), SWAP(0, 1), Rz(pi / 7, 2), Z(0), Y(1), S(2), Id(0), U(pi / 2, -pi, pi / 3, 1),
          Ry(pi / 5, 2)],
         np.array([0.2329753102e-1 - 0.4845363113 * 1j, 0.3413257772 + 0.1522448750 * 1j,
                   0.2797471698 + 0.1345083586e-1 * 1j,
                   -0.878986196e-1 + 0.1970645294 * 1j, -0.2329753102e-1 + 0.4845363113 * 1j,
                   -0.3413257772 - 0.1522448750 * 1j,
                   -0.2797471698 - 0.1345083586e-1 * 1j, 0.878986196e-1 - 0.1970645294 * 1j])),

    ],
)
def test_state_vector_various_native_gates(gates: list[Gate], expected_vector: Matrix):
    batch = run(QCircuit(gates), state_vector_devices)
    for result in batch:
        assert matrix_eq(result.amplitudes, expected_vector)


@pytest.mark.parametrize(
    "gates, basis_states",
    [
        ([H(0), CNOT(0, 1), CNOT(1, 2), ], ["000", "111"]),
        ([H(0), H(2), CNOT(0, 1), Ry(1.87, 1), H(0), CNOT(2, 3), H(4)],
         ["00000", "00011", "00110", "00111", "01000", "01001", "01110", "01111", "10000", "10001", "10110", "10111",
          "11000", "11001", "11110", "11111"]),
        ([X(0), SWAP(0, 1), X(2), Y(0), CNOT(1, 2), S(0), T(1), H(2)], ["110", "111"]),
    ],
)
def test_sample_basis_state_in_samples(gates: list[Gate], basis_states: list[str]):
    c = QCircuit(gates)
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=10000))
    batch = run(c, sampling_devices)
    nb_states = len(basis_states)
    for result in batch:
        assert len(result.samples) == nb_states


# @pytest.mark.parametrize(
#     "instructions",
#     [
#         ([H(0), CNOT(0, 1), CNOT(1, 2)]),
#     ],
# )
# def test_sample_counts_in_trust_interval(instructions: list[Instruction]):
#     c = QCircuit(instructions)
#     shots = 500000
#     expected_counts = [int(count) for count in np.round(shots * run(c, state_vector_devices[0]).probabilities)]
#     c.add(BasisMeasure(list(range(c.nb_qubits)), shots=shots))
#     batch = run(c, sampling_devices)
#     for result in batch:
#         counts = result.counts
#         # check if the true value is inside the trust interval
#         for i in range(len(counts)):
#             test = 100*shots/expected_counts[i]
#             assert np.floor(counts[i]-test) <= expected_counts[i] <= np.ceil(counts[i]+test)
# TODO: doesn't work apparently

@pytest.mark.parametrize(
    "gates, observable, expected_vector",
    [
        ([H(0), Rk(4, 1), H(2), Rx(pi / 3, 0), CRk(6, 1, 2), CNOT(0, 1), X(2)],
         rand_hermitian_matrix(2 ** 3),
         np.array([s(3) / 4 - 1j / 4, s(3) / 4 - 1j / 4, 0, 0, 0, 0, s(3) / 4 - 1j / 4, s(3) / 4 - 1j / 4])),
    ],
)
def test_observable_ideal_case(gates: list[gate], observable: Matrix, expected_vector: Matrix):
    c = QCircuit(gates)
    c.add(ExpectationMeasure(list(range(c.nb_qubits)), Observable(observable)))
    expected_value = expected_vector.transpose().conjugate().dot(observable.dot(expected_vector))
    batch = run(c, sampling_devices)
    for result in batch:
        assert abs(result.expectation_value - expected_value) < (atol + rtol * abs(expected_value))

# @pytest.mark.parametrize(
#     "gates, observable, expected_interval",
#     [
#         ([], np.array([]), (0.0, 0.0)),
#     ],
# )
# def test_observable_shot_noise(gates: list[Instruction], observable: Matrix, expected_interval: tuple[float, float]):
#     pass
# TODO
