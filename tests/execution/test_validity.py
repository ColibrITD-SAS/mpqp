import numpy as np
import numpy.typing as npt
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.execution import ATOSDevice, AWSDevice, GOOGLEDevice, IBMDevice, run
from mpqp.execution.result import BatchResult, Result
from mpqp.gates import *
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.tools import Matrix, atol, rand_hermitian_matrix, rtol
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning
from mpqp.tools.maths import matrix_eq

pi = np.pi
s = np.sqrt
e = np.exp

state_vector_devices = [
    IBMDevice.AER_SIMULATOR_STATEVECTOR,
    GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    ATOSDevice.MYQLM_CLINALG,
    ATOSDevice.MYQLM_PYLINALG,
    AWSDevice.BRAKET_LOCAL_SIMULATOR,
]

sampling_devices = [
    GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
    IBMDevice.AER_SIMULATOR,
    ATOSDevice.MYQLM_CLINALG,
    ATOSDevice.MYQLM_PYLINALG,
    AWSDevice.BRAKET_LOCAL_SIMULATOR,
]


def hae_3_qubit_circuit(
    t1: float, t2: float, t3: float, t4: float, t5: float, t6: float
):
    return QCircuit(
        [
            Ry(t1, 0),
            Ry(t2, 1),
            Ry(t3, 2),
            CNOT(0, 1),
            CNOT(1, 2),
            Ry(t4, 0),
            Ry(t5, 1),
            Ry(t6, 2),
            CNOT(0, 1),
            CNOT(1, 2),
        ]
    )


@pytest.mark.parametrize(
    "parameters, expected_vector",
    [
        ([0, 0, 0, 0, 0, 0], np.array([1, 0, 0, 0, 0, 0, 0, 0])),
        ([pi / 2, 0, -pi, pi / 2, 0, 0], np.array([0, -0.5, 0, 0.5, -0.5, 0, -0.5, 0])),
        (
            [pi / 2, pi, -pi / 2, pi / 5, 0, -pi],
            np.array(
                [
                    np.sin(pi / 10),
                    np.sin(pi / 10),
                    np.cos(pi / 10),
                    np.cos(pi / 10),
                    np.sin(pi / 10),
                    np.sin(pi / 10),
                    -np.cos(pi / 10),
                    -np.cos(pi / 10),
                ]
            )
            * 0.5,
        ),
        (
            [0.34, 0.321, -0.7843, 1.2232, 4.2323, 6.66],
            np.array(
                [
                    0.3812531672,
                    -0.05833733076,
                    0.1494487426,
                    -0.6633351291,
                    -0.5508843680,
                    0.1989958354,
                    0.1014433799,
                    0.1884958074,
                ]
            ),
        ),
    ],
)
def test_state_vector_result_HEA_ansatz(
    parameters: list[float], expected_vector: npt.NDArray[np.complex64]
):
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        batch = run(hae_3_qubit_circuit(*parameters), state_vector_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        assert matrix_eq(result.amplitudes, expected_vector)


@pytest.mark.parametrize(
    "gates, expected_vector",
    [
        (
            [H(0), H(1), H(2), CNOT(0, 1), P(pi / 3, 2), T(0), CZ(1, 2)],
            np.array(
                [
                    s(2) / 4,
                    e(1j * pi / 3) * s(2) / 4,
                    s(2) / 4,
                    -e(1j * pi / 3) * s(2) / 4,
                    (s(2) / 2 + 1j * s(2) / 2) * s(2) / 4,
                    (s(2) / 2 + 1j * s(2) / 2) * e(1j * pi / 3) * s(2) / 4,
                    (s(2) / 2 + 1j * s(2) / 2) * s(2) / 4,
                    (-s(2) / 2 - 1j * s(2) / 2) * e(1j * pi / 3) * s(2) / 4,
                ]
            ),
        ),
        (
            [H(0), Rk(4, 1), H(2), Rx(pi / 3, 0), CRk(6, 1, 2), CNOT(0, 1), X(2)],
            np.array(
                [
                    s(3) / 4 - 1j / 4,
                    s(3) / 4 - 1j / 4,
                    0,
                    0,
                    0,
                    0,
                    s(3) / 4 - 1j / 4,
                    s(3) / 4 - 1j / 4,
                ]
            ),
        ),
        (
            [
                H(0),
                H(1),
                H(2),
                SWAP(0, 1),
                Rz(pi / 7, 2),
                Z(0),
                Y(1),
                S(2),
                Id(0),
                U(pi / 2, -pi, pi / 3, 1),
                Ry(pi / 5, 2),
            ],
            np.array(
                [
                    0.2329753102e-1 - 0.4845363113 * 1j,
                    0.3413257772 + 0.1522448750 * 1j,
                    0.2797471698 + 0.1345083586e-1 * 1j,
                    -0.878986196e-1 + 0.1970645294 * 1j,
                    -0.2329753102e-1 + 0.4845363113 * 1j,
                    -0.3413257772 - 0.1522448750 * 1j,
                    -0.2797471698 - 0.1345083586e-1 * 1j,
                    0.878986196e-1 - 0.1970645294 * 1j,
                ]
            ),
        ),
    ],
)
def test_state_vector_various_native_gates(gates: list[Gate], expected_vector: Matrix):
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        batch = run(QCircuit(gates), state_vector_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        if isinstance(result.device, GOOGLEDevice):
            # TODO : Cirq needs atol 1 as some results differ by 0.1
            assert matrix_eq(result.amplitudes, expected_vector, atol=1)
        else:
            assert matrix_eq(result.amplitudes, expected_vector)


@pytest.mark.parametrize(
    "gates, basis_states",
    [
        (
            [
                H(0),
                CNOT(0, 1),
                CNOT(1, 2),
            ],
            ["000", "111"],
        ),
        (
            [H(0), H(2), CNOT(0, 1), Ry(1.87, 1), H(0), CNOT(2, 3), H(4)],
            [
                "00000",
                "00011",
                "00110",
                "00111",
                "01000",
                "01001",
                "01110",
                "01111",
                "10000",
                "10001",
                "10110",
                "10111",
                "11000",
                "11001",
                "11110",
                "11111",
            ],
        ),
        ([X(0), SWAP(0, 1), X(2), Y(0), CNOT(1, 2), S(0), T(1), H(2)], ["110", "111"]),
    ],
)
def test_sample_basis_state_in_samples(gates: list[Gate], basis_states: list[str]):
    c = QCircuit(gates)
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=10000))
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)
    nb_states = len(basis_states)
    for result in batch:
        assert len(result.samples) == nb_states


@pytest.mark.parametrize(
    "instructions",
    [
        ([H(0), CNOT(0, 1), CNOT(1, 2)]),
    ],
)
def test_sample_counts_in_trust_interval(instructions: list[Gate]):
    c = QCircuit(instructions)
    shots = 500000
    err_rate = 0.1
    err_rate_pourcentage = 1 - np.power(1 - err_rate, (1 / 2))
    res = run(c, state_vector_devices[0])
    assert isinstance(res, Result)
    expected_counts = [int(count) for count in np.round(shots * res.probabilities)]
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=shots))
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        counts = result.counts
        # check if the true value is inside the trust interval
        for i in range(len(counts)):
            trust_interval = np.ceil(
                err_rate_pourcentage * expected_counts[i]
                + shots / 200 * min(1, expected_counts[i] / 90)
            )
            assert (
                np.floor(counts[i] - trust_interval)
                <= expected_counts[i]
                <= np.ceil(counts[i] + trust_interval)
            )


@pytest.mark.parametrize(
    "gates, observable, expected_vector",
    [
        (
            [H(0), Rk(4, 1), H(2), Rx(pi / 3, 0), CRk(6, 1, 2), CNOT(0, 1), X(2)],
            rand_hermitian_matrix(2**3),
            np.array(
                [
                    s(3) / 4 - 1j / 4,
                    s(3) / 4 - 1j / 4,
                    0,
                    0,
                    0,
                    0,
                    s(3) / 4 - 1j / 4,
                    s(3) / 4 - 1j / 4,
                ]
            ),
        ),
    ],
)
def test_observable_ideal_case(
    gates: list[Gate], observable: npt.NDArray[np.complex64], expected_vector: Matrix
):
    c = QCircuit(gates)
    c.add(ExpectationMeasure(list(range(c.nb_qubits)), Observable(observable)))
    expected_value = (
        expected_vector.transpose().conjugate().dot(observable.dot(expected_vector))
    )
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        assert abs(result.expectation_value - expected_value) < (
            atol + rtol * abs(expected_value)
        )


# @pytest.mark.parametrize(
#     "gates, observable, expected_interval",
#     [
#         ([], np.array([]), (0.0, 0.0)),
#     ],
# )
# def test_observable_shot_noise(gates: list[Instruction], observable: Matrix, expected_interval: tuple[float, float]):
#     pass
# TODO
