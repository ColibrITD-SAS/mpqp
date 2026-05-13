from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pytest

from mpqp import (
    ATOSDevice,
    AWSDevice,
    AZUREDevice,
    Barrier,
    BasisMeasure,
    BatchResult,
    Breakpoint,
    ComputationalBasis,
    Depolarizing,
    ExpectationMeasure,
    GOOGLEDevice,
    HadamardBasis,
    IBMDevice,
    Instruction,
    Language,
    Measure,
    Observable,
    PhaseDamping,
    QCircuit,
    Result,
    VariableSizeBasis,
    pI,
    pX,
    pY,
    pZ,
    run,
)
from mpqp.core.instruction.gates.native_gates import NATIVE_GATES
from mpqp.execution import AvailableDevice
from mpqp.gates import *
from mpqp.measures import PauliString
from mpqp.noise.noise_model import NOISE_MODELS
from mpqp.tools import Matrix, atol, rand_hermitian_matrix, rtol
from mpqp.tools.circuit import random_gate, random_noise
from mpqp.tools.errors import (
    DeviceJobIncompatibleError,
)
from mpqp.tools.maths import matrix_eq

pi = np.pi
s = np.sqrt
e = np.exp


state_vector_devices_qiskit: list[AvailableDevice] = [
    device
    for device in IBMDevice
    if not device.is_remote() and device.supports_state_vector()
]

state_vector_devices_cirq: list[AvailableDevice] = [
    device
    for device in GOOGLEDevice
    if not device.is_remote() and device.supports_state_vector()
]

state_vector_devices_braket: list[AvailableDevice] = [
    device
    for device in AWSDevice
    if not device.is_remote() and device.supports_state_vector()
]

state_vector_devices_myqlm: list[AvailableDevice] = [
    device
    for device in ATOSDevice
    if not device.is_remote() and device.supports_state_vector()
]

sampling_devices_qiskit: list[AvailableDevice] = [
    device
    for device in IBMDevice
    if not device.is_remote()
    and device.supports_samples()
    and not device.has_reduced_gate_set()
]

sampling_devices_cirq: list[AvailableDevice] = [
    device
    for device in GOOGLEDevice
    if not device.is_remote() and device.supports_samples()
]

sampling_devices_braket: list[AvailableDevice] = [
    device
    for device in AWSDevice
    if not device.is_remote() and device.supports_samples()
]

sampling_devices_myqlm: list[AvailableDevice] = [
    device
    for device in ATOSDevice
    if not device.is_remote() and device.supports_samples()
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


list_param_expect_vector = [
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
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("parameters, expected_vector", list_param_expect_vector)
def test_state_vector_result_HEA_ansatz_qiskit(
    parameters: list[float], expected_vector: npt.NDArray[np.complex128]
):
    exec_state_vector_result_HEA_ansatz(
        parameters, expected_vector, state_vector_devices_qiskit
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("parameters, expected_vector", list_param_expect_vector)
def test_state_vector_result_HEA_ansatz_myqlm(
    parameters: list[float], expected_vector: npt.NDArray[np.complex128]
):
    exec_state_vector_result_HEA_ansatz(
        parameters, expected_vector, state_vector_devices_myqlm
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("parameters, expected_vector", list_param_expect_vector)
def test_state_vector_result_HEA_ansatz_cirq(
    parameters: list[float], expected_vector: npt.NDArray[np.complex128]
):
    exec_state_vector_result_HEA_ansatz(
        parameters, expected_vector, state_vector_devices_cirq
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize("parameters, expected_vector", list_param_expect_vector)
def test_state_vector_result_HEA_ansatz_braket(
    parameters: list[float], expected_vector: npt.NDArray[np.complex128]
):
    exec_state_vector_result_HEA_ansatz(
        parameters, expected_vector, state_vector_devices_braket
    )


def exec_state_vector_result_HEA_ansatz(
    parameters: list[float],
    expected_vector: npt.NDArray[np.complex128],
    state_vector_devices: list[AvailableDevice],
):
    batch = run(hae_3_qubit_circuit(*parameters), state_vector_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        assert isinstance(result, Result)
        assert matrix_eq(result.amplitudes, expected_vector)


list_gates_expect_vector = [
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
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("gates, expected_vector", list_gates_expect_vector)
def test_state_vector_various_native_gates_qiskit(
    gates: list[Gate], expected_vector: Matrix
):
    exec_state_vector_various_native_gates(
        gates, expected_vector, state_vector_devices_qiskit
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize("gates, expected_vector", list_gates_expect_vector)
def test_state_vector_various_native_gates_braket(
    gates: list[Gate], expected_vector: Matrix
):
    exec_state_vector_various_native_gates(
        gates, expected_vector, state_vector_devices_braket
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("gates, expected_vector", list_gates_expect_vector)
def test_state_vector_various_native_gates_cirq(
    gates: list[Gate], expected_vector: Matrix
):
    exec_state_vector_various_native_gates(
        gates, expected_vector, state_vector_devices_cirq
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("gates, expected_vector", list_gates_expect_vector)
def test_state_vector_various_native_gates_myqlm(
    gates: list[Gate], expected_vector: Matrix
):
    exec_state_vector_various_native_gates(
        gates, expected_vector, state_vector_devices_myqlm
    )


def exec_state_vector_various_native_gates(
    gates: list[Gate],
    expected_vector: Matrix,
    state_vector_devices: list[AvailableDevice],
):
    batch = run(QCircuit(gates), state_vector_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        assert isinstance(result, Result)
        if isinstance(result.device, GOOGLEDevice):
            # TODO : Cirq needs atol 1 as some results differ by 0.1
            assert matrix_eq(result.amplitudes, expected_vector, atol=1)
        else:
            assert matrix_eq(result.amplitudes, expected_vector)


list_gates_basis_states = [
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
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("gates, basis_states", list_gates_basis_states)
def test_sample_basis_state_in_samples_qiskit(
    gates: list[Gate], basis_states: list[str]
):
    exec_sample_basis_state_in_samples(gates, basis_states, sampling_devices_qiskit)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("gates, basis_states", list_gates_basis_states)
def test_sample_basis_state_in_samples_braket(
    gates: list[Gate], basis_states: list[str]
):
    exec_sample_basis_state_in_samples(gates, basis_states, sampling_devices_braket)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("gates, basis_states", list_gates_basis_states)
def test_sample_basis_state_in_samples_cirq(gates: list[Gate], basis_states: list[str]):
    exec_sample_basis_state_in_samples(gates, basis_states, sampling_devices_cirq)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("gates, basis_states", list_gates_basis_states)
def test_sample_basis_state_in_samples_myqlm(
    gates: list[Gate], basis_states: list[str]
):
    exec_sample_basis_state_in_samples(gates, basis_states, sampling_devices_myqlm)


def exec_sample_basis_state_in_samples(
    gates: list[Gate], basis_states: list[str], sampling_devices: list[AvailableDevice]
):
    c = QCircuit(gates)
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=10000))
    batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)
    nb_states = len(basis_states)
    for result in batch:
        print(result.device)
        assert isinstance(result, Result)
        assert len(result.samples) == nb_states


list_instruction_proba = [
    (
        [H(0), CNOT(0, 1), CNOT(1, 2)],
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    ),
    ([CustomGate(np.array([[0, 1], [1, 0]]), [1])], np.array([0.0, 1.0, 0.0, 0.0])),
    (
        [U(0.215, 0.5588, 8, 1)],
        np.array([0.9884881971034313, 0.011511802896568812, 0.0, 0.0]),
    ),
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("instructions, probabilities", list_instruction_proba)
def test_sample_counts_in_trust_interval_qiskit(
    instructions: list[Gate], probabilities: list[float]
):
    exec_sample_counts_in_trust_interval(
        instructions, probabilities, sampling_devices_qiskit
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("instructions, probabilities", list_instruction_proba)
def test_sample_counts_in_trust_interval_cirq(
    instructions: list[Gate], probabilities: list[float]
):
    exec_sample_counts_in_trust_interval(
        instructions, probabilities, sampling_devices_cirq
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize("instructions, probabilities", list_instruction_proba)
def test_sample_counts_in_trust_interval_braket(
    instructions: list[Gate], probabilities: list[float]
):
    exec_sample_counts_in_trust_interval(
        instructions, probabilities, sampling_devices_braket
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("instructions, probabilities", list_instruction_proba)
def test_sample_counts_in_trust_interval_myqlm(
    instructions: list[Gate], probabilities: list[float]
):
    exec_sample_counts_in_trust_interval(
        instructions, probabilities, sampling_devices_myqlm
    )


def exec_sample_counts_in_trust_interval(
    instructions: list[Gate],
    probabilities: list[float],
    sampling_devices: list[AvailableDevice],
):
    c = QCircuit(instructions)
    shots = 50000
    err_rate = 0.2
    err_rate_percentage = 1 - np.power(1 - err_rate, (1 / 2))
    res = run(c, IBMDevice.AER_SIMULATOR_STATEVECTOR)
    expected_counts = [int(count) for count in np.round(shots * res.probabilities)]
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=shots))
    batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        assert isinstance(result, Result)
        print("expected_counts: " + str(expected_counts))
        counts = result.counts
        # check if the true value is inside the trust interval
        for i in range(len(counts)):
            trust_interval = np.ceil(
                err_rate_percentage * expected_counts[i] + shots / 15
            )
            assert (
                np.floor(counts[i] - trust_interval)
                <= expected_counts[i]
                <= np.ceil(counts[i] + trust_interval)
            )


list_gate_obs_vector = [
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
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("gates, observable, expected_vector", list_gate_obs_vector)
def test_observable_ideal_case_qiskit(
    gates: list[Gate], observable: npt.NDArray[np.complex128], expected_vector: Matrix
):
    exec_observable_ideal_case(
        gates, observable, expected_vector, state_vector_devices_qiskit
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("gates, observable, expected_vector", list_gate_obs_vector)
def test_observable_ideal_case_cirq(
    gates: list[Gate], observable: npt.NDArray[np.complex128], expected_vector: Matrix
):
    exec_observable_ideal_case(
        gates, observable, expected_vector, state_vector_devices_cirq
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize("gates, observable, expected_vector", list_gate_obs_vector)
def test_observable_ideal_case_braket(
    gates: list[Gate], observable: npt.NDArray[np.complex128], expected_vector: Matrix
):
    exec_observable_ideal_case(
        gates, observable, expected_vector, state_vector_devices_braket
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("gates, observable, expected_vector", list_gate_obs_vector)
def test_observable_ideal_case_myqlm(
    gates: list[Gate], observable: npt.NDArray[np.complex128], expected_vector: Matrix
):
    exec_observable_ideal_case(
        gates, observable, expected_vector, state_vector_devices_myqlm
    )


def exec_observable_ideal_case(
    gates: list[Gate],
    observable: npt.NDArray[np.complex128],
    expected_vector: Matrix,
    sampling_devices: list[AvailableDevice],
):
    c = QCircuit(gates)
    c.add(
        ExpectationMeasure(
            Observable(observable), list(range(c.nb_qubits)), optimize_measurement=False
        )
    )
    expected_value = float(
        expected_vector.transpose()
        .conjugate()
        .dot(observable.dot(expected_vector))
        .real  # pyright: ignore[reportAttributeAccessIssue]
    )
    batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        evs = result.expectation_values
        assert isinstance(evs, float)
        assert abs(evs - expected_value) < (atol + rtol * abs(expected_value))


@pytest.fixture
def circuits_type():
    circuit_state_vector = QCircuit([H(0), CNOT(0, 1)])

    circuit_samples = deepcopy(circuit_state_vector)
    circuit_samples.add(BasisMeasure())

    observable = np.array([[4, 2, 3, 8], [2, -3, 1, 0], [3, 1, -1, 5], [8, 0, 5, 2]])
    circuit_observable = deepcopy(circuit_state_vector)
    circuit_observable.add(ExpectationMeasure(Observable(observable)))

    observable_ideal = rand_hermitian_matrix(2**2)
    circuit_observable_ideal = deepcopy(circuit_state_vector)
    circuit_observable_ideal.add(ExpectationMeasure(Observable(observable_ideal)))
    return [
        circuit_state_vector,
        circuit_samples,
        circuit_observable,
        circuit_observable_ideal,
    ]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("device", list(IBMDevice))
def test_validity_run_job_type_qiskit(
    device: AvailableDevice, circuits_type: list[QCircuit]
):
    exec_validity_run_job_type(device, circuits_type)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("device", list(GOOGLEDevice))
def test_validity_run_job_type_cirq(
    device: AvailableDevice, circuits_type: list[QCircuit]
):
    exec_validity_run_job_type(device, circuits_type)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("device", list(ATOSDevice))
def test_validity_run_job_type_myqlm(
    device: AvailableDevice, circuits_type: list[QCircuit]
):
    exec_validity_run_job_type(device, circuits_type)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("device", list(AWSDevice))
def test_validity_run_job_type_braket(
    device: AvailableDevice, circuits_type: list[QCircuit]
):
    exec_validity_run_job_type(device, circuits_type)


@pytest.mark.provider("azure")
@pytest.mark.parametrize(
    "device",
    list(AZUREDevice),
)
def test_validity_run_job_type_azure(
    device: AvailableDevice, circuits_type: list[QCircuit]
):
    exec_validity_run_job_type(device, circuits_type)


def exec_validity_run_job_type(device: AvailableDevice, circuits_type: list[QCircuit]):
    circuit_state_vector = circuits_type[0]
    circuit_samples = circuits_type[1]
    circuit_observable = circuits_type[2]
    circuit_observable_ideal = circuits_type[3]

    if not device.is_remote():
        if device.supports_samples():
            assert run(circuit_samples, device) is not None
        else:
            run(circuit_samples, device)

        if device.supports_state_vector():
            assert run(circuit_state_vector, device) is not None
        else:
            if isinstance(device, IBMDevice) and not device.supports_state_vector():
                with pytest.raises(DeviceJobIncompatibleError):
                    run(circuit_state_vector, device)
            else:
                with pytest.raises(NotImplementedError):
                    run(circuit_state_vector, device)

        if device.supports_observable():
            if isinstance(device, GOOGLEDevice) and device.is_processor():
                with pytest.raises(DeviceJobIncompatibleError):
                    run(circuit_observable, device)
                circuit_observable.measurements[0].shots = 10
                assert run(circuit_observable, device) is not None
            else:
                assert run(circuit_observable, device) is not None

        else:
            if isinstance(device, IBMDevice):
                with pytest.raises(DeviceJobIncompatibleError):
                    run(circuit_observable, device)
            else:
                with pytest.raises(NotImplementedError):
                    run(circuit_observable, device)

        if device.supports_observable_ideal():
            if isinstance(device, GOOGLEDevice) and device.is_processor():
                with pytest.raises(DeviceJobIncompatibleError):
                    run(circuit_observable_ideal, device)
                circuit_observable_ideal.measurements[0].shots = 10
                assert run(circuit_observable_ideal, device) is not None
            else:
                assert run(circuit_observable_ideal, device) is not None
        else:
            if isinstance(device, IBMDevice):
                with pytest.raises(DeviceJobIncompatibleError):
                    run(circuit_observable, device)
            else:
                with pytest.raises(NotImplementedError):
                    run(circuit_observable, device)


@pytest.mark.provider("qiskit")
def test_validity_native_gate_to_other_language_qiskit():
    exec_validity_native_gate_to_other_language(Language.QISKIT)


@pytest.mark.provider("cirq")
def test_validity_native_gate_to_other_language_cirq():
    exec_validity_native_gate_to_other_language(Language.CIRQ)


@pytest.mark.provider("braket")
def test_validity_native_gate_to_other_language_braket():
    exec_validity_native_gate_to_other_language(Language.BRAKET)


@pytest.mark.provider("myqlm")
def test_validity_native_gate_to_other_language_myqlm():
    exec_validity_native_gate_to_other_language(Language.MY_QLM)


def test_validity_native_gate_to_other_language_qasm2():
    exec_validity_native_gate_to_other_language(Language.QASM2)


def test_validity_native_gate_to_other_language_qasm3():
    exec_validity_native_gate_to_other_language(Language.QASM3)


def exec_validity_native_gate_to_other_language(language: Language):
    for gate in NATIVE_GATES:
        gate_build = random_gate([gate])

        if language in [Language.MY_QLM, Language.QASM3]:
            with pytest.raises(NotImplementedError):
                gate_build.to_other_language(language)
        else:
            if isinstance(gate_build, ComposedGate):
                assert all(
                    [
                        gate.to_other_language(language) is not None
                        for gate in gate_build.decompose()
                    ]
                )
            else:
                assert gate_build.to_other_language(language) is not None


@pytest.fixture
def measures():
    return [
        BasisMeasure([0, 1]),
        BasisMeasure(
            [0, 1],
            shots=1024,
            basis=VariableSizeBasis([np.array([1, 0]), np.array([0, -1])]),
        ),
        BasisMeasure([0, 1], shots=1024, basis=ComputationalBasis(2)),
        BasisMeasure([0, 1], shots=1024, basis=HadamardBasis(2)),
        ExpectationMeasure(Observable([0.7, -1, 1, 1]), shots=10),
    ]


@pytest.mark.provider("qiskit")
def test_validity_measure_to_other_language_qiskit(measures: list[Measure]):
    exec_validity_measure_to_other_language(Language.QISKIT, measures)


@pytest.mark.provider("cirq")
def test_validity_measure_to_other_language_cirq(measures: list[Measure]):
    exec_validity_measure_to_other_language(Language.CIRQ, measures)


@pytest.mark.provider("braket")
def test_validity_measure_to_other_language_braket(measures: list[Measure]):
    exec_validity_measure_to_other_language(Language.BRAKET, measures)


@pytest.mark.provider("myqlm")
def test_validity_measure_to_other_language_myqlm(measures: list[Measure]):
    exec_validity_measure_to_other_language(Language.MY_QLM, measures)


def test_validity_measure_to_other_language_qasm2(measures: list[Measure]):
    exec_validity_measure_to_other_language(Language.QASM2, measures)


def test_validity_measure_to_other_language_qasm3(measures: list[Measure]):
    exec_validity_measure_to_other_language(Language.QASM3, measures)


def exec_validity_measure_to_other_language(
    language: Language, measures: list[Measure]
):
    for measure in measures:
        if isinstance(measure, ExpectationMeasure):
            with pytest.raises(NotImplementedError):
                measure.to_other_language(language)
        elif language in [
            Language.MY_QLM,
            Language.BRAKET,
            Language.QASM3,
        ]:
            with pytest.raises(NotImplementedError):
                measure.to_other_language(language)
        else:
            assert measure.to_other_language(language) is not None


@pytest.fixture
def pauli_strings():
    return [pI @ pX @ pY @ pZ, pX + pZ, pY]


@pytest.mark.provider("qiskit")
def test_validity_pauli_string_to_other_language_qiskit(
    pauli_strings: list[PauliString],
):
    exec_validity_pauli_string_to_other_language(Language.QISKIT, pauli_strings)


@pytest.mark.provider("cirq")
def test_validity_pauli_string_to_other_language_cirq(pauli_strings: list[PauliString]):
    exec_validity_pauli_string_to_other_language(Language.CIRQ, pauli_strings)


@pytest.mark.provider("braket")
def test_validity_pauli_string_to_other_language_braket(
    pauli_strings: list[PauliString],
):
    exec_validity_pauli_string_to_other_language(Language.BRAKET, pauli_strings)


@pytest.mark.provider("myqlm")
def test_validity_pauli_string_to_other_language_myqlm(
    pauli_strings: list[PauliString],
):
    exec_validity_pauli_string_to_other_language(Language.MY_QLM, pauli_strings)


def test_validity_pauli_string_to_other_language_qasm2(
    pauli_strings: list[PauliString],
):
    exec_validity_pauli_string_to_other_language(Language.QASM2, pauli_strings)


def test_validity_pauli_string_to_other_language_qasm3(
    pauli_strings: list[PauliString],
):
    exec_validity_pauli_string_to_other_language(Language.QASM3, pauli_strings)


def exec_validity_pauli_string_to_other_language(
    language: Language, pauli_strings: list[PauliString]
):

    for pauli_string in pauli_strings:
        if language in [Language.QASM3, Language.QASM2]:
            with pytest.raises(NotImplementedError):
                pauli_string.to_other_language(language)
        else:
            assert pauli_string.to_other_language(language) is not None


@pytest.mark.provider("qiskit")
def test_validity_noise_to_other_language_qiskit():
    exec_validity_noise_to_other_language(Language.QISKIT)


@pytest.mark.provider("cirq")
def test_validity_noise_to_other_language_cirq():
    exec_validity_noise_to_other_language(Language.CIRQ)


@pytest.mark.provider("braket")
def test_validity_noise_to_other_language_braket():
    exec_validity_noise_to_other_language(Language.BRAKET)


@pytest.mark.provider("myqlm")
def test_validity_noise_to_other_language_myqlm():
    exec_validity_noise_to_other_language(Language.MY_QLM)


def test_validity_noise_to_other_language_qasm2():
    exec_validity_noise_to_other_language(Language.QASM2)


def test_validity_noise_to_other_language_qasm3():
    exec_validity_noise_to_other_language(Language.QASM3)


def exec_validity_noise_to_other_language(language: Language):
    for noise in NOISE_MODELS:
        noise_build = random_noise([noise])

        if language in [Language.QASM3, Language.QASM2]:
            with pytest.raises(NotImplementedError):
                noise_build.to_other_language(language)

        elif language in [Language.MY_QLM] and not isinstance(
            noise_build, (Depolarizing, PhaseDamping)
        ):
            with pytest.raises(NotImplementedError):
                noise_build.to_other_language(language)

        else:
            assert noise_build.to_other_language(language) is not None


@pytest.fixture
def other_instr():
    return [
        Breakpoint(),
        Barrier(),
    ]


@pytest.mark.provider("qiskit")
def test_validity_other_instr_to_other_language_qiskit(
    other_instr: list[Instruction],
):
    exec_validity_other_instr_to_other_language(Language.QISKIT, other_instr)


@pytest.mark.provider("cirq")
def test_validity_other_instr_to_other_language_cirq(other_instr: list[Instruction]):
    exec_validity_other_instr_to_other_language(Language.CIRQ, other_instr)


@pytest.mark.provider("braket")
def test_validity_other_instr_to_other_language_braket(
    other_instr: list[Instruction],
):
    exec_validity_other_instr_to_other_language(Language.BRAKET, other_instr)


@pytest.mark.provider("myqlm")
def test_validity_other_instr_to_other_language_myqlm(
    other_instr: list[Instruction],
):
    exec_validity_other_instr_to_other_language(Language.MY_QLM, other_instr)


def test_validity_other_instr_to_other_language_qasm2(
    other_instr: list[Instruction],
):
    exec_validity_other_instr_to_other_language(Language.QASM2, other_instr)


def test_validity_other_instr_to_other_language_qasm3(
    other_instr: list[Instruction],
):
    exec_validity_other_instr_to_other_language(Language.QASM3, other_instr)


def exec_validity_other_instr_to_other_language(
    language: Language, other_instr: list[Instruction]
):
    for instr in other_instr:
        if isinstance(instr, Breakpoint):
            with pytest.raises(NotImplementedError):
                instr.to_other_language(language)
        elif language in [
            Language.MY_QLM,
            Language.CIRQ,
            Language.BRAKET,
            Language.QASM3,
        ]:
            with pytest.raises(NotImplementedError):
                instr.to_other_language(language)
        else:
            assert instr.to_other_language(language) is not None


list_circ_obs = [
    (QCircuit([H(0), H(1)]), Observable([1, 2, 5, 3])),
    (QCircuit([S(0), T(1)]), Observable([-1, 4, 0, 1])),
    (QCircuit([Rx(0.5, 0), Ry(0.6, 1)]), Observable([0, 0, -9, 7])),
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "circuit, observable",
    list_circ_obs,
)
def test_validity_optim_ideal_single_diag_obs_and_regular_run_qiskit(
    circuit: QCircuit, observable: Observable
):
    exec_validity_optim_ideal_single_diag_obs_and_regular_run(
        circuit,
        observable,
        IBMDevice.AER_SIMULATOR,
    )


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "circuit, observable",
    list_circ_obs,
)
def test_validity_optim_ideal_single_diag_obs_and_regular_run_cirq(
    circuit: QCircuit, observable: Observable
):
    exec_validity_optim_ideal_single_diag_obs_and_regular_run(
        circuit, observable, IBMDevice.AER_SIMULATOR
    )


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit, observable",
    list_circ_obs,
)
def test_validity_optim_ideal_single_diag_obs_and_regular_run_braket(
    circuit: QCircuit, observable: Observable
):
    exec_validity_optim_ideal_single_diag_obs_and_regular_run(
        circuit,
        observable,
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
    )


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize(
    "circuit, observable",
    list_circ_obs,
)
def test_validity_optim_ideal_single_diag_obs_and_regular_run_myqlm(
    circuit: QCircuit, observable: Observable
):
    exec_validity_optim_ideal_single_diag_obs_and_regular_run(
        circuit, observable, ATOSDevice.MYQLM_PYLINALG
    )


def exec_validity_optim_ideal_single_diag_obs_and_regular_run(
    circuit: QCircuit, observable: Observable, device: AvailableDevice
):
    e1 = ExpectationMeasure(
        observable, shots=0, optim_diagonal=False, optimize_measurement=False
    )
    e2 = ExpectationMeasure(
        observable, shots=0, optim_diagonal=True, optimize_measurement=False
    )
    c1 = circuit + QCircuit([e1], nb_qubits=2)
    c2 = circuit + QCircuit([e2], nb_qubits=2)
    r1 = run(c1, device)
    r2 = run(c2, device)
    assert isinstance(r1.expectation_values, float)
    assert isinstance(r2.expectation_values, float)
    assert np.isclose(r1.expectation_values, r2.expectation_values)


@pytest.mark.parametrize(
    "circuit, o1, o2",
    [
        (QCircuit([H(0), H(1)]), Observable([1, 2, 5, 3]), Observable([-1, 4, 0, 1])),
        (QCircuit([S(0), T(1)]), Observable([-1, 4, 0, 1]), Observable([0, 0, -9, 7])),
        (
            QCircuit([Rx(0.5, 0), Ry(0.6, 1)]),
            Observable([0, 0, -9, 7]),
            Observable([1, 2, 5, 3]),
        ),
    ],
)
def test_validity_optim_ideal_multi_diag_obs_and_regular_run(
    circuit: QCircuit, o1: Observable, o2: Observable
):
    e1 = ExpectationMeasure([o1, o2], shots=0, optim_diagonal=False)
    e2 = ExpectationMeasure([o1, o2], shots=0, optim_diagonal=True)
    c1 = circuit + QCircuit([e1], nb_qubits=2)
    c2 = circuit + QCircuit([e2], nb_qubits=2)
    br1 = run(
        c1,
        [
            IBMDevice.AER_SIMULATOR,
            # ATOSDevice.MYQLM_PYLINALG,
            # AWSDevice.BRAKET_LOCAL_SIMULATOR,
            # GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        ],
    )
    br2 = run(
        c2,
        [
            IBMDevice.AER_SIMULATOR,
            # ATOSDevice.MYQLM_PYLINALG,
            # AWSDevice.BRAKET_LOCAL_SIMULATOR,
            # GOOGLEDevice.CIRQ_LOCAL_SIMULATOR,
        ],
    )

    assert isinstance(br1, BatchResult)
    assert isinstance(br2, BatchResult)
    for r1, r2 in zip(br1.results, br2.results):
        assert isinstance(r1.expectation_values, dict)
        assert isinstance(r2.expectation_values, dict)
        assert r1.expectation_values.keys() == r2.expectation_values.keys()
        for k in r1.expectation_values:
            assert np.isclose(r1.expectation_values[k], r2.expectation_values[k])
