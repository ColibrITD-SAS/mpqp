import contextlib
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.barrier import Barrier
from mpqp.core.instruction.breakpoint import Breakpoint
from mpqp.core.instruction.gates.native_gates import NATIVE_GATES
from mpqp.core.instruction.instruction import Instruction
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.core.instruction.measurement.measure import Measure
from mpqp.core.instruction.measurement.pauli_string import I as Ip
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.core.instruction.measurement.pauli_string import X as Xp
from mpqp.core.instruction.measurement.pauli_string import Y as Yp
from mpqp.core.instruction.measurement.pauli_string import Z as Zp
from mpqp.core.languages import Language
from mpqp.execution import (
    ATOSDevice,
    AvailableDevice,
    AWSDevice,
    AZUREDevice,
    GOOGLEDevice,
    IBMDevice,
    run,
)
from mpqp.execution.result import BatchResult, Result
from mpqp.gates import *
from mpqp.measures import (
    Basis,
    BasisMeasure,
    ComputationalBasis,
    ExpectationMeasure,
    HadamardBasis,
    Observable,
)
from mpqp.noise.noise_model import NOISE_MODELS, Depolarizing, PhaseDamping
from mpqp.tools import Matrix, atol, rand_hermitian_matrix, rtol
from mpqp.tools.circuit import random_gate, random_noise
from mpqp.tools.errors import (
    DeviceJobIncompatibleError,
    UnsupportedBraketFeaturesWarning,
)
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
        [H(0), CNOT(0, 1), CNOT(1, 2)],
        [CustomGate(UnitaryMatrix(np.array([[0, 1], [1, 0]])), [1])],
        [U(0.215, 0.5588, 8, 1)],
    ],
)
def test_sample_counts_in_trust_interval(instructions: list[Gate]):
    c = QCircuit(instructions)
    shots = 50000
    err_rate = 0.2
    err_rate_percentage = 1 - np.power(1 - err_rate, (1 / 2))
    res = run(c, state_vector_devices[0])
    assert isinstance(res, Result)
    expected_counts = [int(count) for count in np.round(shots * res.probabilities)]
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=shots))
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)
    for result in batch:
        print(result)
        print("expected_counts: " + str(expected_counts))
        counts = result.counts
        # check if the true value is inside the trust interval
        for i in range(len(counts)):
            trust_interval = np.ceil(
                err_rate_percentage * expected_counts[i] + shots / 15
            )
            print(trust_interval)
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
    c.add(ExpectationMeasure(Observable(observable), list(range(c.nb_qubits))))
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


@pytest.mark.parametrize(
    "device",
    list(IBMDevice)
    + list(ATOSDevice)
    + list(AWSDevice)
    + list(GOOGLEDevice)
    + list(AZUREDevice),
)
def test_validity_run_job_type(device: AvailableDevice, circuits_type: list[QCircuit]):
    circuit_state_vector = circuits_type[0]
    circuit_samples = circuits_type[1]
    circuit_observable = circuits_type[2]
    circuit_observable_ideal = circuits_type[3]

    if not device.is_remote():
        if device.supports_samples():
            with (
                pytest.warns(UnsupportedBraketFeaturesWarning)
                if isinstance(device, AWSDevice)
                else contextlib.suppress()
            ):
                assert run(circuit_samples, device) is not None
        else:
            with pytest.raises(NotImplementedError):
                run(circuit_samples, device)

        if device.supports_state_vector():
            with (
                pytest.warns(UnsupportedBraketFeaturesWarning)
                if isinstance(device, AWSDevice)
                else contextlib.suppress()
            ):
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
                with (
                    pytest.warns(UnsupportedBraketFeaturesWarning)
                    if isinstance(device, AWSDevice)
                    else contextlib.suppress()
                ):
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
                with (
                    pytest.warns(UnsupportedBraketFeaturesWarning)
                    if isinstance(device, AWSDevice)
                    else contextlib.suppress()
                ):
                    assert run(circuit_observable_ideal, device) is not None
        else:
            if isinstance(device, IBMDevice):
                with pytest.raises(DeviceJobIncompatibleError):
                    run(circuit_observable, device)
            else:
                with pytest.raises(NotImplementedError):
                    run(circuit_observable, device)


@pytest.mark.parametrize("language", list(Language))
def test_validity_native_gate_to_other_language(language: Language):
    for gate in NATIVE_GATES:
        gate_build = random_gate([gate])

        if language in [Language.MY_QLM, Language.CIRQ, Language.QASM3]:
            with pytest.raises(NotImplementedError):
                gate_build.to_other_language(language)
        else:
            assert gate_build.to_other_language(language) is not None


@pytest.fixture
def measures():
    return [
        BasisMeasure([0, 1]),
        BasisMeasure(
            [0, 1], shots=1024, basis=Basis([np.array([1, 0]), np.array([0, -1])])
        ),
        BasisMeasure([0, 1], shots=1024, basis=ComputationalBasis(3)),
        BasisMeasure([0, 1], shots=1024, basis=HadamardBasis(2)),
        ExpectationMeasure(Observable(np.diag([0.7, -1, 1, 1])), shots=10),
    ]


@pytest.mark.parametrize("language", list(Language))
def test_validity_measure_to_other_language(
    language: Language, measures: list[Measure]
):
    for measure in measures:
        if isinstance(measure, ExpectationMeasure):
            with pytest.raises(NotImplementedError):
                measure.to_other_language(language)
        elif language in [
            Language.MY_QLM,
            Language.CIRQ,
            Language.BRAKET,
            Language.QASM3,
        ]:
            with pytest.raises(NotImplementedError):
                measure.to_other_language(language)
        else:
            assert measure.to_other_language(language) is not None


@pytest.fixture
def pauli_strings():
    return [Ip @ Xp @ Yp @ Zp, Xp + Zp, Yp]


@pytest.mark.parametrize("language", list(Language))
def test_validity_pauli_string_to_other_language(
    language: Language, pauli_strings: list[PauliString]
):

    for pauli_string in pauli_strings:
        if language in [Language.QASM3, Language.QASM2]:
            with pytest.raises(NotImplementedError):
                pauli_string.to_other_language(language)
        else:
            assert pauli_string.to_other_language(language) is not None


@pytest.mark.parametrize("language", list(Language))
def test_validity_noise_to_other_language(language: Language):
    for noise in NOISE_MODELS:
        noise_build = random_noise([noise])

        if language in [Language.CIRQ, Language.QASM3, Language.QASM2]:
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


@pytest.mark.parametrize("language", list(Language))
def test_validity_other_instr_to_other_language(
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
