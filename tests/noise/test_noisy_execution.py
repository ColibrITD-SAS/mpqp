"""Add ``--long`` to the cli args to run this test (disabled by default because
too slow)"""

import sys
from itertools import product

import numpy as np
import pytest

from mpqp import (
    AmplitudeDamping,
    ATOSDevice,
    AWSDevice,
    BasisMeasure,
    BitFlip,
    Depolarizing,
    ExpectationMeasure,
    IBMDevice,
    Observable,
    PhaseDamping,
    QCircuit,
    run,
)
from mpqp.execution import AvailableDevice
from mpqp.gates import *
from mpqp.noise import AmplitudeDamping, BitFlip, Depolarizing, PhaseDamping
from mpqp.tools.theoretical_simulation import validate_noisy_circuit

# noisy_devices: list[Any] = [
#     dev
#     for dev in list(ATOSDevice) + list(AWSDevice) + list(IBMDevice) + list(GOOGLEDevice)
#     if dev.is_noisy_simulator()
# ]
# TODO: in the end this should be automatic as drafted above, but for now only
# one device is stable
noisy_devices_Braket = [AWSDevice.BRAKET_LOCAL_SIMULATOR]
noisy_devices_qiskit = [IBMDevice.AER_SIMULATOR]


@pytest.fixture
def circuit():
    return QCircuit(
        [
            H(0),
            X(1),
            Y(2),
            Z(0),
            S(0),
            S_dagger(1),
            T(0),
            Rx(1.2324, 2),
            Ry(-2.43, 0),
            Rz(1.04, 1),
            Rk(-1, 0),
            P(-323, 2),
            U(1.2, 2.3, 3.4, 2),
            SWAP(2, 1),
            CNOT(0, 1),
            CZ(1, 2),
            # CRk(4, 2, 0)
        ]
    )


@pytest.fixture
def devices_myqlm() -> list[AvailableDevice]:
    devices = []
    if "--long" in sys.argv:
        devices.append(ATOSDevice.QLM_NOISYQPROC)
    return devices


@pytest.fixture
def devices_braket() -> list[AvailableDevice]:
    return [
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
    ]


@pytest.fixture
def devices_IBMDevice() -> list[AvailableDevice]:
    return [
        IBMDevice.AER_SIMULATOR,
        IBMDevice.AER_SIMULATOR_STATEVECTOR,
        IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
        IBMDevice.AER_SIMULATOR_DENSITY_MATRIX,
    ]


@pytest.mark.provider("myqlm")
def test_exec_noisy_expectation_value_execution_without_error_myqlm(
    circuit: QCircuit, devices_myqlm: list[AvailableDevice]
):
    exec_noisy_expectation_value_execution_without_error(circuit, devices_myqlm)


@pytest.mark.provider("braket")
def test_exec_noisy_expectation_value_execution_without_error_braket(
    circuit: QCircuit, devices_braket: list[AvailableDevice]
):
    exec_noisy_expectation_value_execution_without_error(circuit, devices_braket)


@pytest.mark.provider("qiskit")
def test_exec_noisy_expectation_value_execution_without_error_IBMDevice(
    circuit: QCircuit, devices_IBMDevice: list[AvailableDevice]
):
    exec_noisy_expectation_value_execution_without_error(circuit, devices_IBMDevice)


def exec_noisy_expectation_value_execution_without_error(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(
        [
            ExpectationMeasure(
                Observable(np.diag([4, 1, 2, 3, 6, 3, 4, 5])), shots=1023
            ),
            Depolarizing(0.23, [0, 1]),
            BitFlip(0.1),
            AmplitudeDamping(0.4),
            AmplitudeDamping(0.2, 0.3),
            PhaseDamping(0.6),
        ]
    )
    run(circuit, devices)
    assert True


@pytest.mark.provider("myqlm")
def test_all_native_gates_global_noise_execution_without_error_myqlm(
    circuit: QCircuit, devices_myqlm: list[AvailableDevice]
):
    exec_all_native_gates_global_noise_execution_without_error(circuit, devices_myqlm)


@pytest.mark.provider("braket")
def test_all_native_gates_global_noise_execution_without_error_braket(
    circuit: QCircuit, devices_braket: list[AvailableDevice]
):
    exec_all_native_gates_global_noise_execution_without_error(circuit, devices_braket)


@pytest.mark.provider("qiskit")
def test_all_native_gates_global_noise_execution_without_error_IBMDevice(
    circuit: QCircuit, devices_IBMDevice: list[AvailableDevice]
):
    exec_all_native_gates_global_noise_execution_without_error(
        circuit, devices_IBMDevice
    )


def exec_all_native_gates_global_noise_execution_without_error(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(
        [
            BasisMeasure([0, 1, 2], shots=1023),
            Depolarizing(0.23),
            Depolarizing(0.23, [0, 1, 2], dimension=2, gates=[SWAP, CNOT, CZ]),
            BitFlip(0.2, [0, 1, 2]),
            BitFlip(0.1, gates=[CNOT, H]),
            AmplitudeDamping(0.4, gates=[CNOT, H]),
            AmplitudeDamping(0.2, 0.3, [0, 1, 2]),
            PhaseDamping(0.4, [0, 2]),
            PhaseDamping(0.4, gates=[CNOT, H]),
        ]
    )
    run(circuit, devices)
    assert True


@pytest.mark.provider("myqlm")
def test_all_native_gates_local_noise_myqlm(
    circuit: QCircuit, devices_myqlm: list[AvailableDevice]
):
    exec_all_native_gates_local_noise(circuit, devices_myqlm)


@pytest.mark.provider("braket")
def test_all_native_gates_local_noise_braket(
    circuit: QCircuit, devices_braket: list[AvailableDevice]
):
    exec_all_native_gates_local_noise(circuit, devices_braket)


@pytest.mark.provider("qiskit")
def test_all_native_gates_local_noise_IBMDevice(
    circuit: QCircuit, devices_IBMDevice: list[AvailableDevice]
):
    exec_all_native_gates_local_noise(circuit, devices_IBMDevice)


def exec_all_native_gates_local_noise(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(
        [
            BasisMeasure([0, 1, 2], shots=1023),
            Depolarizing(
                0.23, [0, 2], gates=[H, X, Y, Z, S, S_dagger, T, Rx, Ry, Rz, Rk, P, U]
            ),
            Depolarizing(0.23, [0, 1], dimension=2, gates=[SWAP, CNOT, CZ]),
            BitFlip(0.2, [0, 2]),
            BitFlip(0.1, [0, 1], gates=[CNOT, H]),
            AmplitudeDamping(0.4, targets=[0, 1], gates=[CNOT, H]),
            AmplitudeDamping(0.2, 0.3, [0, 1, 2]),
            PhaseDamping(0.4, [0, 1, 2], gates=[CNOT, H]),
        ]
    )
    run(circuit, devices)
    assert True


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "depol_noise, shots, device",
    list(
        product(
            [0.001, 0.01, 0.1, 0.1, 0.2, 0.3],
            [500, 1_000, 5_000, 10_000, 50_000, 100_000],
            noisy_devices_Braket,
        )
    ),
)
def test_validate_depolarizing_noise_braket(
    circuit: QCircuit, depol_noise: float, shots: int, device: AvailableDevice
):
    circuit.add(Depolarizing(depol_noise))
    validate_noisy_circuit(circuit, shots, device)


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "depol_noise, shots, device",
    list(
        product(
            [0.001, 0.01, 0.1, 0.1, 0.2, 0.3],
            [500, 1_000, 5_000, 10_000, 50_000, 100_000],
            noisy_devices_qiskit,
        )
    ),
)
def test_validate_depolarizing_noise_qiskit(
    circuit: QCircuit, depol_noise: float, shots: int, device: AvailableDevice
):
    circuit.add(Depolarizing(depol_noise))
    assert validate_noisy_circuit(circuit, shots, device)
