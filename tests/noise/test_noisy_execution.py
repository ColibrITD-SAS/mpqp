"""Add ``--long`` to the cli args to run this test (disabled by default because 
too slow)"""

import sys
from itertools import product
from typing import Any, Callable, Iterable

import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import (
    BasisMeasure,
    ExpectationMeasure,
    Observable,
)
from mpqp.execution import (
    ATOSDevice,
    AvailableDevice,
    AWSDevice,
    GOOGLEDevice,
    IBMDevice,
    run,
)
from mpqp.gates import *
from mpqp.noise import AmplitudeDamping, BitFlip, Depolarizing, PhaseDamping
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning
from mpqp.tools.theoretical_simulation import validate_noisy_circuit

noisy_devices: list[Any] = [
    dev
    for dev in list(ATOSDevice) + list(AWSDevice) + list(IBMDevice) + list(GOOGLEDevice)
    if dev.is_noisy_simulator()
]
# TODO: in the end this should be automatic as drafted above, but for now only
# one device is stable
noisy_devices = [AWSDevice.BRAKET_LOCAL_SIMULATOR, IBMDevice.AER_SIMULATOR]


def filter_braket_warning(
    action: Callable[[AvailableDevice], Any],
    devices: AvailableDevice,
):
    if (
        isinstance(devices, Iterable)
        and any(isinstance(device, AWSDevice) for device in devices)
    ) or isinstance(devices, AWSDevice):
        with pytest.warns((UnsupportedBraketFeaturesWarning)):
            return action(devices)
    else:
        return action(devices)


@pytest.fixture
def circuit():
    return QCircuit(
        [
            H(0),
            X(1),
            Y(2),
            Z(0),
            S(1),
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
def devices():
    devices: list[AvailableDevice] = [
        AWSDevice.BRAKET_LOCAL_SIMULATOR,
        IBMDevice.AER_SIMULATOR,
        IBMDevice.AER_SIMULATOR_STATEVECTOR,
        IBMDevice.AER_SIMULATOR_MATRIX_PRODUCT_STATE,
        IBMDevice.AER_SIMULATOR_DENSITY_MATRIX,
    ]
    if "--long" in sys.argv:
        devices.append(ATOSDevice.QLM_NOISYQPROC)
    return devices


def test_noisy_expectation_value_execution_without_error(
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
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(circuit, devices)
    assert True


def test_all_native_gates_global_noise_execution_without_error(
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
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(circuit, devices)
    assert True


def test_all_native_gates_local_noise(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(
        [
            BasisMeasure([0, 1, 2], shots=1023),
            Depolarizing(0.23, [0, 2], gates=[H, X, Y, Z, S, T, Rx, Ry, Rz, Rk, P, U]),
            Depolarizing(0.23, [0, 1], dimension=2, gates=[SWAP, CNOT, CZ]),
            BitFlip(0.2, [0, 2]),
            BitFlip(0.1, [0, 1], gates=[CNOT, H]),
            AmplitudeDamping(0.4, targets=[0, 1], gates=[CNOT, H]),
            AmplitudeDamping(0.2, 0.3, [0, 1, 2]),
            PhaseDamping(0.4, [0, 1, 2], gates=[CNOT, H]),
        ]
    )
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(circuit, devices)
    assert True


@pytest.mark.parametrize(
    "depol_noise, shots, device",
    product(
        [0.001, 0.01, 0.1, 0.1, 0.2, 0.3],
        [500, 1_000, 5_000, 10_000, 50_000, 100_000],
        noisy_devices,
    ),
)
def test_validate_depolarizing_noise(
    circuit: QCircuit, depol_noise: float, shots: int, device: AvailableDevice
):
    circuit.add(Depolarizing(depol_noise))
    assert filter_braket_warning(
        lambda d: validate_noisy_circuit(circuit, shots, d), device
    )
