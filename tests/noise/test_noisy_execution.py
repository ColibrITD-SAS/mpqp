"""Add ``--long`` to the cli args to run this test (disabled by default because 
too slow)"""

import sys

import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import (
    BasisMeasure,
    ExpectationMeasure,
    Observable,
)
from mpqp.execution import ATOSDevice, AvailableDevice, AWSDevice, run
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.gates import *
from mpqp.noise import AmplitudeDamping, BitFlip, Depolarizing
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning
from mpqp.tools.theoretical_simulation import chi_square_test, run_experiment


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
    devices: list[AvailableDevice] = [AWSDevice.BRAKET_LOCAL_SIMULATOR]
    if "--long" in sys.argv:
        devices.append(ATOSDevice.QLM_NOISYQPROC)
    return devices


def test_noisy_expectation_value_execution_without_error(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(
        [
            ExpectationMeasure(
                [0, 1, 2],
                observable=Observable(np.diag([4, 1, 2, 3, 6, 3, 4, 5])),
                shots=1023,
            ),
            Depolarizing(0.23, [0, 1]),
            BitFlip(0.1),
            AmplitudeDamping(0.4),
            AmplitudeDamping(0.2, 0.3),
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
        ]
    )
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(circuit, devices)
    assert True


def test_shot_noise(circuit: QCircuit, devices: list[AvailableDevice]):
    shots = 1024
    circuit.add([BasisMeasure()])
    result = _run_single(circuit, devices[0], {})
    experimental_counts = result.counts

    theoretical_counts = run_experiment(circuit, 0, shots)

    r = chi_square_test(experimental_counts, list(theoretical_counts.values()), shots)
    assert r.significant


def test_depol_noise(circuit: QCircuit, devices: list[AvailableDevice]):
    shots = 1024
    depol_noise = 0.3
    circuit.add([BasisMeasure(), Depolarizing(depol_noise)])
    result = _run_single(circuit, devices[0], {})
    experimental_counts = result.counts

    theoretical_counts = run_experiment(circuit, depol_noise, shots)

    r = chi_square_test(experimental_counts, list(theoretical_counts.values()), shots)
    assert r.significant
