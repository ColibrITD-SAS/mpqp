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
from mpqp.noise import Depolarizing, BitFlip, AmplitudeDamping
from mpqp.noise import Depolarizing
from mpqp.tools.theoretical_simulation import (
    chisquare_test,
    process_qcircuit,
    results_to_dict,
    run_experiment,
)
from mpqp.tools.errors import UnsupportedBraketFeaturesWarning


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


@pytest.fixture
def chisquare_test_data() -> tuple[list[int], list[int], int, float]:
    noise_proba = 0.7
    shots = 1024
    alpha = 0.05

    circuit = QCircuit(
        [
            H(0),
            CNOT(0, 1),
            BasisMeasure([0, 1], shots=1024),
            Depolarizing(noise_proba, [0, 1]),
        ],
        label="Noise-Testing",
    )
    initial_state = [1, 0, 0, 0]

    # mpqp experiement
    run_mpqp = _run_single(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR, {})
    mpqp_counts = run_mpqp.counts

    # theoretical experiement
    gate_map, gates = process_qcircuit(circuit)
    run_theoretical = run_experiment(initial_state, gates, gate_map, noise_proba, shots)

    num_qubits = circuit.nb_qubits
    results_dict = results_to_dict(run_theoretical, num_qubits, shots)
    theoretical_counts = list(results_dict.values())

    return mpqp_counts, theoretical_counts, shots, alpha


def test_noisy_expectation_value_execution_without_error(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(
        ExpectationMeasure(
            [0, 1, 2],
            observable=Observable(np.diag([4, 1, 2, 3, 6, 3, 4, 5])),
            shots=1023,
        )
    )
    circuit.add(Depolarizing(0.23, [0, 1]))
    circuit.add(BitFlip(0.1))
    circuit.add(AmplitudeDamping(0.4))
    circuit.add(AmplitudeDamping(0.2, 0.3))
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(circuit, devices)
    assert True


def test_all_native_gates_global_noise_execution_without_error(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(BasisMeasure([0, 1, 2], shots=1023))
    circuit.add(Depolarizing(0.23))
    circuit.add(Depolarizing(0.23, [0, 1, 2], dimension=2, gates=[SWAP, CNOT, CZ]))
    circuit.add(BitFlip(0.2, [0, 1, 2]))
    circuit.add(BitFlip(0.1, gates=[CNOT, H]))
    circuit.add(AmplitudeDamping(0.4, gates=[CNOT, H]))
    circuit.add(AmplitudeDamping(0.2, 0.3, [0, 1, 2]))
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(circuit, devices)
    assert True


def test_all_native_gates_local_noise(
    circuit: QCircuit, devices: list[AvailableDevice]
):
    circuit.add(BasisMeasure([0, 1, 2], shots=1023))
    circuit.add(
        Depolarizing(0.23, [0, 2], gates=[H, X, Y, Z, S, T, Rx, Ry, Rz, Rk, P, U])
    )
    circuit.add(Depolarizing(0.23, [0, 1], dimension=2, gates=[SWAP, CNOT, CZ]))
    circuit.add(BitFlip(0.2, [0, 2]))
    circuit.add(BitFlip(0.1, [0, 1], gates=[CNOT, H]))
    circuit.add(AmplitudeDamping(0.4, targets=[0, 1], gates=[CNOT, H]))
    circuit.add(AmplitudeDamping(0.2, 0.3, [0, 1, 2]))
    with pytest.warns(UnsupportedBraketFeaturesWarning):
        run(circuit, devices)
    assert True


def test_chisquare_expected_counts_calculation(
    chisquare_test_data: tuple[list[int], list[int], int, float]
):
    mpqp_counts, theoretical_counts, shots, _ = chisquare_test_data
    result = chisquare_test(mpqp_counts, theoretical_counts, shots)

    theoretical_probabilities = [count / shots for count in theoretical_counts]
    expected_counts = [int(tp * shots) for tp in theoretical_probabilities]

    assert result.expected_counts == expected_counts


def test_chisquare_p_value_calculation(
    chisquare_test_data: tuple[list[int], list[int], int, float]
):
    from scipy.stats import chisquare

    mpqp_counts, theoretical_counts, shots, _ = chisquare_test_data
    result = chisquare_test(mpqp_counts, theoretical_counts, shots)

    _, expected_p_value = chisquare(mpqp_counts, result.expected_counts)
    assert result.p_value == expected_p_value


def test_chisquare_significant(
    chisquare_test_data: tuple[list[int], list[int], int, float]
):
    mpqp_counts, theoretical_counts, shots, alpha = chisquare_test_data

    result = chisquare_test(mpqp_counts, theoretical_counts, shots, alpha)

    assert result.p_value < alpha


def test_chisquare_empty_counts():
    mpqp_counts = []
    theoretical_counts = []
    shots = 1

    result = chisquare_test(mpqp_counts, theoretical_counts, shots)

    assert result.p_value == 1.0
    assert not result.significant, "Result should not be significant when empty counts"
