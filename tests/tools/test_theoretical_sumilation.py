import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.execution import AWSDevice
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.gates import *
from mpqp.noise import Depolarizing
from mpqp.tools.theoretical_simulation import chi_square_test, run_experiment


@pytest.fixture
def chi_square_test_data() -> tuple[list[int], list[int], int, float]:
    noise_prob = 0.7
    shots = 1024
    alpha = 0.05

    circuit = QCircuit(
        [
            H(0),
            CNOT(0, 1),
            BasisMeasure([0, 1], shots=1024),
            Depolarizing(noise_prob, [0, 1]),
        ],
        label="Noise-Testing",
    )

    run_mpqp = _run_single(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR, {})
    mpqp_counts = run_mpqp.counts

    run_theoretical = run_experiment(circuit, noise_prob, shots)
    theoretical_counts = list(run_theoretical.values())

    return mpqp_counts, theoretical_counts, shots, alpha


def test_chi_square_significant(
    chi_square_test_data: tuple[list[int], list[int], int, float]
):
    mpqp_counts, theoretical_counts, shots, alpha = chi_square_test_data

    result = chi_square_test(mpqp_counts, theoretical_counts, shots, alpha)

    assert result.p_value < alpha


def test_chi_square_empty_counts():
    mpqp_counts = []
    theoretical_counts = []
    shots = 1

    result = chi_square_test(mpqp_counts, theoretical_counts, shots)

    assert result.p_value == 1.0
    assert not result.significant, "Result should not be significant when empty counts"
