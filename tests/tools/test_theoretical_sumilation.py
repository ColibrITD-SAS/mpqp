import numpy as np
import numpy.typing as npt
import pytest

from mpqp import QCircuit
from mpqp.core.instruction.measurement import BasisMeasure
from mpqp.gates import *
from mpqp.tools.theoretical_simulation import theoretical_probs, validate


@pytest.mark.parametrize(
    "experiment, theoretical_probs, alpha",
    [
        (
            [165, 145, 195, 37, 114, 185, 78, 61],
            np.array([0.168, 0.147, 0.198, 0.037, 0.116, 0.188, 0.079, 0.062]),
            0.0013,
        )
    ],
)
def test_validation_success(
    experiment: list[int], theoretical_probs: npt.NDArray[np.float32], alpha: float
):
    assert validate(experiment, theoretical_probs, alpha)


@pytest.mark.parametrize(
    "experiment, theoretical_probs, alpha",
    [
        (
            [165, 145, 195, 37, 114, 185, 78, 61],
            np.array([0.168, 0.147, 0.198, 0.037, 0.116, 0.188, 0.079, 0.062]),
            0.0012,
        )
    ],
)
def test_validation_failure(
    experiment: list[int], theoretical_probs: npt.NDArray[np.float32], alpha: float
):
    assert not validate(experiment, theoretical_probs, alpha)


def test_simulation():
    circuit = QCircuit(
        [
            H(0),
            CNOT(0, 1),
            BasisMeasure([0, 1], shots=1024),
        ],
        label="Noise-Testing",
    )

    assert np.allclose(np.array([0.5, 0, 0, 0.5]), theoretical_probs(circuit, 0))
