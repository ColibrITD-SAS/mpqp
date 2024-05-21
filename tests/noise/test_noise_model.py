import pytest
from braket.circuits.noises import Depolarizing as BraketDepolarizing
from qat.quops.quantum_channels import QuantumChannelKraus

from mpqp.core.languages import Language
from mpqp.gates import *
from mpqp.noise import Depolarizing
from mpqp.noise.noise_model import NoiseModel


def test_depolarizing_valid_params():
    noise = Depolarizing(0.1, [0])
    assert noise.proba == 0.1


@pytest.mark.parametrize(
    "args, error",
    [
        (
            (-0.1, [0], 1),
            "Invalid probability: -0.1 but should have been between 0 and 1.",
        ),
        (
            (1.5, [0], 1),
            "Invalid probability: 1.5 but should have been between 0 and 1.",
        ),
        (
            (0.1, [0], 2),
            "Number of target qubits 1 should be higher than the dimension 2.",
        ),
        (
            (0.1, [0], 0),
            "Dimension of the depolarizing channel must be strictly greater "
            "than 1, but got 0 instead.",
        ),
    ],
)
def test_depolarizing_wrong_params(args: tuple[float, list[int], int], error: str):
    with pytest.raises(ValueError, match=error):
        Depolarizing(*args)


@pytest.fixture
def noise():
    return Depolarizing(0.3, [0])


def test_depolarizing_braket_export(noise: NoiseModel):
    braket_noise = noise.to_other_language(Language.BRAKET)
    assert isinstance(braket_noise, BraketDepolarizing)
    assert braket_noise.probability == 0.3
    assert braket_noise.qubit_count == 1


def test_depolarizing_qlm_export(noise: NoiseModel):
    qlm_noise = noise.to_other_language(Language.MY_QLM)
    assert isinstance(qlm_noise, QuantumChannelKraus)
