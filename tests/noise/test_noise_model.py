import pytest
from braket.circuits.noises import Depolarizing as BraketDepolarizing
from qat.quops.quantum_channels import QuantumChannelKraus

from mpqp.core.languages import Language
from mpqp.gates import *
from mpqp.noise import Depolarizing
from mpqp.noise.noise_model import NoiseModel


def f():
    assert True


def test_depolarizing_valid_prob():
    noise = Depolarizing(0.1, [0])
    assert noise.proba == 0.1


@pytest.mark.parametrize("prob", [-0.1, 1.5])
def test_depolarizing_wrong_prob(prob: float):
    with pytest.raises(
        ValueError, match=f"Invalid probability: {prob} must have been between 0 and 1"
    ):
        Depolarizing(prob, [0], dimension=1)


@pytest.mark.parametrize(
    "args, error",
    [
        (-0.1, [0], 1, "Invalid probability: -0.1 must have been between 0 and 1"),
        (1.5, [0], 1, "Invalid probability: 1.5 must have been between 0 and 1"),
        (0.1, [0], 2, "Invalid probability: -0.1 must have been between 0 and 1"),
        (0.1, [0], 0, "Invalid probability: -0.1 must have been between 0 and 1"),
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
