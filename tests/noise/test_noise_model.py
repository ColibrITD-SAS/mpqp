import pytest
from braket.circuits.noises import Depolarizing as BraketDepolarizing
from qat.quops.quantum_channels import QuantumChannelKraus

from mpqp.core.languages import Language
from mpqp.gates import *
from mpqp.noise import Depolarizing


def f():
    assert True


def test_depolarizing_valid_prob():
    noise = Depolarizing(0.1, [0])
    assert noise.proba == 0.1


def test_depoalrizing_negative_prob():
    with pytest.raises(
        ValueError, match="Invalid probability: -0.1 must have been between 0 and 1"
    ):
        Depolarizing(-0.1, [0], dimension=1)


def test_depoalrizing_invalid_prob():
    with pytest.raises(
        ValueError, match="Invalid probability: 1.5 must have been between 0 and 1"
    ):
        Depolarizing(1.5, [0])


def test_depolarizing_invalid_targets():
    with pytest.raises(
        ValueError,
        match="Number of target qubits 1 should be higher than the dimension 2.",
    ):
        Depolarizing(0.1, [0], dimension=2)


def test_depoalrizing_invalid_dimension():
    with pytest.raises(
        ValueError,
        match="Dimension of the depolarizing channel must be strictly greater than 1, but got 0 instead.",
    ):
        Depolarizing(0.0, [0], dimension=0)


@pytest.mark.parametrize("language", [Language.BRAKET, Language.MY_QLM])
def test_depolarizing_to_other_language_supported(language):
    noise = Depolarizing(0.3, [0])

    if language == Language.BRAKET:
        braket_noise = noise.to_other_language(language)
        assert isinstance(braket_noise, BraketDepolarizing)
        assert braket_noise.probability == 0.3
        assert braket_noise.qubit_count == 1

    elif language == Language.MY_QLM:
        qlm_noise = noise.to_other_language(Language.MY_QLM)
        assert isinstance(qlm_noise, QuantumChannelKraus)
