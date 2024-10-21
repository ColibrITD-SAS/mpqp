import pytest

from mpqp.core.circuit import QCircuit
from mpqp.core.instruction.gates.native_gates import NativeGate
from mpqp.core.languages import Language
from mpqp.execution.providers.ibm import generate_qiskit_noise_model
from mpqp.gates import *
from mpqp.noise import AmplitudeDamping, BitFlip, Depolarizing, NoiseModel, PhaseDamping


def test_depolarizing_valid_params():
    noise = Depolarizing(0.1, [0])
    assert noise.prob == 0.1


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
            "Dimension of a multi-dimensional NoiseModel must be strictly greater"
            " than 1, but got 0 instead.",
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
    from braket.circuits.noises import Depolarizing as BraketDepolarizing

    braket_noise = noise.to_other_language(Language.BRAKET)
    assert isinstance(braket_noise, BraketDepolarizing)
    assert braket_noise.probability == 0.3
    assert braket_noise.qubit_count == 1


def test_depolarizing_qlm_export(noise: NoiseModel):
    from qat.quops.quantum_channels import QuantumChannelKraus

    qlm_noise = noise.to_other_language(Language.MY_QLM)
    assert isinstance(qlm_noise, QuantumChannelKraus)


@pytest.fixture
def circuit():
    return QCircuit([H(0), CNOT(0, 1), SWAP(1, 2), Z(2)])


@pytest.mark.parametrize(
    "prob, targets, dimension, gates, expected_noisy_gates",
    [
        (0.3, None, 1, None, ["cx", "z", "swap", "h"]),
        (0.3, [0, 1, 2], 1, None, ["cx", "z", "swap", "h"]),
        (0.3, None, 1, [H, Z], ["z", "h"]),
        (0.3, [0, 1, 2], 1, [H, Z], ["z", "h"]),
        (0.3, [0, 1], 1, None, ["noisy_identity_0", "cx", "h"]),
        (0.3, [0, 1], 1, [H, Z], ["h"]),
        (0.3, None, 2, None, ["swap", "cx"]),
        (0.3, [0, 1, 2], 2, None, ["swap", "cx"]),
        (0.3, None, 2, [CNOT, SWAP], ["swap", "cx"]),
        (0.3, [0, 1, 2], 2, [CNOT, SWAP], ["swap", "cx"]),
        (0.3, [0, 1], 2, None, ["cx"]),
        (0.3, [1, 2], 2, None, ["swap"]),
    ],
)
def test_depolarizing_qiskit_export(
    circuit: QCircuit,
    prob: float,
    targets: list[int],
    dimension: int,
    gates: list[type[NativeGate]],
    expected_noisy_gates: list[str],
):
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_aer.noise.errors.standard_errors import depolarizing_error

    noise = Depolarizing(prob=prob, targets=targets, dimension=dimension, gates=gates)

    assert isinstance(circuit, QCircuit)
    circuit.add(noise)

    qiskit_error = noise.to_other_language(Language.QISKIT)
    expected_error = depolarizing_error(prob, dimension)

    qiskit_noise_model, _ = generate_qiskit_noise_model(circuit)
    noisy_instructions = qiskit_noise_model.noise_instructions

    assert qiskit_error == expected_error
    assert isinstance(qiskit_noise_model, Qiskit_NoiseModel)
    assert sorted(noisy_instructions) == sorted(expected_noisy_gates)


@pytest.mark.parametrize(
    "prob, targets, gates, expected_noisy_gates",
    [
        (0.3, None, None, ["h", "z", "swap", "cx"]),
        (0.3, [0, 1, 2], None, ["h", "z", "swap", "cx"]),
        (0.3, None, [H, CNOT, SWAP, Z], ["h", "z", "swap", "cx"]),
        (0.3, [0, 1, 2], [CNOT, SWAP], ["cx", "swap"]),
        (0.3, [0, 1], None, ["h", "noisy_identity_0", "cx"]),
        (0.3, [0, 1], [H, CNOT, SWAP], ["h", "noisy_identity_0", "cx"]),
    ],
)
def test_bitflip_qiskit_export(
    circuit: QCircuit,
    prob: float,
    targets: list[int],
    gates: list[type[NativeGate]],
    expected_noisy_gates: list[str],
):
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_aer.noise.errors.standard_errors import pauli_error

    noise = BitFlip(prob=prob, targets=targets, gates=gates)

    assert isinstance(circuit, QCircuit)
    circuit.add(noise)

    qiskit_error = noise.to_other_language(Language.QISKIT)
    expected_error = pauli_error([("X", prob), ("I", 1 - prob)])

    qiskit_noise_model, _ = generate_qiskit_noise_model(circuit)
    noisy_instructions = qiskit_noise_model.noise_instructions

    assert qiskit_error == expected_error
    assert isinstance(qiskit_noise_model, Qiskit_NoiseModel)
    assert sorted(noisy_instructions) == sorted(expected_noisy_gates)


@pytest.mark.parametrize(
    "gamma, prob, targets, gates, expected_noisy_gates",
    [
        (0.3, 1.0, None, None, ["h", "z", "swap", "cx"]),
        (0.3, 0.1, [0, 1, 2], None, ["h", "z", "swap", "cx"]),
        (0.3, 0.1, None, [H, CNOT, SWAP, Z], ["h", "z", "swap", "cx"]),
        (0.3, 0.1, [0, 1, 2], [CNOT, SWAP], ["cx", "swap"]),
        (0.3, 0.1, [0, 1], None, ["h", "noisy_identity_0", "cx"]),
        (0.3, 0.1, [0, 1], [H, CNOT, SWAP], ["h", "noisy_identity_0", "cx"]),
    ],
)
def test_amplitudedamping_qiskit_export(
    circuit: QCircuit,
    gamma: float,
    prob: float,
    targets: list[int],
    gates: list[type[NativeGate]],
    expected_noisy_gates: list[str],
):
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_aer.noise.errors.standard_errors import amplitude_damping_error

    noise = AmplitudeDamping(gamma=gamma, prob=prob, targets=targets, gates=gates)

    assert isinstance(circuit, QCircuit)
    circuit.add(noise)

    qiskit_error = noise.to_other_language(Language.QISKIT)
    expected_error = amplitude_damping_error(
        gamma, 1 - prob  # pyright: ignore[reportArgumentType]
    )

    qiskit_noise_model, _ = generate_qiskit_noise_model(circuit)
    noisy_instructions = qiskit_noise_model.noise_instructions

    assert qiskit_error == expected_error
    assert isinstance(qiskit_noise_model, Qiskit_NoiseModel)
    assert sorted(noisy_instructions) == sorted(expected_noisy_gates)


@pytest.mark.parametrize(
    "gamma, targets, gates, expected_noisy_gates",
    [
        (0.3, None, None, ["h", "z", "swap", "cx"]),
        (0.3, [0, 1, 2], None, ["h", "z", "swap", "cx"]),
        (0.3, None, [H, CNOT, SWAP, Z], ["h", "z", "swap", "cx"]),
        (0.3, [0, 1, 2], [CNOT, SWAP], ["cx", "swap"]),
        (0.3, [0, 1], None, ["h", "noisy_identity_0", "cx"]),
        (0.3, [0, 1], [H, CNOT, SWAP], ["h", "noisy_identity_0", "cx"]),
    ],
)
def test_phasedamping_qiskit_export(
    circuit: QCircuit,
    gamma: float,
    targets: list[int],
    gates: list[type[NativeGate]],
    expected_noisy_gates: list[str],
):
    from qiskit_aer.noise import NoiseModel as Qiskit_NoiseModel
    from qiskit_aer.noise.errors.standard_errors import phase_damping_error

    noise = PhaseDamping(gamma=gamma, targets=targets, gates=gates)

    assert isinstance(circuit, QCircuit)
    circuit.add(noise)

    qiskit_error = noise.to_other_language(Language.QISKIT)
    expected_error = phase_damping_error(gamma)

    qiskit_noise_model, _ = generate_qiskit_noise_model(circuit)
    noisy_instructions = qiskit_noise_model.noise_instructions

    assert qiskit_error == expected_error
    assert isinstance(qiskit_noise_model, Qiskit_NoiseModel)
    assert sorted(noisy_instructions) == sorted(expected_noisy_gates)
