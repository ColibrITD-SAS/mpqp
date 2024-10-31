import numpy as np
import numpy.typing as npt
import pytest

from mpqp import QCircuit
from mpqp.execution import IBMDevice
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.gates import *
from mpqp.measures import (
    Basis,
    BasisMeasure,
    ComputationalBasis,
    HadamardBasis,
    VariableSizeBasis,
)


@pytest.mark.parametrize(
    "init_vectors, size",
    [
        ([np.array([1, 0]), np.array([0, -1])], 1),
    ],
)
def test_right_init_basis(init_vectors: list[npt.NDArray[np.complex64]], size: int):
    b = Basis(init_vectors)
    assert b.nb_qubits == size
    assert (b.basis_vectors[i] == init_vectors[i] for i in range(len(init_vectors)))


@pytest.mark.parametrize(
    "init_vectors, part_of_error",
    [
        ([np.array([1, 0]), np.array([0, -1]), np.array([0, -1])], "number of vector"),
        ([np.array([1, 0])], "number of vector"),
        ([np.array([1, 0, 0]), np.array([0, -1])], "same size"),
        ([np.array([1, 1]), np.array([1, -1])], "normalized"),
        ([np.array([1, 1]) / np.sqrt(2), np.array([0, -1])], "orthogonal"),
    ],
)
def test_wrong_init_basis(
    init_vectors: list[npt.NDArray[np.complex64]], part_of_error: str
):
    with pytest.raises(ValueError) as error:
        Basis(init_vectors)
    assert part_of_error in error.exconly()


@pytest.mark.parametrize(
    "basis, size, is_initialized, result_pp",
    [
        (
            ComputationalBasis,
            3,
            True,
            (
                "Basis: [\n"
                "    [1, 0, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 1, 0, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 1, 0, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 1, 0, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 1, 0, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 1, 0, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 1, 0],\n"
                "    [0, 0, 0, 0, 0, 0, 0, 1]\n"
                "]\n"
            ),
        ),
        (
            ComputationalBasis,
            2,
            False,
            (
                "Basis: [\n"
                "    [1, 0, 0, 0],\n"
                "    [0, 1, 0, 0],\n"
                "    [0, 0, 1, 0],\n"
                "    [0, 0, 0, 1]\n"
                "]\n"
            ),
        ),
        (
            HadamardBasis,
            2,
            True,
            (
                "Basis: [\n"
                "    [0.5, 0.5, 0.5, 0.5],\n"
                "    [0.5, -0.5, 0.5, -0.5],\n"
                "    [0.5, 0.5, -0.5, -0.5],\n"
                "    [0.5, -0.5, -0.5, 0.5]\n"
                "]\n"
            ),
        ),
    ],
)
def test_basis_implementations(
    basis: type[VariableSizeBasis],
    size: int,
    is_initialized: bool,
    result_pp: str,
    capsys: pytest.CaptureFixture[str],
):
    if is_initialized:
        b = basis(size)
    else:
        b = basis()
        b.set_size(size)
    b.pretty_print()
    captured = capsys.readouterr()
    assert captured.out == result_pp


def test_run_with_custom_basis():
    theoretical_results = {'0': 0.5, '5': 0.5}

    v0 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    v1 = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    v2 = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    v3 = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    v4 = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    v5 = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    v6 = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    v7 = np.array([0, 0, 0, 0, 0, 0, 0, 1])

    custom_basis = Basis([v0, v1, v2, v3, v4, v5, v6, v7])
    circuit = QCircuit([H(0), CNOT(0, 1), SWAP(1, 2), Z(2)])
    circuit.add(BasisMeasure([0, 1, 2], basis=custom_basis, shots=1024))

    res = _run_single(circuit, IBMDevice.AER_SIMULATOR, {})

    actual_probabilities = {}
    for sample in res.samples:
        index = sample.index
        probability = sample.probability
        actual_probabilities[index] = probability

    for index, theoretical_prob in theoretical_results.items():
        actual_prob = actual_probabilities.get(int(index), 0)
        assert np.isclose(actual_prob, theoretical_prob, atol=0.02)
