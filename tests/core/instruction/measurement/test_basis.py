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
    expected_probabilities = {'0': 0.5, '5': 0.5}

    basis_vectors = [np.eye(8)[i] for i in range(8)]
    custom_basis = Basis(basis_vectors)

    circuit = QCircuit([H(0), CNOT(0, 1), SWAP(1, 2), Z(2)])
    circuit.add(BasisMeasure([0, 1, 2], basis=custom_basis, shots=1024))

    res = _run_single(circuit, IBMDevice.AER_SIMULATOR, {})
    actual_probabilities = {sample.index: sample.probability for sample in res.samples}

    assert all(
        abs((actual_probabilities.get(int(index)) or 0) - expected_prob) < 0.02
        for index, expected_prob in expected_probabilities.items()
    )
