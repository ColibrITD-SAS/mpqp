import pytest

import numpy as np
import numpy.typing as npt

from mpqp.gates import Gate, Z, X, SWAP, CustomGate, UnitaryMatrix


@pytest.mark.parametrize(
    "gate, inverse",
    [
        (Z(0), Z(0)),
        (
            CustomGate(UnitaryMatrix(np.diag([1, 1j])), [0]),
            CustomGate(UnitaryMatrix(np.diag([1, -1j])), [0]),
        ),
    ],
)
def test_inverse(gate: Gate, inverse: Gate):
    assert gate.inverse().is_equivalent(inverse)


@pytest.mark.parametrize(
    "g1, g2",
    [
        (Z(0), Z(0)),
        (
            CustomGate(UnitaryMatrix(np.diag([1, -1])), [0]),
            Z(0),
        ),
        (
            CustomGate(UnitaryMatrix(np.diag([1.00000001, -1])), [0]),
            Z(0),
        ),
    ],
)
def test_is_equivalent(g1: Gate, g2: Gate):
    assert g1.is_equivalent(g2)


@pytest.mark.parametrize(
    "g1, g2",
    [
        (Z(0), X(0)),
        (
            CustomGate(UnitaryMatrix(np.diag([1, -1j])), [0]),
            Z(0),
        ),
    ],
)
def test_is_not_equivalent(g1: Gate, g2: Gate):
    assert not g1.is_equivalent(g2)


@pytest.mark.parametrize(
    "gate, pow, result_matrix",
    [
        (
            SWAP(0, 1),
            2,
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ),
        (
            SWAP(0, 1),
            -1,
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        ),
    ],
)
def test_power(gate: Gate, pow: float, result_matrix: npt.NDArray[np.complex64]):
    assert np.allclose(gate.power(pow).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "g1, g2, result_matrix",
    [
        (
            X(0),
            Z(0),
            np.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]]),
        ),
    ],
)
def test_tensor_product(g1: Gate, g2: Gate, result_matrix: npt.NDArray[np.complex64]):
    assert np.allclose(g1.tensor_product(g2).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "g1, g2, result_matrix",
    [
        (X(0), Z(0), np.array([[0, -1], [1, 0]])),
    ],
)
def test_product(g1: Gate, g2: Gate, result_matrix: npt.NDArray[np.complex64]):
    assert np.allclose(g1.product(g2).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "g1, scalar, result_matrix",
    [
        (X(0), 1j, np.array([[0, 1j], [1j, 0]])),
    ],
)
def test_scalar_product(
    g1: Gate, scalar: complex, result_matrix: npt.NDArray[np.complex64]
):
    assert np.allclose(g1.scalar_product(scalar).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "g1, g2, result_matrix",
    [
        (X(0), Z(0), np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
    ],
)
def test_add_product(g1: Gate, g2: Gate, result_matrix: npt.NDArray[np.complex64]):
    assert np.allclose(g1.plus(g2).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "g1, g2, result_matrix",
    [
        (X(0), Z(0), np.array([[-1, 1], [1, 1]]) / np.sqrt(2)),
    ],
)
def test_sub_product(g1: Gate, g2: Gate, result_matrix: npt.NDArray[np.complex64]):
    assert np.allclose(g1.minus(g2).to_matrix(), result_matrix)
