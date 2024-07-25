import numpy as np
import pytest
from sympy import Expr, I, pi, symbols

from mpqp.gates import *
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import cos, exp, matrix_eq, sin

theta: Expr
k: Expr
theta, k = symbols("θ k")
c, s, e = cos(theta), sin(theta), exp(1.0 * I * theta)
c2, s2, e2 = (
    cos(theta / 2),  # pyright: ignore[reportOperatorIssue]
    sin(theta / 2),  # pyright: ignore[reportOperatorIssue]
    exp(1.0 * I * theta / 2),
)


@pytest.mark.parametrize(
    "angle, result_matrix",
    [
        (0, np.eye(2)),
        (np.pi, np.diag([1, -1])),
        (np.pi / 3, np.diag([1, 0.5 + 0.8660254j])),
        (theta, np.diag([1, e])),  # pyright: ignore
    ],
)
def test_P(angle: float, result_matrix: Matrix):
    assert matrix_eq(P(angle, 0).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "theta, phi, gamma, result_matrix",
    [
        (0, 0, 0, np.eye(2)),
        (0, 0, np.pi, np.diag([1, -1])),
        (0, np.pi, 0, np.diag([1, -1])),
        (np.pi, 0, 0, np.array([[0, -1], [1, 0]])),
        (
            2.75,
            1.23,
            3.08,
            np.array(
                [
                    [0.19454771 + 0.0j, 0.97903306 - 0.06037761j],
                    [0.32785147 + 0.92448072j, -0.07618831 - 0.1790088j],
                ]
            ),
        ),
        (
            theta,
            0,
            0,
            np.array([[c2, -1.0 * s2], [1.0 * s2, 1.0 * c2]]),  # pyright: ignore
        ),
    ],
)
def test_U(theta: float, phi: float, gamma: float, result_matrix: Matrix):
    assert matrix_eq(U(theta, phi, gamma, 0).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "angle, result_matrix",
    [
        (0, np.eye(2)),
        (np.pi, np.array([[0, -1j], [-1j, 0]])),
        (
            np.pi / 5,
            np.array(
                [
                    [0.95105652 + 0.0j, 0.0 - 0.30901699j],
                    [0.0 - 0.30901699j, 0.95105652 + 0.0j],
                ]
            ),
        ),
        (
            theta,
            np.array([[c2, -1j * s2], [-1j * s2, c2]]),  # pyright: ignore
        ),
    ],
)
def test_Rx(angle: float, result_matrix: Matrix):
    assert matrix_eq(Rx(angle, 0).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "angle, result_matrix",
    [
        (0, np.eye(2)),
        (np.pi, np.array([[0, -1], [1, 0]])),
        (
            np.pi / 5,
            np.array([[[0.95105652, -0.30901699], [0.30901699, 0.95105652]]]),
        ),
        (
            theta,
            np.array([[c2, -s2], [s2, c2]]),
        ),
    ],
)
def test_Ry(angle: float, result_matrix: Matrix):
    assert matrix_eq(Ry(angle, 0).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "angle, result_matrix",
    [
        (0, np.eye(2)),
        (np.pi, np.diag([-1j, 1j])),
        (np.pi / 5, np.diag([0.95105652 - 0.30901699j, 0.95105652 + 0.30901699j])),
        (theta, np.diag([1 / e2, e2])),  # pyright: ignore
    ],
)
def test_Rz(angle: float, result_matrix: Matrix):
    assert matrix_eq(Rz(angle, 0).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "angle_bin_pow, result_matrix",
    [
        (0, np.diag([1, 1])),
        (1, np.diag([1, -1])),
        (5, np.diag([1, np.exp(1j * np.pi / 2**4)])),
        (k, np.diag([1, exp(1.0 * 2 ** (1 - k) * I * pi)])),  # pyright: ignore
    ],
)
def test_Rk(angle_bin_pow: int, result_matrix: Matrix):
    assert matrix_eq(Rk(angle_bin_pow, 0).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "angle_bin_pow, result_matrix",
    [
        (0, np.diag([1, 1, 1, 1])),
        (1, np.diag([1, 1, 1, -1])),
        (5, np.diag([1, 1, 1, np.exp(1j * np.pi / 2**4)])),
        (k, np.diag([1, 1, 1, exp(1.0 * 2 ** (1 - k) * I * pi)])),  # pyright: ignore
    ],
)
def test_CRk(angle_bin_pow: int, result_matrix: Matrix):
    assert matrix_eq(CRk(angle_bin_pow, 0, 1).to_matrix(), result_matrix)
