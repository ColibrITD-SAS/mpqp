import pytest
from mpqp.execution.vqa.qubo import *
from mpqp.tools.maths import matrix_eq

x = Qubo('x')
y = Qubo('y')
z = Qubo('z')
a = Qubo('a')

@pytest.mark.parametrize(
        "expr, res",
        [
            (
                2 * x,
                [(2, ['x'])]
            ),
            (
                x * 2,
                [(2, ['x'])]
            ),
            (
                x * 2 + 2,
                [(2, ['x']), (2, [])]
            ),
            (
                x * 2 + 3 * y,
                [(2, ['x']), (3, ['y'])]
            ),
            (
                4 * x * y,
                [(4, ['x', 'y'])],
            ),
            (
                3 * x * y - 2 * x + 3 * y,
                [(3, ['x', 'y']), (-2, ['x']), (3, ['y'])]
            ),
            (
                2 * x + 3 * y + 4 * x * z + a,
                [(2, ['x']), (3, ['y']), (4, ['x', 'z']), (1, ['a'])]
            ),
            (
                3 * x * y + 4 * z + 6 * y * z + 2 * y,
                [(3, ['x', 'y']), (4, ['z']), (6, ['y', 'z']), (2, ['y'])]
            ),
            (
                2 * x + y + 3 * x + 3 * z,
                [(2, ['x']), (1, ['y']), (3, ['x']), (3, ['z'])]
            ),
            (
                3 * y * x - 4 * x - 2 * y + 1,
                [(3, ['y', 'x']), (-4, ['x']), (-2, ['y']),(1, [])]
            ),
            (
                3 * x + 2 * y + 100 * (x & y),
                [(3, ['x']), (2, ['y']), (100, ['x', 'y'])]
            ),
            (
                3 * x + 2 * y + 100 * (x | y),
                [(3, ['x']), (2, ['y']), (100, ['x']), (100, ['y']), (-100, ['x', 'y'])]
            ),
            (
                3 * x + 2 * y + 100 * (x ^ y),
                [(3, ['x']), (2, ['y']), (100, ['x']), (100, ['y']), (-200, ['x', 'y'])]
            ),
            (
                4 * (2 + y),
                [(4, ['y']), (8, [])]
            )
        ]
)
def test_Qubo_coeffs(expr : Qubo, res : npt.NDArray[np.complex64]):
    assert(expr.get_coeffs(coeffs=[]) == res)


@pytest.mark.parametrize(
    "expr, matrix",
    [
        (2 * x, np.array([[2.0]])),
        (x * 2, np.array([[2.0]])),
        (x * 2 + 2, np.array([[2.0]])),
        (x * 2 + 3 * y, np.array([[2, 0], [0, 3]])),
        (4 * x * y, np.array([[0, 2], [2, 0]])),
        (3 * x * y - 2 * x + 3 * y, np.array([[-2.0, 1.5], [1.5, 3.0]])),
        (3 * y * x - 4 * x - 2 * y + 1, np.array([[-2, 1.5], [1.5, -4]])),
        (
            3 * x * y + 4 * z + 6 * y * z + 2 * y,
            np.array([[0, 1.5, 0], [1.5, 2, 3], [0, 3, 4]]),
        ),
        (2 * x + y + 3 * x + 3 * z, np.array([[5, 0, 0], [0, 1, 0], [0, 0, 3]])),
        (
            2 * x + 3 * y + 4 * x * z + a,
            np.array([[2, 0, 2, 0], [0, 3, 0, 0], [2, 0, 0, 0], [0, 0, 0, 1]]),
        ),
        (3 * x + 2 * y + 100 * (x & y), np.array([[3, 50], [50, 2]])),
        (3 * x + 2 * y + 100 * (x | y), np.array([[103, -50], [-50, 102]])),
        (3 * x + 2 * y + 100 * (x ^ y), np.array([[103, -100], [-100, 102]])),
    ],
)
def test_Qubo_weight_matrix(expr: Qubo, matrix: npt.NDArray[np.complex64]):
    assert matrix_eq(expr.create_matrix()[0].astype(np.complex64), matrix)


x0 = Qubo('x0')
x1 = Qubo('x1')
x2 = Qubo('x2')
x3 = Qubo('x3')

@pytest.mark.parametrize(
    "expression, matrix",
    [
        (2 * x0, np.array([[0, 0], [0, 2]])),
        (x0 * 2, np.array([[0, 0], [0, 2]])),
        (
            x0 * 2 + 2,
            np.array(
                [
                    [2, 0],
                    [0, 4],
                ]
            ),
        ),
        (
            x * 2 + 3 * y,
            np.array([[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 5]]),
        ),
        (4 * x * y, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4]])),
        (
            3 * x * y - 2 * x + 3 * y,
            np.array([[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, -2, 0], [0, 0, 0, 4]]),
        ),
        (
            3 * x0 * x1 - 4 * x0 - 2 * x1 + 1,
            np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -3, 0], [0, 0, 0, -2]]),
        ),
        (
            3 * x * y + 4 * z + 6 * y * z + 2 * y,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 12, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 4, 0, 0],
                    [0, 0, 0, 0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 0, 0, 0, 15],
                ]
            ),
        ),
        (
            2 * x + y + 3 * x + 3 * z,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0, 0],
                    [0, 0, 0, 0, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 8, 0, 0],
                    [0, 0, 0, 0, 0, 0, 6, 0],
                    [0, 0, 0, 0, 0, 0, 0, 9],
                ]
            ),
        ),
        (
            2 * x + 3 * y + 4 * x * z + a,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                ]
            ),
        ),
    ],
)
def test_Qubo_cost_hamiltonian(expression: Qubo, matrix: npt.NDArray[np.complex64]):
    assert matrix_eq(expression.to_cost_hamiltonian().matrix, matrix)
