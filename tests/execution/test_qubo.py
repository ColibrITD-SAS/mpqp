import pytest

from mpqp.execution.vqa.qubo import *
from mpqp.tools.maths import matrix_eq

x = QuboAtom('x')
y = QuboAtom('y')
z = QuboAtom('z')
a = QuboAtom('a')

x0 = QuboAtom('x0')
x1 = QuboAtom('x1')
x2 = QuboAtom('x2')
x3 = QuboAtom('x3')


@pytest.mark.parametrize(
    "expr, res",
    [
        (2 * x, [(2, ['x'])]),
        (x * 2, [(2, ['x'])]),
        (x * 2 + 2, [(2, ['x']), (2, [])]),
        (x * 2 + 3 * y, [(2, ['x']), (3, ['y'])]),
        (-x, [(-1, ['x'])]),
        (
            4 * x * y,
            [(4, ['x', 'y'])],
        ),
        (3 * x * y - 2 * x + 3 * y, [(3, ['x', 'y']), (-2, ['x']), (3, ['y'])]),
        (
            2 * x + 3 * y + 4 * x * z + a,
            [(2, ['x']), (3, ['y']), (4, ['x', 'z']), (1, ['a'])],
        ),
        (
            3 * x * y + 4 * z + 6 * y * z + 2 * y,
            [(3, ['x', 'y']), (4, ['z']), (6, ['y', 'z']), (2, ['y'])],
        ),
        (2 * x + y + 3 * x + 3 * z, [(2, ['x']), (1, ['y']), (3, ['x']), (3, ['z'])]),
        (
            3 * y * x - 4 * x - 2 * y + 1,
            [(3, ['y', 'x']), (-4, ['x']), (-2, ['y']), (1, [])],
        ),
        (
            3 * x * y - 4 * x - 2 * y - 3 * z + 1,
            [(3, ['x', 'y']), (-4, ['x']), (-2, ['y']), (-3, ['z']), (1, [])],
        ),
        (3 * x + 2 * y + 100 * (x & y), [(3, ['x']), (2, ['y']), (100, ['x', 'y'])]),
        (
            3 * x + 2 * y + 100 * (x | y),
            [(3, ['x']), (2, ['y']), (100, ['x']), (100, ['y']), (-100, ['x', 'y'])],
        ),
        (
            3 * x + 2 * y + 100 * (x ^ y),
            [(3, ['x']), (2, ['y']), (100, ['x']), (100, ['y']), (-200, ['x', 'y'])],
        ),
        (4 * (2 + y), [(4, ['y']), (8, [])]),
    ],
)
def test_Qubo_coeffs(expr: Qubo, res: npt.NDArray[np.complex64]):
    assert expr.get_terms_and_coefs() == res


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
            3 * x * y - 4 * x - 2 * y - 3 * z + 1,
            np.array([[-4, 1.5, 0], [1.5, -2, 0], [0, 0, -3]]),
        ),
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
    assert matrix_eq(expr.weight_matrix()[0].astype(np.complex64), matrix)


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
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 12, 0],
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
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                ]
            ),
        ),
    ],
)
def test_Qubo_cost_hamiltonian(expression: Qubo, matrix: npt.NDArray[np.complex64]):
    assert matrix_eq(expression.to_cost_hamiltonian().matrix, matrix)


@pytest.mark.parametrize(
    "operand1, operand2, expected",
    [
        (x0, x1, x0 + x1),
        (1, x0, 1 + x0),
        (x0, 1, x0 + 1),
        (x0, 0, x0),
        (0, x0, x0),
        (x0, x0, 2 * x0),
        (x0, x0 * x0, 2 * x0),
        (x0 * x1, x1 * x0, 2 * x0 * x1),
        (x0 + x1, -x1, x0),
        (x0 * x1, x0 * x2, x0 * (x1 + x2)),
        (x0 + 2, 3, x0 + 5),
        (x0, ~x0, QuboConstant(1)),
        (~x0, x0, QuboConstant(1)),
        (x0, -x0, QuboConstant(0)),
        (-x0, x0, QuboConstant(0)),
        (2 * x0 * x1, -2 * x1 * x0, QuboConstant(0)),
        (x0 + x1, -(x0 + x1), QuboConstant(0)),
    ],
)
def test_Qubo_addition(operand1: Qubo, operand2: Qubo, expected: Qubo):
    addition = operand1 + operand2
    assert matrix_eq(
        addition.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        addition.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )


@pytest.mark.parametrize(
    "operand1, operand2, expected",
    [
        (x0, x1, x0 - x1),
        (1, x0, 1 - x0),
        (x0, 1, x0 - 1),
        (x0, 0, x0),
        (0, x0, -x0),
        (x0, x0, -2 * x0),
        (x0, x0 * x0, -2 * x0),
        (x0 * x1, x1 * x0, -2 * x0 * x1),
        (x0 + x1, -x1, x0 + 2 * x1),
        (x0 + x1, x1, x0),
        (x0 * x1, x0 * x2, x0 * (x1 - x2)),
        (x0 + 2, 3, x0 - 1),
    ],
)
def test_Qubo_subtraction(operand1: Qubo, operand2: Qubo, expected: Qubo):
    subtraction = operand1 - operand2
    assert matrix_eq(
        subtraction.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        subtraction.simplify().to_cost_hamiltonian().matrix,
        expected.simplify().to_cost_hamiltonian().matrix,
    )


@pytest.mark.parametrize(
    "operand1, operand2, expected",
    [
        (x0, x1, x0 * x1),
        (x0, x1, x0 & x1),
        (1, x0, x0),
        (x0, 1, x0),
        (-1, x0, -x0),
        (x0, -1, -x0),
        (x0, 0, QuboConstant(0)),
        (0, x0, QuboConstant(0)),
        (x0 + 2 * x1 + x0 * x1, 0, QuboConstant(0)),
        (x0, x0, x0 * x0),
        (x0, x0, x0),
        (x0 * x0, 1, x0),
        (x0, ~x0, QuboConstant(0)),
        (x0 + x1, 1, x0 + x1),
        (1, x0 + x1, x0 + x1),
        (x2, x0 + x1, x2 * (x0 + x1)),
        (x0 + x1, x2, x0 * x2 + x1 * x2),
        (x2, x0 - x1, x2 * (x0 - x1)),
        (x0 - x1, x2, x0 * x2 - x1 * x2),
        (x0, x1 + x2, x0 * (x1 + x2)),
        (x0, x1 + x2, x0 * x1 + x0 * x2),
        (-x0, x1 + x2, -x0 * x1 - x0 * x2),
        (-x0, x1 + x2, -x0 * (x1 + x2)),
        (-x0, x1, -x0 * x1),
        (x0, -x1, -x0 * x1),
        (x0 + x1, x1, x0 * x1 + x1),
        (x0 + x1, x1, x0 * x1 + x1 * x1),
        (x0 * x1, 3, 3 * x0 * x1),
        (3, x0 * x1, 3 * x0 * x1),
        (x0 * x1, -3, -3 * x0 * x1),
        (-3, x0 * x1, -3 * x0 * x1),
        (x0 + 2, 3, 3 * x0 + 6),
        (3, x0 + 2, 3 * x0 + 6),
        (
            0.7 * x0 + 0.3 * x1 - 0.02,
            0.7 * x0 + 0.3 * x1 - 0.02,
            0.462 * x0 + 0.078 * x1 + 0.42 * x0 * x1 + 0.0004,
        ),
        (
            3 * x0 + x0 - 2 * x1,
            0.7 * x0 + 0.3 * x1 - 0.02,
            2.72 * x0 - 0.56 * x1 - 0.2 * x0 * x1,
        ),
        (
            3 * x0 * x1 - 5 * x2 * x3 + 4 * x0 * x2,
            -1,
            -(3 * x0 * x1 - 5 * x2 * x3 + 4 * x0 * x2),
        ),
        (
            3 * x0 * x1 - 5 * x2 * x3 + 4 * x0 * x2,
            -1,
            -1 * (3 * x0 * x1 - 5 * x2 * x3 + 4 * x0 * x2),
        ),
        (
            3 * x0 * x1 - 5 * x2 * x3 + 4 * x0 * x2,
            -1,
            -3 * x0 * x1 + 5 * x2 * x3 - 4 * x0 * x2,
        ),
    ],
)
def test_Qubo_multiplication(operand1: Qubo, operand2: Qubo, expected: Qubo):
    multiplication = operand1 * operand2
    assert matrix_eq(
        multiplication.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        multiplication.simplify().to_cost_hamiltonian().matrix,
        expected.simplify().to_cost_hamiltonian().matrix,
    )


@pytest.mark.parametrize(
    "operand, expected",
    [
        (x0, -x0),
        (-x0, x0),
        (QuboConstant(3), QuboConstant(-3)),
        (QuboConstant(-3), QuboConstant(3)),
        (x0 * x1, -x0 * x1),
        (-x0 * x1, x0 * x1),
        (x0 + x1, -(x0 + x1)),
        (x0 + x1, -x0 - x1),
        (x0 - x1, -(x0 - x1)),
        (x0 * x0 + 3 * x1 - 2 * x1 * x2, -x0 - 3 * x1 + 2 * x1 * x2),
    ],
)
def test_Qubo_opposite_sign(operand: Qubo, expected: Qubo):
    opposite = -operand
    assert matrix_eq(
        opposite.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        opposite.simplify().to_cost_hamiltonian().matrix,
        expected.simplify().to_cost_hamiltonian().matrix,
    )
    if isinstance(operand, QuboAtom):
        assert str(operand) == str(-opposite)


@pytest.mark.parametrize(
    "operand1, operand2, expected",
    [
        (x0, x1, x0 * x1),
        (x0, x1, x1 & x0),
        (x0, ~x0, QuboConstant(0)),
        (~x0, x0, QuboConstant(0)),
    ],
)
def test_QuboAtom_AND(operand1: QuboAtom, operand2: QuboAtom, expected: Qubo):
    logical_and = operand1 & operand2
    assert matrix_eq(
        logical_and.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        logical_and.simplify().to_cost_hamiltonian().matrix,
        expected.simplify().to_cost_hamiltonian().matrix,
    )


@pytest.mark.parametrize(
    "operand1, operand2, expected",
    [
        (x0, x1, x0 + x1 - x0 * x1),
        (x0, x0, x0),
        (~x0, ~x0, ~x0),
        (x0, ~x0, QuboConstant(1)),
        (~x0, x0, QuboConstant(1)),
    ],
)
def test_QuboAtom_OR(operand1: QuboAtom, operand2: QuboAtom, expected: Qubo):
    logical_or = operand1 | operand2
    assert matrix_eq(
        logical_or.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        logical_or.simplify().to_cost_hamiltonian().matrix,
        expected.simplify().to_cost_hamiltonian().matrix,
    )


@pytest.mark.parametrize(
    "operand1, operand2, expected",
    [
        (x0, x0, QuboConstant(0)),
        (~x0, ~x0, QuboConstant(0)),
        (x0, x1, x0 + x1 - 2 * x0 * x1),
        (x0, ~x1, x0 + ~x1 - 2 * x0 * (~x1)),
        (~x0, x1, x0 + ~x1 - 2 * (~x0) * x1),
        (~x0, x0, QuboConstant(1)),
        (x0, ~x0, QuboConstant(1)),
    ],
)
def test_QuboAtom_XOR(operand1: QuboAtom, operand2: QuboAtom, expected: Qubo):
    logical_xor = operand1 ^ operand2
    assert matrix_eq(
        logical_xor.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        logical_xor.simplify().to_cost_hamiltonian().matrix,
        expected.simplify().to_cost_hamiltonian().matrix,
    )


@pytest.mark.parametrize(
    "operand, expected",
    [
        (x0, ~x0),
        (~x0, x0),
    ],
)
def test_QuboAtom_NOT(operand: QuboAtom, expected: Qubo):
    logical_not = ~operand
    assert matrix_eq(
        logical_not.to_cost_hamiltonian().matrix,
        expected.to_cost_hamiltonian().matrix,
    )
    assert matrix_eq(
        logical_not.simplify().to_cost_hamiltonian().matrix,
        expected.simplify().to_cost_hamiltonian().matrix,
    )
    if isinstance(expected, QuboAtom):
        assert str(operand) == str(~logical_not)


def test_Qubo_error():
    x = QuboAtom("x")
    y = QuboAtom("y")
    z = QuboAtom("z")
    with pytest.raises(ValueError):
        print(x * y * z)
    with pytest.raises(ValueError):
        print(x * (y * z))
    with pytest.raises(ValueError):
        print((x + 2) * y * (z + 1))
    with pytest.raises(ValueError):
        print(x * ((z * y) + x * 2))
