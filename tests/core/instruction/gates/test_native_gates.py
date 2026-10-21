import numpy as np
import pytest
from sympy import Expr, I, pi, symbols

from mpqp.core.circuit import QCircuit
from mpqp.core.languages import Language
from mpqp.gates import *
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import cos, exp, matrix_eq, sin

theta: Expr
phi: Expr
k: Expr
theta, phi, k = symbols("θ φ k")

c, s, e = cos(theta), sin(theta), exp(1.0 * I * theta)
c2, s2, e2 = (
    cos(theta / 2),
    sin(theta / 2),
    exp(1.0 * I * theta / 2),
)
e_phi = exp(1.0 * I * phi)


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
            np.array([[c2, -1.0 * s2], [1.0 * s2, 1.0 * c2]]),
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
            np.array([[c2, -1j * s2], [-1j * s2, c2]]),
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
    "angle, result_matrix",
    [
        (0, np.eye(4)),
        (
            np.pi,
            np.array(
                [
                    [0, 0, 0, -1j],
                    [0, 0, -1j, 0],
                    [0, -1j, 0, 0],
                    [-1j, 0, 0, 0],
                ]
            ),
        ),
        (
            theta,
            np.array(
                [
                    [c2, 0, 0, -1j * s2],
                    [0, c2, -1j * s2, 0],
                    [0, -1j * s2, c2, 0],
                    [-1j * s2, 0, 0, c2],
                ]
            ),
        ),
    ],
)
def test_Rxx(angle: float, result_matrix: Matrix):
    assert matrix_eq(Rxx(angle, 0, 1).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "angle, result_matrix",
    [
        (0, np.eye(4)),
        (
            np.pi,
            np.array(
                [
                    [0, 0, 0, 1j],
                    [0, 0, -1j, 0],
                    [0, -1j, 0, 0],
                    [1j, 0, 0, 0],
                ]
            ),
        ),
        (
            theta,
            np.array(
                [
                    [c2, 0, 0, 1j * s2],
                    [0, c2, -1j * s2, 0],
                    [0, -1j * s2, c2, 0],
                    [1j * s2, 0, 0, c2],
                ]
            ),
        ),
    ],
)
def test_Ryy(angle: float, result_matrix: Matrix):
    assert matrix_eq(Ryy(angle, 0, 1).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "angle, result_matrix",
    [
        (0, np.eye(4)),
        (np.pi, np.diag([-1j, 1j, 1j, -1j])),
        (
            np.pi / 3,
            np.diag(
                [
                    np.exp(-1j * np.pi / 6),
                    np.exp(1j * np.pi / 6),
                    np.exp(1j * np.pi / 6),
                    np.exp(-1j * np.pi / 6),
                ]
            ),
        ),
        (theta, np.diag([1 / e2, e2, e2, 1 / e2])),  # pyright: ignore
    ],
)
def test_Rzz(angle: float, result_matrix: Matrix):
    assert matrix_eq(Rzz(angle, 0, 1).to_matrix(), result_matrix)


@pytest.mark.parametrize(
    "theta, phi, result_matrix",
    [
        (0, 0, np.eye(2)),
        (np.pi, 0, np.array([[0, -1j], [-1j, 0]])),
        (np.pi, np.pi / 2, np.array([[0, -1], [1, 0]])),
        (
            theta,
            phi,
            np.array(
                [
                    [c2, -1j * s2 / e_phi],
                    [-1j * s2 * e_phi, c2],
                ]
            ),
        ),
    ],
)
def test_PRX(theta: float, phi: float, result_matrix: Matrix):
    assert matrix_eq(PRX(theta, phi, 0).to_matrix(), result_matrix)


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


@pytest.fixture
def symbolic_circuits() -> list[tuple[QCircuit, dict[str, float], float]]:
    return [
        (
            QCircuit([Ry(2 * theta + phi, 0)]),
            {"θ": 0.2, "φ": 0.1},
            float(np.sin(0.25) ** 2),
        ),
        (
            QCircuit([Ry(theta, 0), Ry(theta, 0)]),
            {"θ": 0.2},
            float(np.sin(0.2) ** 2),
        ),
        (
            QCircuit([Ry(pi / 2, 0), Ry(theta, 0)]),
            {"θ": 0.2},
            float(np.sin((np.pi / 2 + 0.2) / 2) ** 2),
        ),
        (QCircuit([Ry(pi / 2, 0)]), {}, 0.5),
        (
            QCircuit([Ry(sin(theta), 0), Ry(theta, 0)]),
            {"θ": 0.2},
            float(np.sin((np.sin(0.2) + 0.2) / 2) ** 2),
        ),
    ]


@pytest.mark.provider("qiskit")
def test_qiskit_symbolic_expressions(
    symbolic_circuits: list[tuple[QCircuit, dict[str, float], float]],
):
    from qiskit.quantum_info import Statevector

    for circuit, values, expected_probability in symbolic_circuits:
        translated = circuit.to_other_language(Language.QISKIT)
        assert {parameter.name for parameter in translated.parameters} == set(values)
        assert len(translated.parameters) == len(values)
        bound = translated.assign_parameters(
            {parameter: values[parameter.name] for parameter in translated.parameters}
        )
        assert Statevector(bound).probabilities()[1] == pytest.approx(
            expected_probability
        )
        assert len(translated.parameters) == len(values)


@pytest.mark.provider("braket")
def test_braket_symbolic_expressions(
    symbolic_circuits: list[tuple[QCircuit, dict[str, float], float]],
):
    from braket.devices import LocalSimulator

    simulator = LocalSimulator()
    for circuit, values, expected_probability in symbolic_circuits:
        translated = circuit.to_other_language(Language.BRAKET)
        assert {str(parameter) for parameter in translated.parameters} == set(values)
        assert len(translated.parameters) == len(values)
        bound = translated.make_bound_circuit(values) # pyright: ignore[reportArgumentType]
        result = simulator.run(bound.probability(), shots=0).result() # pyright: ignore[reportAttributeAccessIssue]
        assert result.values[0][1] == pytest.approx(expected_probability) # pyright: ignore[reportAttributeAccessIssue]
        assert len(translated.parameters) == len(values)


@pytest.mark.provider("cirq")
def test_cirq_symbolic_expressions(
    symbolic_circuits: list[tuple[QCircuit, dict[str, float], float]],
):
    import cirq

    simulator = cirq.Simulator(dtype=np.complex128)
    for circuit, values, expected_probability in symbolic_circuits:
        translated = circuit.to_other_language(Language.CIRQ)
        assert cirq.parameter_names(translated) == set(values)
        bound = cirq.resolve_parameters(translated, values) # pyright: ignore[reportArgumentType]
        assert not cirq.is_parameterized(bound)
        result = simulator.simulate(bound)
        assert abs(result.final_state_vector[1]) ** 2 == pytest.approx(
            expected_probability
        )
        assert cirq.parameter_names(translated) == set(values)
